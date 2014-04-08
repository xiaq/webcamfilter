#include <stdint.h>
#include <stdlib.h>

extern "C" {
#include "kernel.h"
}

enum {
	HALF_WINDOW = 2,
	BACKLOG_SIZE = 5,
};

typedef struct {
	uint32_t *backlogs[BACKLOG_SIZE+1];
} KernelContext;

__device__ inline static double distance(int32_t u, int32_t v) {
	int r = u & 0xff - v & 0xff,
		g = (u >> 8) & 0xff - (v >> 8) & 0xff,
		b = (u >> 16) & 0xff - (v >> 16) & 0xff;
	return sqrt(double(r * r + g * g + b * b));
}

#define in(i, j) in[i*w + j]
#define out(i, j) out[i*w + j]
#define FOR_WINDOW(i2, j2, i, j, h, w) \
	for (i2 = max(0, i - HALF_WINDOW); min(h, i2 < i + HALF_WINDOW); i2++) \
		for (j2 = max(0, j - HALF_WINDOW); min(w, j2 < j + HALF_WINDOW); j2++)

__global__ void denoiseKernelSpatial(
		const uint32_t *in, uint32_t *out, int h,
		int w, int perThread) {
	int k = blockDim.x * blockIdx.x + threadIdx.x;
	int lower = k * perThread;
	int upper = (k + 1) * perThread;
	if (k == 256 * 32 - 1) {
		upper = h * w;
	}
	for (k = lower; k < upper; k++) {
		int i = k / w, j = k % w;
		double dmin = INFINITY;
		int i2, j2, i3, j3;
		FOR_WINDOW(i2, j2, i, j, h, w) {
			double d = 0;
			FOR_WINDOW(i3, j3, i, j, h, w) {
				d += distance(in(i2, j2), in(i3, j3));
			}
			if (dmin > d) {
				dmin = d;
				out(i, j) = in(i2, j2);
			}
		}
	}
}
#undef in
#undef out

__global__ void denoise_kernel_temporal_vmf(
		KernelContext *ctx, int nbacklogs, uint32_t *out,
		int h, int w, int perThread) {
	int k = blockDim.x * blockIdx.x + threadIdx.x;
	int lower = k * perThread;
	int upper = (k + 1) * perThread;
	if (k == 256 * 32 - 1) {
		upper = h * w;
	}
	uint32_t **backlogs = ctx->backlogs;
	for (k = lower; k < upper; k++) {
		double dmin = INFINITY;
		int i, j;
		for (i = 0; i < nbacklogs; i++) {
			double d = 0;
			for (j = 0; j < nbacklogs; j++) {
				if (i != j) {
					d += distance(backlogs[i][k], backlogs[j][k]);
				}
			}
			if (dmin > d) {
				dmin = d;
				out[k] = backlogs[i][k];
			}
		}
	}
}

enum {
	D0_THRESHOLD = 200,
	DSUM_THRESHOLD = 400,
};

__global__ void denoiseKernelTemporalAvg(
		KernelContext *ctx, uint32_t *out,
		int h, int w, int perThread) {
	int k = blockDim.x * blockIdx.x + threadIdx.x;
	int lower = k * perThread;
	int upper = (k + 1) * perThread;
	if (k == 256 * 32 - 1) {
		upper = h * w;
	}
	uint32_t **backlogs = ctx->backlogs;
	for (k = lower; k < upper; k++) {
		int i, r = 0, g = 0, b = 0;
		for (i = 0; i < BACKLOG_SIZE && backlogs[i]; i++) {
			uint32_t v = backlogs[i][k];
			r += v & 0xff;
			g += (v >> 8) & 0xff;
			b += (v >> 16) & 0xff;
			// lastv = v;
		}
		r /= i;
		g /= i;
		b /= i;
		r &= 0xff;
		g &= 0xff;
		b &= 0xff;
		out[k] = r | (g << 8) | (b << 16);
	}
}

__global__ void denoiseKernelAdaptiveTemporalAvg(
		KernelContext *ctx, uint32_t *out,
		int h, int w, int perThread) {
	int k = blockDim.x * blockIdx.x + threadIdx.x;
	int lower = k * perThread;
	int upper = (k + 1) * perThread;
	if (k == 256 * 32 - 1) {
		upper = h * w;
	}
	uint32_t **backlogs = ctx->backlogs;
	for (k = lower; k < upper; k++) {
		int i, r = 0, g = 0, b = 0;
		uint32_t v0 = backlogs[0][k], lastv = backlogs[0][k];
		int dsum = 0;
		for (i = 0; i < BACKLOG_SIZE && backlogs[i]; i++) {
			uint32_t v = backlogs[i][k];
			dsum += distance(v, lastv);
			if (distance(v0, v) > D0_THRESHOLD || dsum > DSUM_THRESHOLD) {
				break;
			}
			r += v & 0xff;
			g += (v >> 8) & 0xff;
			b += (v >> 16) & 0xff;
			lastv = v;
		}
		// r = g = b = 0xff * (i-1) / (BACKLOG_SIZE-1);
		r /= i;
		g /= i;
		b /= i;
		r &= 0xff;
		g &= 0xff;
		b &= 0xff;
		out[k] = r | (g << 8) | (b << 16);
	}
}

void denoise(void *p, int m, const uint32_t *in, uint32_t *out, int h, int w) {
	KernelContext *ctx = (KernelContext*) p;
	int threadsPerBlock = 256;
	int blocksPerGrid = 32;
	int perThread = h * w / (threadsPerBlock * blocksPerGrid);

	int size = h * w * sizeof(uint32_t);
	uint32_t *dout;
	cudaMalloc(&dout, size);

	switch (m) {
	case SPATIAL:
		uint32_t *din;
		cudaMalloc(&din, size);
		cudaMemcpy(din, in, size, cudaMemcpyHostToDevice);
		denoiseKernelSpatial<<<blocksPerGrid, threadsPerBlock>>>(
				din, dout, h, w, perThread);
		cudaFree(din);
		break;
	case TEMPORAL_AVG:
	case ADAPTIVE_TEMPORAL_AVG:
		// Cyclically right shift backlog
		for (int i = BACKLOG_SIZE; i > 0; i--) {
			ctx->backlogs[i] = ctx->backlogs[i-1];
		}
		ctx->backlogs[0] = ctx->backlogs[BACKLOG_SIZE];
		if (ctx->backlogs[0] == 0) {
			cudaMalloc(&ctx->backlogs[0], size);
		}
		cudaMemcpy(ctx->backlogs[0], in, size, cudaMemcpyHostToDevice);

		KernelContext *dctx;
		cudaMalloc(&dctx, sizeof(KernelContext));
		cudaMemcpy(dctx, ctx, sizeof(KernelContext), cudaMemcpyHostToDevice);
		switch (m) {
		case TEMPORAL_AVG:
			denoiseKernelTemporalAvg<<<blocksPerGrid,
				threadsPerBlock>>>(dctx, dout, h, w, perThread);
			break;
		case ADAPTIVE_TEMPORAL_AVG:
			denoiseKernelAdaptiveTemporalAvg<<<blocksPerGrid,
				threadsPerBlock>>>(dctx, dout, h, w, perThread);
			break;
		}
		cudaFree(dctx);
		break;
	}

	cudaMemcpy(out, dout, size, cudaMemcpyDeviceToHost);
	cudaFree(dout);
}

void *kernelInit() {
	KernelContext *ctx = (KernelContext*) calloc(1, sizeof(KernelContext));
	return ctx;
}

void kernelFinalize(void *p) {
	KernelContext *ctx = (KernelContext*) p;
	for (int i = 0; i < BACKLOG_SIZE; i++) {
		cudaFree(ctx->backlogs[i]);
	}
	free(ctx);
}
