#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

extern "C" {
#include "kernel.h"
}

enum {
	HALF_WINDOW = 2,
	BACKLOG_SIZE = 5,
	KNN_NELEMS = 5,
};

typedef struct {
	uint32_t v;
	double d;
} knnElem;

typedef struct {
	uint32_t *backlogs[BACKLOG_SIZE+1];
} KernelContext;

__device__ inline static double distance(int32_t u, int32_t v) {
	int r = (u & 0xff) - (v & 0xff),
		g = ((u >> 8) & 0xff) - ((v >> 8) & 0xff),
		b = ((u >> 16) & 0xff) - ((v >> 16) & 0xff);
	return sqrt(double(r * r + g * g + b * b));
}

#define in(i, j) in[i*w + j]
#define out(i, j) out[i*w + j]
#define FOR_WINDOW(i2, j2, i, j, h, w, wd) \
	for (i2 = max(0, i - wd); min(h, i2 < i + wd); i2++) \
		for (j2 = max(0, j - wd); min(w, j2 < j + wd); j2++)

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
		FOR_WINDOW(i2, j2, i, j, h, w, HALF_WINDOW) {
			double d = 0;
			FOR_WINDOW(i3, j3, i, j, h, w, HALF_WINDOW) {
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

enum {
	D0_THRESHOLD = 100,
	DSUM_THRESHOLD = 250,
};

__global__ void denoiseKernelAdaptiveTemporalAvg(
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
		int c = nbacklogs / 2, i, j, r = 0, g = 0, b = 0;
		int dsum = 0;
		uint32_t vc = backlogs[c][k];
		/*
		for (i = c; i >= 0; i--) {
			uint32_t v = backlogs[i][k];
			dsum += distance(v, lastv);
			if (distance(vc, v) > D0_THRESHOLD || dsum > DSUM_THRESHOLD) {
				break;
			}
			r += v & 0xff;
			g += (v >> 8) & 0xff;
			b += (v >> 16) & 0xff;
			lastv = v;
		}
		j = i;
		lastv = vc;
		for (i = c + 1; i < BACKLOG_SIZE && backlogs[i]; i++) {
			uint32_t v = backlogs[i][k];
			dsum += distance(v, lastv);
			if (distance(vc, v) > D0_THRESHOLD || dsum > DSUM_THRESHOLD) {
				break;
			}
			r += v & 0xff;
			g += (v >> 8) & 0xff;
			b += (v >> 16) & 0xff;
			lastv = v;
		}
		*/
		uint32_t vi_last = vc, vj_last = vc;
		for (i = j = c; i < nbacklogs && j >= 0; i++, j--) {
			uint32_t vi = backlogs[i][k], vj = backlogs[j][k];
			dsum += distance(vi, vi_last) + distance(vj, vj_last);
			if (distance(vc, vi) > D0_THRESHOLD ||
				distance(vc, vj) > D0_THRESHOLD ||
				dsum > DSUM_THRESHOLD) {
				break;
			}
			r += vi & 0xff;
			g += (vi >> 8) & 0xff;
			b += (vi >> 16) & 0xff;
			r += vj & 0xff;
			g += (vj >> 8) & 0xff;
			b += (vj >> 16) & 0xff;
			vi_last = vi;
			vj_last = vj;
		}
		r /= i - j;
		g /= i - j;
		b /= i - j;
		r &= 0xff;
		g &= 0xff;
		b &= 0xff;

		// r = g = b = 0xff * (i - j) / nbacklogs;
		out[k] = r | (g << 8) | (b << 16);
	}
}

__global__ void denoiseKernelZlokolica(
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
	}
}

__global__ void denoiseKernelKNN(
		KernelContext *ctx, int nbacklogs, uint32_t *out,
		int h, int w, int perThread) {
	int k = blockDim.x * blockIdx.x + threadIdx.x;
	int lower = k * perThread;
	int upper = (k + 1) * perThread;
	if (k == 256 * 32 - 1) {
		upper = h * w;
	}
	uint32_t **backlogs = ctx->backlogs;
	// Use at most 3 backlogs
	if (nbacklogs > 3) {
		nbacklogs = 3;
	}
	for (k = lower; k < upper; k++) {
		int i = k / w, j = k % w;
		uint32_t center;
		if (nbacklogs == 1) {
			center = backlogs[0][k];
		} else {
			center = backlogs[1][k];
		}
		knnElem elems[KNN_NELEMS+1];
		int nelems = 0;
		int i2, j2, t, e;
		FOR_WINDOW(i2, j2, i, j, h, w, 1) {
			for (t = 0; t < nbacklogs; t++) {
				uint32_t v = backlogs[t][i2*w+j2];
				double d = distance(v, center);
				// Insert v into elems
				for (e = nelems-1; e >= 0 && elems[e].d > d; e--) {
					/*
					if (e < KNN_NELEMS-1) {
						elems[e+1] = elems[e];
					}
					*/
					elems[e+1] = elems[e];
				}
				if (1) {
				//if (e < KNN_NELEMS-1) {
					elems[e+1].v = v;
					elems[e+1].d = d;
				}
				if (nelems < KNN_NELEMS) {
					nelems++;
				}
			}
		}
		uint32_t r = 0, g = 0, b = 0;
		for (e = 0; e < nelems; e++) {
			uint32_t v = elems[e].v;
			r += v & 0xff;
			g += (v >> 8) & 0xff;
			b += (v >> 16) & 0xff;
		}
		r /= e;
		g /= e;
		b /= e;
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
	case KNN:
	case ZLOKOLICA:
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
		int i;
		for (i = 0; i < BACKLOG_SIZE && ctx->backlogs[i]; i++)
			;
		switch (m) {
		case TEMPORAL_AVG:
			denoiseKernelTemporalAvg<<<blocksPerGrid,
				threadsPerBlock>>>(dctx, dout, h, w, perThread);
			break;
		case KNN:
			denoiseKernelKNN<<<blocksPerGrid,
				threadsPerBlock>>>(dctx, i, dout, h, w, perThread);
			break;
		case ZLOKOLICA:
			denoiseKernelZlokolica<<<blocksPerGrid,
				threadsPerBlock>>>(dctx, dout, h, w, perThread);
			break;
		case ADAPTIVE_TEMPORAL_AVG:
			denoiseKernelAdaptiveTemporalAvg<<<blocksPerGrid,
				threadsPerBlock>>>(dctx, i, dout, h, w, perThread);
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
