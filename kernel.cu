#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

extern "C" {
#include "kernel.h"
}

enum {
	HALF_WINDOW = 2,
	BACKLOG_SIZE = 5,
	NDIFFS = 2,
	KNN_NELEMS = 10,
	KNN_FRAMES = 5,
};

typedef struct {
	uint32_t v;
	double d;
} knnElem;

typedef struct {
	uint32_t *backlogs[BACKLOG_SIZE+1];
	uint32_t *diffs[NDIFFS+1];
	uint32_t **dbacklogs, **ddiffs;
} KernelContext;

__device__ inline static double distance(int32_t u, int32_t v) {
	int r = (u & 0xff) - (v & 0xff),
		g = ((u >> 8) & 0xff) - ((v >> 8) & 0xff),
		b = ((u >> 16) & 0xff) - ((v >> 16) & 0xff);
	return sqrt(double(r * r + g * g + b * b));
}

__device__ uint32_t diff(uint32_t u, uint32_t v) {
	double d = distance(u, v);
	return d * 0x10000;
}

__device__ uint32_t lum(uint32_t u) {
	int r = u & 0xff,
		g = (u >> 8) & 0xff,
		b = (u >> 16) & 0xff;
	int l = 0.299 * r + 0.587 * g + 0.114 * b;
	return l > 0xff ? 0xff : l;
}

__device__ uint32_t diffPixel(uint32_t u, uint32_t v) {
	int ur = u & 0xff, ug = (u >> 8) & 0xff, ub = (u >> 16) & 0xff;
	int vr = v & 0xff, vg = (v >> 8) & 0xff, vb = (v >> 16) & 0xff;
	/*
	int r = (ur - vr + 0xff) / 2;
	int g = (ug - vg + 0xff) / 2;
	int b = (ub - vb + 0xff) / 2;
	*/
	int r = abs(ur - vr);
	int g = abs(ug - vg);
	int b = abs(ub - vb);
	return r | (g << 8) | (b << 16);
	/*
	int ur = u & 0xff, ug = (u >> 8) & 0xff, ub = (u >> 16) & 0xff;
	int vr = v & 0xff, vg = (v >> 8) & 0xff, vb = (v >> 16) & 0xff;
	int prod = ur * vr + ug * vg + ub * vb;
	double um = sqrt(double(ur*ur + ug*ug + ub*ub));
	double vm = sqrt(double(vr*vr + vg*vg + vb*vb));
	double arg = acos(prod / (um * vm));
	int p = arg / (3.1415926/2) * 0xff;
	if (p > 0xff) {
		p = 0xff;
	}
	*/
	/*
	int p = lum(u) - lum(v);
	if (p < 0) {
		p = -p;
	}
	*/
	/*
	double d = distance(u, v);
	uint32_t p = d / 1.732;
	if (p > 0xff) {
		p = 0xff;
	}
	*/
	// return p | (p << 8) | (p << 16);
}

__global__ static void updateDiff(
		uint32_t *d, uint32_t *b0, uint32_t *b1, int w) {
	if (!(b0 && b1)) {
		return;
	}
	int k = (blockDim.x * blockIdx.x + threadIdx.x) * w +
		blockDim.y * blockIdx.y + threadIdx.y;
	d[k] = diff(b0[k], b1[k]);
}

#define in(i, j) in[(i)*w + (j)]
#define out(i, j) out[(i)*w + (j)]
#define FOR_WINDOW(i2, j2, i, j, h, w, wd) \
	for (i2 = max(0, i - wd); min(h, i2 < i + wd); i2++) \
		for (j2 = max(0, j - wd); min(w, j2 < j + wd); j2++)

__global__ void denoiseKernelSpatial(
		const uint32_t *in, uint32_t *out, int h, int w) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;
	int k = i * w + j;

	double dmin = INFINITY;
	int i2, j2, i3, j3;
	FOR_WINDOW(i2, j2, i, j, h, w, HALF_WINDOW) {
		double d = 0;
		FOR_WINDOW(i3, j3, i, j, h, w, HALF_WINDOW) {
			d += distance(in(i2, j2), in(i3, j3));
		}
		if (dmin > d) {
			dmin = d;
			out[k] = in(i2, j2);
		}
	}
}

__global__ void denoiseKernelSobel(
		const uint32_t *in, int frame, double *sout, uint32_t *out, int h,
		int w) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	int k = x * w + y;

	if (x == 0 || x == h-1 || y == 0 || y == w-1) {
		out[k] = 0;
		if (sout != 0) {
			sout[k] = 0;
		}
		return;
	}
	int gx = lum(in(x-1, y+1)) + 2 * lum(in(x, y+1)) + lum(in(x+1, y+1))
		   - lum(in(x-1, y-1)) - 2 * lum(in(x, y-1)) - lum(in(x+1, y-1));
	int gy = lum(in(x-1, y-1)) + 2 * lum(in(x-1, y)) + lum(in(x-1, y+1))
		   - lum(in(x+1, y-1)) - 2 * lum(in(x+1, y)) - lum(in(x+1, y+1));
	/*
	int g = sqrt((gx * gx + gy * gy) / 50.0);
	*/
	int g = ((gx * gx + gy * gy) > 2000) * 0xff;
	/*
	if (g > 0x5) {
		g = 0xff;
	} else {
		g = 0;
	}
	*/
	out[k] = g | (g << 8) | (g << 16);
	if (sout != 0) {
		sout[k] = sqrt(double(gx * gx + gy * gy));
	}
}

__global__ void denoiseKernelTemporalAvg(
		uint32_t **backlogs, uint32_t *out, int w) {
	int k = (blockDim.x * blockIdx.x + threadIdx.x) * w +
		blockDim.y * blockIdx.y + threadIdx.y;

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

enum {
	D0_THRESHOLD = 100,
	DSUM_THRESHOLD = 250,
};

__global__ void denoiseKernelAdaptiveTemporalAvg(
		uint32_t **backlogs, int nbacklogs, uint32_t *out, int w) {
	int k = (blockDim.x * blockIdx.x + threadIdx.x) * w +
		blockDim.y * blockIdx.y + threadIdx.y;

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

	//r = g = b = 0xff * (i - j) / nbacklogs;
	out[k] = r | (g << 8) | (b << 16);
}

enum {
	Z_THRM = 20 * 0x10000,
	Z_THRD = 10,
};

__global__ void denoiseKernelMotion(
		uint32_t **diffs, uint32_t *out, int h, int w) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;
	int k = i * w + j;

	int i2, j2;
	int t;
	// Motion detection
	for (t = 0; t < NDIFFS && diffs[t]; t++) {
		if (diffs[t][k] < Z_THRM) {
			continue;
		}
		int excess = 0;
		FOR_WINDOW(i2, j2, i, j, h, w, 1) {
			int k2 = i2 * w + j2;
			if (k2 != k && diffs[t][k2] >= Z_THRM) {
				excess++;
				if (excess == 2) {
					goto after_md;
				}
			}
		}
	}
after_md:
	int v = 0xff * (NDIFFS - t) / NDIFFS;
	out[k] = v | (v << 8) | (v << 16);
}

__global__ void denoiseKernelZlokolica(
		uint32_t **backlogs, uint32_t **diffs, int nbacklogs, uint32_t *out,
		int h, int w) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;
	int k = i * w + j;

	uint32_t *in = backlogs[0];

	if (nbacklogs == 1) {
		out[k] = backlogs[0][k];
		return;
	}

	int i2, j2;
	int nf;
	// Motion detection
	for (nf = 0; nf < NDIFFS && diffs[nf]; nf++) {
		if (diffs[nf][k] < Z_THRM) {
			continue;
		}
		int excess = 0;
		FOR_WINDOW(i2, j2, i, j, h, w, 1) {
			int k2 = i2 * w + j2;
			if (k2 != k && diffs[nf][k2] >= Z_THRM) {
				excess++;
				if (excess == 2) {
					goto after_md;
				}
			}
		}
	}
after_md:
	if (nf < nbacklogs-1) {
		nf++;
	}

	// Edge detection
	bool edge = false;
	if (i > 0 && i < h-1 && j > 0 && j < w-1) {
		int gi = lum(in(i-1, j+1)) + 2 * lum(in(i, j+1)) + lum(in(i+1, j+1))
			   - lum(in(i-1, j-1)) - 2 * lum(in(i, j-1)) - lum(in(i+1, j-1));
		int gj = lum(in(i-1, j-1)) + 2 * lum(in(i-1, j)) + lum(in(i-1, j+1))
			   - lum(in(i+1, j-1)) - 2 * lum(in(i+1, j)) - lum(in(i+1, j+1));
		edge = (gi * gi + gj * gj) > 1250;
	}

	uint32_t center = backlogs[0][k];
	knnElem elems[KNN_NELEMS+1];
	int nelems = 0;
	int t, e;
	FOR_WINDOW(i2, j2, i, j, h, w, 1) {
		for (t = 0; t < nf; t++) {
			uint32_t v = backlogs[t][i2*w+j2];
			double d = distance(v, center);
			if (edge && d > Z_THRD) {
				continue;
			}
			// Insert v into elems
			for (e = nelems-1; e >= 0 && elems[e].d > d; e--) {
				if (e < KNN_NELEMS-1) {
					elems[e+1] = elems[e];
				}
			}
			if (e < KNN_NELEMS-1) {
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

__global__ void denoiseKernelDiff(
		uint32_t **backlogs, int nbacklogs, uint32_t *out, int w) {
	int k = (blockDim.x * blockIdx.x + threadIdx.x) * w +
		blockDim.y * blockIdx.y + threadIdx.y;

	if (nbacklogs > 1) {
		out[k] = diffPixel(backlogs[0][k], backlogs[1][k]);
	} else {
		out[k] = 0;
	}
}

__global__ void denoiseKernelKNN(
		uint32_t **backlogs, int nbacklogs, uint32_t *out, int h, int w) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;
	int k = i * w + j;

	// Use at most KNN_FRAMES frames
	if (nbacklogs > KNN_FRAMES) {
		nbacklogs = KNN_FRAMES;
	}

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
				if (e < KNN_NELEMS-1) {
					elems[e+1] = elems[e];
				}
			}
			if (e < KNN_NELEMS-1) {
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

void denoise(void *p, int m, const uint32_t *in, uint32_t *out, int h, int w) {
	static int frame = 0;
	KernelContext *ctx = (KernelContext*) p;
	dim3 threadsPerBlock(8, 8);
	dim3 blocksPerGrid(h / 8, w / 8);

	int size = h * w * sizeof(uint32_t);
	uint32_t *dout;
	cudaMalloc(&dout, size);

	switch (m) {
	case SPATIAL:
	case SOBEL:
		uint32_t *din;
		cudaMalloc(&din, size);
		cudaMemcpy(din, in, size, cudaMemcpyHostToDevice);
		switch (m) {
		case SPATIAL:
			denoiseKernelSpatial<<<blocksPerGrid, threadsPerBlock>>>(
					din, dout, h, w);
			break;
		case SOBEL:
			double *dsobel = 0;
			denoiseKernelSobel<<<blocksPerGrid, threadsPerBlock>>>(
					din, frame, dsobel, dout, h, w);
			break;
		}
		cudaFree(din);
		break;
	case DIFF:
	case TEMPORAL_AVG:
	case ADAPTIVE_TEMPORAL_AVG:
	case KNN:
	case ZLOKOLICA:
	case MOTION:
		// Cyclically right shift backlog
		for (int i = BACKLOG_SIZE; i > 0; i--) {
			ctx->backlogs[i] = ctx->backlogs[i-1];
		}
		ctx->backlogs[0] = ctx->backlogs[BACKLOG_SIZE];
		if (ctx->backlogs[0] == 0) {
			cudaMalloc(&ctx->backlogs[0], size);
		}
		cudaMemcpy(ctx->backlogs[0], in, size, cudaMemcpyHostToDevice);

		switch (m) {
		case ZLOKOLICA:
		case MOTION:
			for (int i = NDIFFS; i > 0; i--) {
				ctx->diffs[i] = ctx->diffs[i-1];
			}
			ctx->diffs[0] = ctx->diffs[NDIFFS];
			if (ctx->diffs[0] == 0) {
				cudaMalloc(&ctx->diffs[0], size);
			}
		}

		cudaMemcpy(ctx->dbacklogs, ctx->backlogs, sizeof(ctx->backlogs),
				cudaMemcpyHostToDevice);
		cudaMemcpy(ctx->ddiffs, ctx->diffs, sizeof(ctx->diffs),
				cudaMemcpyHostToDevice);

		int i;
		for (i = 0; i < BACKLOG_SIZE && ctx->backlogs[i]; i++)
			;

		switch (m) {
		case DIFF:
			denoiseKernelDiff<<<blocksPerGrid,
				threadsPerBlock>>>(ctx->dbacklogs, i, dout, w);
			break;
		case TEMPORAL_AVG:
			denoiseKernelTemporalAvg<<<blocksPerGrid,
				threadsPerBlock>>>(ctx->dbacklogs, dout, w);
			break;
		case KNN:
			denoiseKernelKNN<<<blocksPerGrid,
				threadsPerBlock>>>(ctx->dbacklogs, i, dout, h, w);
			break;
		case ZLOKOLICA:
			updateDiff<<<blocksPerGrid, threadsPerBlock>>>(
					ctx->diffs[0], ctx->backlogs[0], ctx->backlogs[1], w);
			denoiseKernelZlokolica<<<blocksPerGrid,
				threadsPerBlock>>>(ctx->dbacklogs, ctx->ddiffs, i, dout, h, w);
			break;
		case MOTION:
			updateDiff<<<blocksPerGrid, threadsPerBlock>>>(
					ctx->diffs[0], ctx->backlogs[0], ctx->backlogs[1], w);
			denoiseKernelMotion<<<blocksPerGrid,
				threadsPerBlock>>>(ctx->ddiffs, dout, h, w);
			break;
		case ADAPTIVE_TEMPORAL_AVG:
			denoiseKernelAdaptiveTemporalAvg<<<blocksPerGrid,
				threadsPerBlock>>>(ctx->dbacklogs, i, dout, w);
			break;
		}
		break;
	}

	cudaMemcpy(out, dout, size, cudaMemcpyDeviceToHost);
	cudaFree(dout);
	frame++;
}

void *kernelInit() {
	KernelContext *ctx = (KernelContext*) calloc(1, sizeof(KernelContext));
	cudaMalloc(&ctx->dbacklogs, sizeof(ctx->backlogs));
	cudaMalloc(&ctx->ddiffs, sizeof(ctx->diffs));
	return ctx;
}

void kernelFinalize(void *p) {
	KernelContext *ctx = (KernelContext*) p;
	for (int i = 0; i < BACKLOG_SIZE; i++) {
		cudaFree(ctx->backlogs[i]);
	}
	for (int i = 0; i < NDIFFS; i++) {
		cudaFree(ctx->diffs[i]);
	}
	cudaFree(ctx->dbacklogs);
	cudaFree(ctx->ddiffs);
	free(ctx);
}
