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

enum {
	Z_THRM = 10 * 0x10000,
	Z_THRD = 10,
};

typedef struct {
	uint32_t v;
	uint32_t d;
} knnElem;

typedef struct {
	uint32_t *backlogs[BACKLOG_SIZE+1];
	uint32_t *diffs[NDIFFS+1];
	bool *motions[NDIFFS+1];
	bool *m0;

	uint32_t **dbacklogs, **ddiffs;
	bool **dmotions;
} KernelContext;

__device__ __host__ uint32_t lum(uint32_t u) {
	int r = u & 0xff,
		g = (u >> 8) & 0xff,
		b = (u >> 16) & 0xff;
	int l = 0.299 * r + 0.587 * g + 0.114 * b;
	return l > 0xff ? 0xff : l;
}

/*
__device__ inline static double distance(int32_t u, int32_t v) {
	int r = (u & 0xff) - (v & 0xff),
		g = ((u >> 8) & 0xff) - ((v >> 8) & 0xff),
		b = ((u >> 16) & 0xff) - ((v >> 16) & 0xff);
	return sqrt(double(r * r + g * g + b * b));
}
*/
__device__ __host__ inline static uint32_t distance(int32_t u, int32_t v) {
	return abs(int(lum(u)) - int(lum(v)));
}

__device__ uint32_t diff(uint32_t u, uint32_t v) {
	/*
	double d = distance(u, v);
	*/
	int d = abs(int(lum(u)) - int(lum(v)));
	return d * 0x10000;
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
	for (i2 = max(0, i - wd); i2 < min(h, i + wd); i2++) \
		for (j2 = max(0, j - wd); j2 < min(w, j + wd); j2++)

__global__ static void updateMotion(
		bool *m, uint32_t *d, int h, int w) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;
	int k = i * w + j;

	m[k] = false;
	if (d[k] < Z_THRM) {
		return;
	}

	int i2, j2;
	int excess = 0;
	FOR_WINDOW(i2, j2, i, j, h, w, 1) {
		int k2 = i2 * w + j2;
		if (k2 != k && d[k2] >= Z_THRM) {
			excess++;
			if (excess == 2) {
				m[k] = true;
				return;
			}
		}
	}
}

__global__ static void dilateMotion(
		bool *m, bool *m0, int h, int w) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;
	int k = i * w + j;

	if (m0[k]) {
		m[k] = true;
	} else {
		int i2, j2;
		FOR_WINDOW(i2, j2, i, j, h, w, 3) {
			int k2 = i2 * w + j2;
			if (m0[k2]) {
				m[k] = true;
				return;
			}
		}
		m[k] = false;
	}
}

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

__global__ void denoiseKernelSpatialKNN(
		const uint32_t *in, uint32_t *out, int h, int w) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;
	int k = i * w + j;
	uint32_t center = in[k];
	int nelems = 0;
	knnElem elems[KNN_NELEMS+1];
	int i2, j2, e;
	FOR_WINDOW(i2, j2, i, j, h, w, 3) {
		int m = abs(i2-i) + abs(j2-j);
		if (m == 1 || m == 2) {
			continue;
		}
		uint32_t v = in[i2*w+j2];
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

__global__ void denoiseKernelMotion(
		bool **motions, uint32_t *out, int h, int w) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;
	int k = i * w + j;

	/*
	int t;
	// Motion detection
	for (t = 0; t < NDIFFS && motions[t] && !motions[t][k]; t++)
		;
	int v = 0xff * (NDIFFS - t) / NDIFFS;
	*/
	int v = 0;
	if (motions[0] && motions[0][k]) {
		v = 0xff;
	}
	out[k] = v | (v << 8) | (v << 16);
}

__global__ void denoiseKernelAKNN(
		uint32_t **backlogs, bool **motions, int nbacklogs, uint32_t *out,
		int h, int w) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;
	int k = i * w + j;

	uint32_t *in = backlogs[0];

	if (nbacklogs == 1) {
		out[k] = backlogs[0][k];
		return;
	}

	int nf;
	// Motion detection
	for (nf = 0; nf < NDIFFS && motions[nf] && !motions[nf][k]; nf++)
		;
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
	int i2, j2;
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

// Do a cyclic shift on array elements. The array must have at least (n+1)
// slots.
template <class T>
void ror(T *a, int n) {
	for (int i = n; i > 0; i--) {
		a[i] = a[i-1];
	}
	a[0] = a[n];
}

void denoise(void *p, int m, const uint32_t *in, uint32_t *out, int h, int w) {
	static int frame = 0;
	KernelContext *ctx = (KernelContext*) p;
	dim3 blockDim(4, 32);
	dim3 gridDim(h / blockDim.x, w / blockDim.y);

	int size = h * w * sizeof(uint32_t);
	uint32_t *dout;
	cudaMalloc(&dout, size);

	switch (m) {
	case SPATIAL:
	case SPATIAL_KNN:
	case SOBEL:
		uint32_t *din;
		cudaMalloc(&din, size);
		cudaMemcpy(din, in, size, cudaMemcpyHostToDevice);
		switch (m) {
		case SPATIAL:
			/*
			int i, j;
			for (i = 0; i < h; i++) {
				for (j = 0; j < w; j++) {
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
			}
			*/
			denoiseKernelSpatial<<<gridDim, blockDim>>>(
					din, dout, h, w);
			break;
		case SPATIAL_KNN:
			denoiseKernelSpatialKNN<<<gridDim, blockDim>>>(
					din, dout, h, w);
			break;
		case SOBEL:
			double *dsobel = 0;
			denoiseKernelSobel<<<gridDim, blockDim>>>(
					din, frame, dsobel, dout, h, w);
			break;
		}
		cudaFree(din);
		break;
	case DIFF:
	case TEMPORAL_AVG:
	case ADAPTIVE_TEMPORAL_AVG:
	case KNN:
	case AKNN:
	case MOTION:
		ror(ctx->backlogs, BACKLOG_SIZE);
		if (ctx->backlogs[0] == 0) {
			cudaMalloc(&ctx->backlogs[0], size);
		}
		cudaMemcpy(ctx->backlogs[0], in, size, cudaMemcpyHostToDevice);

		switch (m) {
		case AKNN:
		case MOTION:
			ror(ctx->diffs, NDIFFS);
			if (ctx->diffs[0] == 0) {
				cudaMalloc(&ctx->diffs[0], size);
			}
			updateDiff<<<gridDim, blockDim>>>(
					ctx->diffs[0], ctx->backlogs[0], ctx->backlogs[1], w);
			ror(ctx->motions, NDIFFS);
			if (ctx->motions[0] == 0) {
				cudaMalloc(&ctx->motions[0],
						size / sizeof(uint32_t) * sizeof(bool));
			}
			if (ctx->m0 == 0) {
				cudaMalloc(&ctx->m0,
						size / sizeof(uint32_t) * sizeof(bool));
			}
			updateMotion<<<gridDim, blockDim>>>(
					ctx->m0, ctx->diffs[0], h, w);
			dilateMotion<<<gridDim, blockDim>>>(
					ctx->motions[0], ctx->m0, h, w);
		}

		cudaMemcpy(ctx->dbacklogs, ctx->backlogs, sizeof(ctx->backlogs),
				cudaMemcpyHostToDevice);
		cudaMemcpy(ctx->ddiffs, ctx->diffs, sizeof(ctx->diffs),
				cudaMemcpyHostToDevice);
		cudaMemcpy(ctx->dmotions, ctx->motions, sizeof(ctx->motions),
				cudaMemcpyHostToDevice);

		int i;
		for (i = 0; i < BACKLOG_SIZE && ctx->backlogs[i]; i++)
			;

		switch (m) {
		case DIFF:
			denoiseKernelDiff<<<gridDim,
				blockDim>>>(ctx->dbacklogs, i, dout, w);
			break;
		case TEMPORAL_AVG:
			denoiseKernelTemporalAvg<<<gridDim,
				blockDim>>>(ctx->dbacklogs, dout, w);
			break;
		case KNN:
			denoiseKernelKNN<<<gridDim,
				blockDim>>>(ctx->dbacklogs, i, dout, h, w);
			break;
		case AKNN:
			denoiseKernelAKNN<<<gridDim,
				blockDim>>>(ctx->dbacklogs, ctx->dmotions, i, dout, h, w);
			break;
		case MOTION:
			denoiseKernelMotion<<<gridDim,
				blockDim>>>(ctx->dmotions, dout, h, w);
			break;
		case ADAPTIVE_TEMPORAL_AVG:
			denoiseKernelAdaptiveTemporalAvg<<<gridDim,
				blockDim>>>(ctx->dbacklogs, i, dout, w);
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
	cudaMalloc(&ctx->dmotions, sizeof(ctx->motions));
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
	cudaFree(ctx->dmotions);
	cudaFree(ctx->m0);
	free(ctx);
}
