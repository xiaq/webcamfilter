#include <stdint.h>

extern "C" {
#include "core.h"
}

enum {
	HALF_WINDOW = 2,
};

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

__global__ void denoise_kernel(const uint32_t *in, uint32_t *out, int h,
		int w, int perthread) {
	int k = blockDim.x * blockIdx.x + threadIdx.x;
	int lower = k * perthread;
	int upper = (k + 1) * perthread;
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

void denoise(const uint32_t *in, uint32_t *out, int h, int w) {
	int threads_per_block = 256;
	int blocks_per_grid = 32;
	int perthread = h * w / (threads_per_block * blocks_per_grid);

	int size = h * w * sizeof(uint32_t);
	uint32_t *din, *dout;
	cudaMalloc(&din, size);
	cudaMalloc(&dout, size);

	cudaMemcpy(din, in, size, cudaMemcpyHostToDevice);

	denoise_kernel<<<blocks_per_grid, threads_per_block>>>(
			din, dout, h, w, perthread);

	cudaMemcpy(out, dout, size, cudaMemcpyDeviceToHost);

	cudaFree(din);
	cudaFree(dout);
}
