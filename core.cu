#include <stdint.h>

extern "C" {
#include "core.h"
}

enum {
	HALF_WINDOW = 1,
};

inline static double norm(uint32_t v) {
	double r = v & 0xff, g = (v >> 8) & 0xff, b = (v >> 16) & 0xff;
	return sqrt(r * r + g * g + b * b);
}

#define in(i, j) in[i*w + j]
#define out(i, j) out[i*w + j]
#define FOR_WINDOW(i2, j2, i, j, h, w) \
	for (i2 = max(0, i - HALF_WINDOW); min(h, i2 < i + HALF_WINDOW); i2++) \
		for (j2 = max(0, j - HALF_WINDOW); min(w, j2 < j + HALF_WINDOW); j2++)

void denoise(const uint32_t *in, uint32_t *out, int h, int w) {
	int i, i2, i3, j, j2, j3;
	for (i = 0; i < h; i++) {
		for (j = 0; j < w; j++) {
			double dmin = INFINITY;
			FOR_WINDOW(i2, j2, i, j, h, w) {
				double d = 0;
				FOR_WINDOW(i3, j3, i, j, h, w) {
					d += norm(in(i2, j2) - in(i3, j3));
				}
				if (dmin > d) {
					dmin = d;
					out(i, j) = in(i2, j2);
				}
			}
		}
	}
}
#undef in
#undef out
