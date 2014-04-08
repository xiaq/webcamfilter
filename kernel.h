extern void denoise(void*, int, const uint32_t *in, uint32_t *out, int h, int w);
extern void *kernelInit(void);
extern void kernelFinalize(void*);

enum {
	SPATIAL,
	TEMPORAL_AVG,
	ADAPTIVE_TEMPORAL_AVG,
};
