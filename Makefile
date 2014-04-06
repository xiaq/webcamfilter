CC := clang
NVCC := nvcc
CFLAGS := $(shell pkg-config --cflags gstreamer-1.0) -g
LDFLAGS := $(shell pkg-config --libs gstreamer-1.0)
SO_LDFLAGS := $(shell pkg-config --libs gstreamer-video-1.0) -lm -L/opt/cuda/lib64 -lcuda -lcudart

all: main plugin.so

plugin.so: plugin.c kernel.o
	$(CC) $(CFLAGS) $(SO_LDFLAGS) -shared -fPIC -DPIC -o $@ $+

kernel.o: kernel.cu
	$(NVCC) -Xcompiler -fPIC -DPIC -c -arch=sm_20 -o $@ $<
