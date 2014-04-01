CC := clang
CFLAGS := $(shell pkg-config --cflags gstreamer-1.0) -g
LDFLAGS := $(shell pkg-config --libs gstreamer-1.0)

SO_LDFLAGS := $(shell pkg-config --libs gstreamer-video-1.0) -lm

all: main libgstwebcamfilter.so

libgstwebcamfilter.so: plugin.c
	$(CC) $(CFLAGS) $(SO_LDFLAGS) -shared -fPIC -DPIC -o $@ $<
