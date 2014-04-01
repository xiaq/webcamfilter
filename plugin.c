#include <math.h>
#include <stdlib.h>

#include <gst/gst.h>

#include "plugin.h"

GST_DEBUG_CATEGORY_STATIC(gst_webcam_filter_debug);
#define GST_CAT_DEFAULT gst_webcam_filter_debug

// Filter props
enum {
	PROP_0,
	PROP_SILENT
};

// The capabilities of the inputs and outputs
#define FORMATS "{ RGBx }"

#define CAPS GST_STATIC_CAPS( \
		GST_VIDEO_CAPS_MAKE(FORMATS) ";" \
		GST_VIDEO_CAPS_MAKE_WITH_FEATURES("ANY", FORMATS) \
		)

static GstStaticPadTemplate sink_factory = GST_STATIC_PAD_TEMPLATE(
		"sink", GST_PAD_SINK, GST_PAD_ALWAYS, CAPS);

static GstStaticPadTemplate src_factory = GST_STATIC_PAD_TEMPLATE(
		"src", GST_PAD_SRC, GST_PAD_ALWAYS, CAPS);

#define gst_webcam_filter_parent_class parent_class
G_DEFINE_TYPE(GstWebcamFilter, gst_webcam_filter, GST_TYPE_VIDEO_FILTER);

// GObject vmethod implementations

enum {
	HALF_WINDOW = 1,
};

inline static int min(int a, int b) {
	return a < b ? a : b;
}

inline static int max(int a, int b) {
	return a > b ? a : b;
}

#define FOR_WINDOW(i2, j2, i, j, h, w) \
	for (i2 = max(0, i - HALF_WINDOW); min(h, i2 < i + HALF_WINDOW); i2++) \
		for (j2 = max(0, j - HALF_WINDOW); min(w, j2 < j + HALF_WINDOW); j2++)

inline static double norm(guint32 v) {
	double r = v & 0xff, g = (v >> 8) & 0xff, b = (v >> 16) & 0xff;
	return sqrt(r * r + g * g + b * b);
}

#define in(i, j) indata[i*w + j]
#define out(i, j) outdata[i*w + j]
static GstFlowReturn transform(GstVideoFilter *filter,
		GstVideoFrame *inframe, GstVideoFrame *outframe) {
	guint32 *indata = (guint32*) inframe->data[0];
	guint32 *outdata = (guint32*) outframe->data[0];

	int i, i2, i3, j, j2, j3,
		w = inframe->info.width, h = inframe->info.height;
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

	return GST_FLOW_OK;
}

static void gst_webcam_filter_class_init(GstWebcamFilterClass *klass) {
	GstElementClass *ge_class = (GstElementClass*) klass;
	gst_element_class_set_details_simple(ge_class,
			"WebcamFilter",
			"Generic",
			"Generic Template Element",
			" <<xiaqqaix@gmail.com>>");
	gst_element_class_add_pad_template(ge_class,
			gst_static_pad_template_get(&src_factory));
	gst_element_class_add_pad_template(ge_class,
			gst_static_pad_template_get(&sink_factory));

	GstBaseTransformClass *bt_class = (GstBaseTransformClass*) klass;
	/*
	bt_class->transform_caps =
		GST_DEBUG_FUNCPTR(gst_webcam_filter_transform_caps);
	bt_class->src_event =
		GST_DEBUG_FUNCPTR();
	*/

	GstVideoFilterClass *vf_class = (GstVideoFilterClass*) klass;
	// vf_class->set_info = GST_DEBUG_FUNCPTR(gst_webcam_filter_set_info);
	vf_class->transform_frame = GST_DEBUG_FUNCPTR(transform);
}

static void gst_webcam_filter_init(GstWebcamFilter *filter) {
}

static gboolean plugin_init(GstPlugin *plugin) {
	GST_DEBUG_CATEGORY_INIT(gst_webcam_filter_debug, "webcamfilter",
			0, "Template webcamfilter");
	return gst_element_register(plugin, "webcamfilter", GST_RANK_NONE,
			GST_TYPE_WEBCAMFILTER);
}

#define PACKAGE "webcamfilter"

GST_PLUGIN_DEFINE(
		GST_VERSION_MAJOR,
		GST_VERSION_MINOR,
		webcamfilter,
		"A filter for webcam video",
		plugin_init,
		"0.1dev",
		"LGPL",
		"GStreamer",
		"http://gstreamer.net/"
		)
