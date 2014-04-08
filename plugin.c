#include <math.h>
#include <stdlib.h>
#include <stdint.h>

#include <gst/gst.h>

#include "plugin.h"
#include "kernel.h"

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

static GstFlowReturn transform(GstVideoFilter *filter,
		GstVideoFrame *inframe, GstVideoFrame *outframe) {
	GstWebcamFilter *wfilter = (GstWebcamFilter*) filter;
	denoise(wfilter->ctx, TEMPORAL_AVG,
			(const uint32_t*) inframe->data[0],
			(uint32_t*) outframe->data[0],
			inframe->info.height, inframe->info.width);

	return GST_FLOW_OK;
}

static void gst_webcam_filter_init(GstWebcamFilter *filter) {
	filter->ctx = kernelInit();
}

static void gst_webcam_filter_finalize(GObject *obj) {
	GstWebcamFilter *filter = (GstWebcamFilter*) obj;
	kernelFinalize(filter->ctx);
}

static void gst_webcam_filter_class_init(GstWebcamFilterClass *klass) {
	GObjectClass *go_class = (GObjectClass*) klass;
	go_class->finalize = gst_webcam_filter_finalize;

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
