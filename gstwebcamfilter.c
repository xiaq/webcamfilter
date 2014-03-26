#include <gst/gst.h>

#include "gstwebcamfilter.h"

GST_DEBUG_CATEGORY_STATIC(gst_webcam_filter_debug);
#define GST_CAT_DEFAULT gst_webcam_filter_debug

// Filter props
enum {
	PROP_0,
	PROP_SILENT
};

// The capabilities of the inputs and outputs
static GstStaticPadTemplate sink_factory = GST_STATIC_PAD_TEMPLATE(
		"sink", GST_PAD_SINK, GST_PAD_ALWAYS, GST_STATIC_CAPS("ANY"));

static GstStaticPadTemplate src_factory = GST_STATIC_PAD_TEMPLATE(
		"src", GST_PAD_SRC, GST_PAD_ALWAYS, GST_STATIC_CAPS("ANY"));

#define gst_webcam_filter_parent_class parent_class
G_DEFINE_TYPE(GstWebcamFilter, gst_webcam_filter, GST_TYPE_VIDEO_FILTER);

// GObject vmethod implementations

/*
static void gst_webcam_filter_set_property(
		GObject *obj, guint prop_id, const GValue *value, GParamSpec *pspec) {
	GstWebcamFilter *filter = GST_WEBCAMFILTER(obj);
	switch (prop_id) {
	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(obj, prop_id, pspec);
		break;
	}
}

static void gst_webcam_filter_get_property(
		GObject *obj, guint prop_id, GValue *value, GParamSpec *pspec) {
	GstWebcamFilter *filter = GST_WEBCAMFILTER(obj);
	switch (prop_id) {
	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(obj, prop_id, pspec);
		break;
	}
}

static gboolean gst_webcam_filter_set_info(GstVideoFilter *filter,
		GstCaps *incaps, GstVideoInfo *ininfo,
		GstCaps *outcaps, GstVideoInfo *outinfo) {
}
*/

GstFlowReturn gst_webcam_filter_transform_frame(GstVideoFilter *filter,
		GstVideoFrame *inframe, GstVideoFrame *outframe) {
	gst_video_frame_copy(outframe, inframe);
	return GST_FLOW_OK;
}

static void gst_webcam_filter_class_init(GstWebcamFilterClass *klass) {
	GObjectClass *gob_class = (GObjectClass*) klass;
	/*
	gob_class->set_property = gst_webcam_filter_set_property;
	gob_class->get_property = gst_webcam_filter_get_property;
	*/

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
	vf_class->transform_frame =
		GST_DEBUG_FUNCPTR(gst_webcam_filter_transform_frame);
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
