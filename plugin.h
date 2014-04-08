#ifndef __GST_WEBCAMFILTER_H__
#define __GST_WEBCAMFILTER_H__

#include <gst/video/gstvideofilter.h>

G_BEGIN_DECLS

#define GST_TYPE_WEBCAMFILTER \
	(gst_webcam_filter_get_type())
#define GST_WEBCAMFILTER(obj) \
	(G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_WEBCAMFILTER,GstWebcamFilter))
#define GST_WEBCAMFILTER_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_WEBCAMFILTER,GstWebcamFilterClass))
#define GST_IS_WEBCAMFILTER(obj) \
	(G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_WEBCAMFILTER))
#define GST_IS_WEBCAMFILTER_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_WEBCAMFILTER))

typedef struct {
	GstVideoFilter parent;
	void *ctx;
} GstWebcamFilter;

typedef struct {
	GstVideoFilterClass parent_class;
} GstWebcamFilterClass;

GType gst_webcam_filter_get_type(void);

G_END_DECLS

#endif /* __GST_WEBCAMFILTER_H__ */
