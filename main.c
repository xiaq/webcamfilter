#include <stdlib.h>
#include <sys/stat.h>

#include <gst/gst.h>
#include <glib.h>

static gboolean bus_call(GstBus *bus, GstMessage *msg, gpointer data) {
	GMainLoop* loop = (GMainLoop*) data;
	gchar *debug;
	GError *error;

	switch (GST_MESSAGE_TYPE(msg)) {
	case GST_MESSAGE_EOS:
		g_print("EOS\n");
		// g_main_loop_quit(loop);
		break;
	case GST_MESSAGE_ERROR:
		gst_message_parse_error(msg, &error, &debug);
		g_free(debug);

		g_printerr("Error: %s\n", error->message);
		g_error_free(error);

		g_main_loop_quit(loop);
		break;
	default:
		break;
	}
	return TRUE;
}

static void on_decode_pad_added(GstElement *decode, GstPad *pad, gpointer data) {
	static gboolean linked = FALSE;
	if (linked) {
		return;
	}
	linked = TRUE;

	g_print("Dynamic pad created, linking decodebin/queue\n");
	GstElement *queue = (GstElement*) data;
	GstPad *sink = gst_element_get_static_pad(queue, "sink");
	gst_pad_link(pad, sink);
	gst_object_unref(sink);
}

typedef struct {
	gchar *input;
} flags_t;

void init(int *pargc, char ***pargv, flags_t *f) {
	GError *err = NULL;
	GOptionEntry entries[] = {
		{ NULL }
	};
	GOptionContext *ctx = g_option_context_new(
			"[FILE] - xiaq's thesis project");
	g_option_context_add_main_entries(ctx, entries, NULL);
	g_option_context_add_group(ctx, gst_init_get_option_group());
	if (!g_option_context_parse(ctx, pargc, pargv, &err)) {
		g_printerr("Failed to initialize: %s\n", err->message);
		g_error_free(err);
		exit(1);
	}
	switch (*pargc) {
	case 1:
		break;
	case 2:
		f->input = (*pargv)[1];
		break;
	default:
		g_printerr("%s", g_option_context_get_help(ctx, TRUE, NULL));
		exit(1);
		break;
	}
}

GstElement *check_element(GstElement *e, const char *name) {
	if (!e) {
		g_printerr("Failed to create element: %s\n", name);
		exit(1);
	}
	return e;
}

GstElement *make_element(char *factory, char *name) {
	return check_element(gst_element_factory_make(factory, name), name);
}

int main(int argc, char **argv) {
	// Init
	flags_t flags = { "/dev/video0" };
	init(&argc, &argv, &flags);

	struct stat input_stat;
	if (stat(flags.input, &input_stat)) {
		g_printerr("Bad file: %s\n", flags.input);
		return 1;
	}

	// Set up loop and pipeline
	GMainLoop *loop = g_main_loop_new(NULL, FALSE);
	GstElement *pipeline = gst_pipeline_new("xth-pipeline");
	check_element(pipeline, "xth-pipeline");

	GstBus *bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline));
	guint bus_watch_id = gst_bus_add_watch(bus, bus_call, loop);
	gst_object_unref(bus);

	// Set up source and sink
	GstElement *source, *sink;

	if (S_ISCHR(input_stat.st_mode)) {
		// Assume V4L2 device
		source = gst_bin_new("v4l2-source-bin");
		check_element(source, "v4l2-source-bin");

		GstElement *vsrc, *flip;
		vsrc = make_element("v4l2src", "v4l2src");
		g_object_set(G_OBJECT(vsrc), "device", flags.input, NULL);
		flip = make_element("videoflip", "video-flip");
		g_object_set(G_OBJECT(flip), "method", 4, NULL);

		gst_bin_add_many(GST_BIN(source), vsrc, flip, NULL);
		gst_element_link(vsrc, flip);

		gst_element_add_pad(source, gst_ghost_pad_new("src",
					gst_element_get_static_pad(flip, "src")));
	} else {
		// Assume video file source
		source = gst_bin_new("file-source-bin");
		check_element(source, "file-source-bin");

		GstElement *fsrc, *decode, *queue;
		fsrc = make_element("filesrc", "file-source");
		g_object_set(G_OBJECT(fsrc), "location", flags.input, NULL);
		decode = make_element("decodebin", "decode-bin");
		queue = make_element("queue", "queue");

		gst_bin_add_many(GST_BIN(source), fsrc, decode, queue, NULL);
		gst_element_link(fsrc, decode);
		g_signal_connect(decode, "pad-added",
				G_CALLBACK(on_decode_pad_added), queue);
		gst_element_add_pad(source, gst_ghost_pad_new("src",
					gst_element_get_static_pad(queue, "src")));
	}

	sink = make_element("xvimagesink", "video-output");

	gst_bin_add_many(GST_BIN(pipeline), source, sink, NULL);
	gst_element_link(source, sink);

	// Start rolling
	g_print("Playing...\n");
	gst_element_set_state(pipeline, GST_STATE_PLAYING);

	g_print("Running...\n");
	g_main_loop_run(loop);

	g_print("Stopping...\n");
	gst_element_set_state(pipeline, GST_STATE_NULL);

	g_print("Cleaning up...\n");
	gst_object_unref(GST_OBJECT(pipeline));
	g_source_remove(bus_watch_id);
	g_main_loop_unref(loop);

	return 0;
}
