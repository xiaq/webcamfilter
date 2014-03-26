flags = [
    '-I/usr/include/gstreamer-1.0',
    '-I/usr/include/glib-2.0',
    '-I/usr/lib/glib-2.0/include',
]

def FlagsForFile(fname, **kwargs):
    return dict(flags=flags, do_cache=True)
