#!/usr/bin/env python3
"""Headless check that concurrent pipeline rebuilds leave exactly one display.

Usage:  python3 tools/display_pipeline_race.py [path/to/ad_blocker.py] [label]

Needs the live ustreamer on :9090 but NO display: kmssink is swapped for a
fakesink, and everything else -- _init_pipeline, the pipeline string, the
thermal frame gate and its token bucket, souphttpsrc on the real stream -- is
the production code. It reproduces the race that ran two display pipelines at
once when the TV came on (health-monitor reconnect restart and display retry
loop building in the same second), then turns the 30fps thermal cap on and
counts what reaches the "screen".

Expected on a fixed tree: 1 pipeline PLAYING, 1 ustreamer client, ~30fps.
Against the pre-fix code it showed either 2 pipelines splitting the cap
(~15fps each) or a single wedged pipeline at ~0fps, depending on timing.
"""

import importlib.util, json, re, sys, threading, time, urllib.request
sys.path.insert(0, '/home/radxa/Minus/src'); sys.path.insert(0, '/home/radxa/Minus')
import gi; gi.require_version('Gst', '1.0')
from gi.repository import Gst
Gst.init(None)

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def clients():
    with urllib.request.urlopen('http://localhost:9090/state', timeout=5) as r:
        return json.load(r)['result']['stream']['clients']

def run(modpath, label):
    mod = load(modpath, 'ab_' + label)
    created = []
    real = Gst.parse_launch
    def headless(desc):
        desc = re.sub(r'kmssink[^!]*$', 'fakesink sync=false', desc)
        p = real(desc); created.append(p); return p
    mod.Gst.parse_launch = headless
    try:
        B = mod.DRMAdBlocker
        b = object.__new__(B)
        b.pipeline = None; b.bus = None
        b.ustreamer_port = 9090; b.plane_id = 0; b.connector_id = 0
        b._saved_color_settings = {'saturation': 1.15, 'brightness': 0.0, 'contrast': 1.0, 'hue': 0.0}
        b._fps_lock = threading.Lock(); b._frame_count = 0; b._fps_start_time = time.time()
        b._current_fps = 0.0; b._last_buffer_time = time.time()
        b._consecutive_failures = 0; b._last_restart_time = 0; b._success_reset_time = 0
        b._thermal_fps_cap = False; b._framegate_rate = 30.0
        b._framegate_credit = 0.0; b._framegate_last_ts = None
        if hasattr(B, '_teardown_pipeline'):
            b._build_lock = threading.RLock()
        base = clients()

        # The production race: two recovery paths build at the same instant.
        bar = threading.Barrier(2)
        def path():
            bar.wait(); b._init_pipeline(); b.pipeline.set_state(Gst.State.PLAYING)
        ts = [threading.Thread(target=path) for _ in range(2)]
        [t.start() for t in ts]; [t.join() for t in ts]
        time.sleep(5)
        live = [p for p in created if p.get_state(0)[1] == Gst.State.PLAYING]
        n_clients = clients() - base

        # Thermal cap on; count what reaches the "display" on EVERY live pipeline.
        counts = {}
        for i, p in enumerate(live):
            counts[i] = 0
            def cb(pad, info, i=i): counts[i] += 1; return Gst.PadProbeReturn.OK
            p.get_by_name('fpsprobe').get_static_pad('src').add_probe(Gst.PadProbeType.BUFFER, cb)
        b._thermal_fps_cap = True
        time.sleep(3); [counts.__setitem__(k, 0) for k in counts]; time.sleep(10)
        shown_idx = next((i for i, p in enumerate(live) if p is b.pipeline), None)
        print(f"[{label}] pipelines built={len(created)} still PLAYING={len(live)} "
              f"extra ustreamer clients={n_clients}")
        for i in counts:
            tag = 'ON SCREEN' if i == shown_idx else 'ORPHANED '
            print(f"[{label}]   {tag} pipeline: {counts[i]/10:5.1f} fps with the 30fps cap")
    finally:
        for p in created: p.set_state(Gst.State.NULL)
        mod.Gst.parse_launch = real
        time.sleep(2)

if __name__ == '__main__':
    run(sys.argv[1] if len(sys.argv) > 1 else '/home/radxa/Minus/src/ad_blocker.py',
        sys.argv[2] if len(sys.argv) > 2 else 'current')
