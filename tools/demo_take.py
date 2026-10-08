#!/usr/bin/env python3
"""One demo take: play a video segment on the Google TV and capture it.

Launches VIDEO_ID a few seconds before START_S, waits until the media
session reports the video itself playing near START_S (pre-roll ads report
their own small positions), then runs tools/demo_capture.py long enough for
the segment to come out of Minus's A/V delay, logging the video position
against the box's monotonic clock (syncs.jsonl) so the composer can find
the moment START_S reached the output.

Needs root (the ADB key is root's). Output is chowned to uid 1000.

usage: sudo python3 tools/demo_take.py OUT_DIR VIDEO_ID START_S [SECONDS=30] [DELAY_S=5.4]
"""
import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import clip_farm  # noqa: E402
import demo_capture  # noqa: E402
import label_clips  # noqa: E402

LEAD_S = 8


def main():
    out, vid, start = sys.argv[1], sys.argv[2], float(sys.argv[3])
    seconds = float(sys.argv[4]) if len(sys.argv) > 4 else 30.0
    delay = float(sys.argv[5]) if len(sys.argv) > 5 else 5.4
    os.makedirs(out, exist_ok=True)
    dev = clip_farm.connect(os.environ.get('CLIP_FARM_TV_IP', '192.168.1.203'))
    t_launch = time.monotonic()
    dev.shell(f"am start -a android.intent.action.VIEW -d "
              f"'https://www.youtube.com/watch?v={vid}&t={int(start - LEAD_S)}s' "
              f"com.google.android.youtube.tv")
    pos = None
    while time.monotonic() - t_launch < 120:
        s = clip_farm.best_sync(dev)
        if s and s['updated'] >= s['device_uptime'] - (s['board_mid'] - t_launch) + 0.2:
            p = label_clips.vpos(s, s['board_mid'])
            if start - LEAD_S - 6 <= p <= start + 1:
                pos = p
                break
        time.sleep(1)
    if pos is None:
        sys.exit(f"{vid}: never reached {start - LEAD_S:.0f}s (ad, Restricted Mode?)")
    length = (start - pos) + seconds + delay + 4
    syncs = open(os.path.join(out, 'syncs.jsonl'), 'w')
    stop = time.monotonic() + length

    def log_syncs():
        while time.monotonic() < stop:
            s = clip_farm.best_sync(dev, n=2)
            if s:
                syncs.write(json.dumps({**s, 'pos': label_clips.vpos(s, s['board_mid'])}) + '\n')
                syncs.flush()
            time.sleep(1)

    th = threading.Thread(target=log_syncs)
    th.start()
    sys.argv = ['demo_capture', out, str(length)]
    demo_capture.main()
    th.join()
    syncs.close()
    with open(os.path.join(out, 'take.json'), 'w') as f:
        json.dump({'video': vid, 'start_s': start, 'seconds': seconds, 'delay_s': delay}, f)
    for root, dirs, files in os.walk(out):
        for n in dirs + files:
            os.chown(os.path.join(root, n), 1000, 1000)
    os.chown(out, 1000, 1000)


if __name__ == '__main__':
    main()
