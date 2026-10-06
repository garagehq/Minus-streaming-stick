#!/usr/bin/env python3
"""Measure live name-mute timing against a YouTube caption track.

Runs on the Minus box while an Android TV (Google TV / Fire TV) plays a
YouTube video through it. Collects:

  * clock sync: repeated ADB round trips reading the device's
    elapsedRealtime (/proc/uptime) and YouTube's MediaSession position, so
    video position can be computed for any instant on the box's
    time.monotonic() clock;
  * Minus's detection log (GET /api/name-mute/log).

Saves everything to a JSON file; analyse with name_mute_live_analyze.py.

usage (as root, for the ADB key):
  python3 tests/name_mute_live_measure.py <device-ip> <seconds> <out.json>
"""
import json
import os
import re
import sys
import time
import urllib.request

from adb_shell.adb_device import AdbDeviceTcp
from adb_shell.auth.sign_pythonrsa import PythonRSASigner

KEY = os.environ.get('ADBKEY', '/root/.android/adbkey')
POS_RE = re.compile(r'state=PLAYING\(3\), position=(\d+), buffered position=\d+, '
                    r'speed=([0-9.]+), updated=(\d+)')


def connect(ip):
    d = AdbDeviceTcp(ip, 5555, default_transport_timeout_s=10)
    signer = PythonRSASigner(open(KEY + '.pub').read(), open(KEY).read())
    d.connect(rsa_keys=[signer], auth_timeout_s=5)
    return d


def sync_sample(dev):
    """One round trip: (rtt, board_mid, device_uptime, position_s, updated_s, speed)."""
    t0 = time.monotonic()
    out = dev.shell("cat /proc/uptime; dumpsys media_session | grep -m1 'state=PLAYING'",
                    read_timeout_s=10)
    t1 = time.monotonic()
    up = float(out.split()[0])
    m = POS_RE.search(out)
    if not m:
        return None
    return {'rtt': t1 - t0, 'board_mid': (t0 + t1) / 2, 'device_uptime': up,
            'position': int(m.group(1)) / 1000.0, 'speed': float(m.group(2)),
            'updated': int(m.group(3)) / 1000.0}


def sync(dev, n=5):
    samples = [s for s in (sync_sample(dev) for _ in range(n)) if s]
    samples.sort(key=lambda s: s['rtt'])
    return samples[:2]   # the fastest round trips bound the clock offset best


def main():
    ip, secs, out_path = sys.argv[1], float(sys.argv[2]), sys.argv[3]
    dev = connect(ip)
    start = time.monotonic()
    # Sync often: YouTube mid-roll ads stop the video clock, and the analyzer
    # only scores stretches where consecutive syncs agree.
    every = float(os.environ.get('SYNC_EVERY_S', '5'))
    syncs = [sync(dev, n=25)]
    print(f"initial sync: rtt {syncs[0][0]['rtt']*1000:.0f}ms, "
          f"video at {syncs[0][0]['position']:.1f}s", flush=True)
    while time.monotonic() - start < secs:
        time.sleep(every)
        try:
            syncs.append(sync(dev))
        except Exception as e:
            print(f"sync failed: {e}", flush=True)
            try:
                dev = connect(ip)
            except Exception:
                pass
        if len(syncs) % max(1, int(60 / every)) == 0 and syncs[-1]:
            print(f"{time.monotonic() - start:6.0f}s synced (rtt "
                  f"{syncs[-1][0]['rtt']*1000:.0f}ms)", flush=True)
    log = json.load(urllib.request.urlopen(
        f'http://localhost/api/name-mute/log?since={start}', timeout=10))
    json.dump({'start': start, 'end': time.monotonic(), 'syncs': syncs,
               'detections': log['detections']}, open(out_path, 'w'), indent=1)
    print(f"saved {len(log['detections'])} detections, {len(syncs)} syncs -> {out_path}")


if __name__ == '__main__':
    main()
