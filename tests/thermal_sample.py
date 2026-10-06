#!/usr/bin/env python3
"""Sample SoC thermals, CPU throttling and stream health while Minus runs.

Every INTERVAL seconds records: hottest thermal zone, per-cluster CPU clock
(policy0 = A55, policy4/policy6 = the two A76 pairs), whether any cpufreq
cooling device is throttling, whole-system CPU utilisation, Minus's
display/stream fps and the ASR per-window latency. Prints a summary at the
end and writes the raw samples to OUT.json.

usage: python3 tests/thermal_sample.py SECONDS OUT.json [interval_s]
"""
import glob
import json
import statistics
import sys
import time
import urllib.request


def read(path, cast=int):
    try:
        with open(path) as f:
            return cast(f.read().strip())
    except Exception:
        return None


def cpu_times():
    with open('/proc/stat') as f:
        v = [int(x) for x in f.readline().split()[1:]]
    return sum(v), v[3] + v[4]          # total, idle+iowait


def status():
    try:
        with urllib.request.urlopen('http://localhost/api/status', timeout=2) as r:
            return json.load(r)
    except Exception:
        return {}


def main():
    secs, out = float(sys.argv[1]), sys.argv[2]
    interval = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0
    zones = glob.glob('/sys/class/thermal/thermal_zone*/temp')
    cooling = [p for p in glob.glob('/sys/class/thermal/cooling_device*')
               if (read(p + '/type', str) or '').startswith('cpufreq')]
    samples = []
    t_end = time.time() + secs
    prev = cpu_times()
    while time.time() < t_end:
        time.sleep(interval)
        cur = cpu_times()
        busy = 1 - (cur[1] - prev[1]) / max(1, cur[0] - prev[0])
        prev = cur
        st = status()
        asr = st.get('asr') or {}
        samples.append({
            't': round(time.time(), 1),
            'temp_c': max((read(z) or 0) for z in zones) / 1000,
            'mhz': {p: (read(f'/sys/devices/system/cpu/cpufreq/{p}/scaling_cur_freq') or 0) // 1000
                    for p in ('policy0', 'policy4', 'policy6')},
            'throttled': any((read(c + '/cur_state') or 0) > 0 for c in cooling),
            'cpu_busy': round(busy, 3),
            'fps_display': st.get('fps'),
            'fps_stream': st.get('fps_stream'),
            'asr_p50': asr.get('p50_latency_s'),
        })
    json.dump(samples, open(out, 'w'))
    t = [s['temp_c'] for s in samples]
    big = [s['mhz']['policy4'] for s in samples]
    fs = [s['fps_stream'] for s in samples if s['fps_stream'] is not None]
    print(f"thermal: temp mean {statistics.mean(t):.1f}C max {max(t):.1f}C | "
          f"throttled {100 * sum(s['throttled'] for s in samples) / len(samples):.0f}% of samples | "
          f"A76(policy4) mean {statistics.mean(big):.0f}MHz | "
          f"CPU busy mean {100 * statistics.mean(s['cpu_busy'] for s in samples):.0f}% | "
          f"stream fps min {min(fs) if fs else 'n/a'} mean {statistics.mean(fs) if fs else 0:.1f}")


if __name__ == '__main__':
    main()
