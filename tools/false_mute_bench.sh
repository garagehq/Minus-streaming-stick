#!/bin/bash
# False-mute benchmark: play non-LeBron videos through Minus and log every
# name mute with the text that triggered it.
#
# usage (as root): tools/false_mute_bench.sh DEVICE_IP OUTDIR SECONDS_PER_VIDEO \
#                    "label|ENV=1 ..." -- VIDEO_ID [VIDEO_ID ...]
set -u
IP=$1 OUT=$2 SECS=$3; shift 3
CONFIGS=(); while [ "$1" != "--" ]; do CONFIGS+=("$1"); shift; done; shift
VIDEOS=("$@")
DROPIN=/etc/systemd/system/minus.service.d/fp-bench.conf
mkdir -p "$OUT" /etc/systemd/system/minus.service.d
for cfg in "${CONFIGS[@]}"; do
    label=${cfg%%|*}; envs=${cfg#*|}
    { echo "[Service]"; for e in $envs; do echo "Environment=$e"; done; } > "$DROPIN"
    systemctl daemon-reload && systemctl restart minus
    sleep 45
    curl -s -X POST localhost/api/asr/enable >/dev/null
    for vid in "${VIDEOS[@]}"; do
        python3 - "$IP" "$vid" <<'EOF'
import sys
from adb_shell.adb_device import AdbDeviceTcp
from adb_shell.auth.sign_pythonrsa import PythonRSASigner
d = AdbDeviceTcp(sys.argv[1], 5555, default_transport_timeout_s=9)
d.connect(rsa_keys=[PythonRSASigner('', open('/root/.android/adbkey').read())], auth_timeout_s=10)
d.shell(f"am start -a android.intent.action.VIEW -d 'https://www.youtube.com/watch?v={sys.argv[2]}&t=0s' com.google.android.youtube.tv")
EOF
        sleep 8
        start=$(python3 -c "import time;print(time.monotonic())")
        c0=$(curl -s localhost/api/status | python3 -c "import sys,json;print(json.load(sys.stdin)['asr']['inference_count'])")
        sleep "$SECS"
        curl -s "localhost/api/name-mute/log?since=$start" > "$OUT/$label.$vid.json"
        python3 - "$OUT/$label.$vid.json" "$label" "$vid" "$SECS" "$c0" <<'EOF'
import json, sys, urllib.request
d = json.load(open(sys.argv[1]))['detections']
c1 = json.load(urllib.request.urlopen('http://localhost/api/status'))['asr']['inference_count']
u = [x for x in d if not x.get('duplicate')]
print(f"== {sys.argv[2]} {sys.argv[3]}: {len(u)} mutes in {int(sys.argv[4]) / 60:.0f} min "
      f"(asr {sum(x['source'] == 'asr' for x in u)}, caption {sum(x['source'] == 'caption' for x in u)}), "
      f"asr inferences {c1 - int(sys.argv[5])}")
for x in u:
    print(f"   {x['source']:7} {x['label'][:90]!r}")
EOF
    done
done
rm -f "$DROPIN"; rmdir /etc/systemd/system/minus.service.d 2>/dev/null
systemctl daemon-reload && systemctl restart minus
