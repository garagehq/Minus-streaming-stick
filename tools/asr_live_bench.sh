#!/bin/bash
# Live ASR/delay benchmark for the name muter.
#
# For each configuration: put its environment in a systemd drop-in, restart
# Minus, start a YouTube video on the Google TV from 0:00, then measure name
# mute timing (tests/name_mute_live_measure.py) and thermals
# (tests/thermal_sample.py) for SECONDS, and score against the caption track.
# Removes the drop-in at the end so the service returns to its defaults.
#
# usage (as root): tools/asr_live_bench.sh DEVICE_IP VIDEO_ID CAPTIONS.json3 SECONDS OUTDIR \
#                    "label|ENV=1 ENV2=2" ["label2|..."]
set -u
IP=$1 VID=$2 CAP=$3 SECS=$4 OUT=$5; shift 5
HERE=$(cd "$(dirname "$0")/.." && pwd)
DROPIN=/etc/systemd/system/minus.service.d/asr-bench.conf
mkdir -p "$OUT" /etc/systemd/system/minus.service.d
for cfg in "$@"; do
    label=${cfg%%|*}; envs=${cfg#*|}
    { echo "[Service]"; for e in $envs; do echo "Environment=$e"; done; } > "$DROPIN"
    systemctl daemon-reload && systemctl restart minus
    sleep 45
    python3 - "$IP" "$VID" <<'EOF'
import sys
from adb_shell.adb_device import AdbDeviceTcp
from adb_shell.auth.sign_pythonrsa import PythonRSASigner
d = AdbDeviceTcp(sys.argv[1], 5555, default_transport_timeout_s=9)
d.connect(rsa_keys=[PythonRSASigner('', open('/root/.android/adbkey').read())], auth_timeout_s=10)
d.shell(f"am start -a android.intent.action.VIEW -d 'https://www.youtube.com/watch?v={sys.argv[2]}&t=0s' com.google.android.youtube.tv")
EOF
    sleep 8
    python3 "$HERE/tests/thermal_sample.py" "$SECS" "$OUT/$label.thermal.json" > "$OUT/$label.thermal.txt" &
    tp=$!
    if ! python3 "$HERE/tests/name_mute_live_measure.py" "$IP" "$SECS" "$OUT/$label.json" > "$OUT/$label.measure.txt" 2>&1; then
        kill $tp 2>/dev/null
        echo "=== $label FAILED: $(tail -1 "$OUT/$label.measure.txt")"
        continue
    fi
    wait $tp
    {
        echo "=== $label ($envs)"
        python3 "$HERE/tests/name_mute_live_analyze.py" "$OUT/$label.json" "$CAP" 2>&1 | grep -vE "^(extra|missed)"
        cat "$OUT/$label.thermal.txt"
    } | tee "$OUT/$label.result.txt"
done
rm -f "$DROPIN"; rmdir /etc/systemd/system/minus.service.d 2>/dev/null
systemctl daemon-reload && systemctl restart minus
