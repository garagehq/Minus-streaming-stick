#!/bin/bash
# End-to-end regression run ON a Minus box. NEVER runs the Wi-Fi / AP tests:
# on 2026-10-15 a full `pytest tests` on a box drove the live Wi-Fi API and took
# it off the network (tests/live_guard.py). Those two files are excluded here
# by name, whatever else is selected.
#
# Run on the box as the normal user (radxa), from any checkout of this repo:
#
#   tests/run_box_e2e.sh                  # 1. full test suite on the box's hardware
#   tests/run_box_e2e.sh --live-ui        # + 2. Playwright tests against the running service
#   tests/run_box_e2e.sh --source-cycle   # + 3. sleep/wake the HDMI source over adb
#   tests/run_box_e2e.sh --all            # all three
#
# 1. pytest on the box: real NPU runtimes, models and GStreamer, so tests that
#    skip on a dev host run here. Needs pytest (pip install --user pytest).
# 2. --live-ui: tests/test_status_leds_ui.py, test_review_ui.py, test_ir_ui.py
#    with MINUS_LIVE_UI_TESTS=1. They act for real: the IR test presses Power
#    on the IR blaster, the review test classifies screenshots and undoes it.
#    IR and LED settings are put back afterwards; HDMI signal is reported
#    before/after so an IR press that switched something off is visible.
# 3. --source-cycle: puts the HDMI source (Fire TV / Android TV) to sleep with
#    adb, checks Minus shows NO SIGNAL and stays there, wakes it with adb and
#    checks video, audio and detection come back with no Traceback. Autonomous
#    mode is disabled for the cycle (it would wake the source itself) and put
#    back. Needs the `adb` package and root's ADB key (the one Minus uses, so
#    the TV shows no new authorisation prompt); FIRE_TV_IP overrides the IP
#    read from the journal. sudo is used for adb/journalctl (SUDO_ASKPASS ok).
#
# Exit status: 0 only if every selected part passed.
set -u
cd "$(dirname "$0")/.."

LIVE_UI=0; CYCLE=0
for a in "$@"; do
  case "$a" in
    --live-ui) LIVE_UI=1 ;;
    --source-cycle) CYCLE=1 ;;
    --all) LIVE_UI=1; CYCLE=1 ;;
    -h|--help) sed -n '2,32p' "$0"; exit 0 ;;
    *) echo "unknown option $a" >&2; exit 2 ;;
  esac
done

API=http://localhost
SUDO=(sudo); [ -n "${SUDO_ASKPASS:-}" ] && SUDO=(sudo -A)
FAIL=()
step() { echo; echo "=== $*"; }
api_get() { curl -s --max-time 5 "$API$1"; }
json() { python3 -c "import sys,json; d=json.load(sys.stdin); print(d$1)" 2>/dev/null; }
hdmi() { api_get /api/status | json ".get('hdmi_signal')"; }

# ---------------------------------------------------------------- 1. pytest
step "1. test suite on this box (Wi-Fi/AP tests excluded)"
python3 -m pytest --version >/dev/null 2>&1 || { echo "pytest missing: python3 -m pip install --user pytest"; exit 2; }
python3 -m pytest -q -p no:cacheprovider -rfE \
    --ignore=tests/test_wifi_portal.py --ignore=tests/test_wifi_ap_recovery.py \
    tests 2>&1 | grep -E "^(FAILED|ERROR)|passed|failed|error"
[ "${PIPESTATUS[0]}" = 0 ] || FAIL+=("unit suite")

# --------------------------------------------------------------- 2. live UI
if [ $LIVE_UI = 1 ]; then
  step "2. live UI tests against the running service"
  ir_before=$(api_get /api/ir/status | json ".get('enabled')")
  leds_before=$(api_get /api/leds/status | json ".get('enabled')")
  hdmi_before=$(hdmi)
  for f in tests/test_status_leds_ui.py tests/test_review_ui.py tests/test_ir_ui.py; do
    echo "--- $f"
    MINUS_LIVE_UI_TESTS=1 python3 -m pytest -q -p no:cacheprovider -rfE "$f" 2>&1 \
        | grep -E "^(FAILED|ERROR)|passed|failed|error"
    [ "${PIPESTATUS[0]}" = 0 ] || FAIL+=("$f")
  done
  # Put IR and LEDs back the way they were.
  [ "$(api_get /api/ir/status | json ".get('enabled')")" = "$ir_before" ] || \
      curl -s -XPOST "$API/api/ir/$([ "$ir_before" = True ] && echo enable || echo disable)" >/dev/null
  [ "$(api_get /api/leds/status | json ".get('enabled')")" = "$leds_before" ] || \
      curl -s -XPOST "$API/api/leds/$([ "$leds_before" = True ] && echo enable || echo disable)" >/dev/null
  echo "HDMI signal before/after: $hdmi_before / $(hdmi)"
  [ "$hdmi_before" = "$(hdmi)" ] || echo "WARNING: HDMI signal changed during the live UI tests (IR Power press?)"
fi

# ---------------------------------------------------------- 3. source cycle
if [ $CYCLE = 1 ]; then
  step "3. source sleep/wake over adb"
  ADB=("${SUDO[@]}" env HOME=/root adb)
  ip=${FIRE_TV_IP:-$("${SUDO[@]}" journalctl -u minus --since -7d 2>/dev/null \
        | grep -oE '\[FireTV\] Connected to [0-9.]+' | tail -1 | awk '{print $NF}')}
  if ! command -v adb >/dev/null || [ -z "$ip" ]; then
    echo "needs the adb package and a source IP (FIRE_TV_IP)"; FAIL+=("source cycle: setup")
  elif [ "$(hdmi)" != True ]; then
    echo "no HDMI signal to start from"; FAIL+=("source cycle: no signal at start")
  else
    dev="$ip:5555"
    "${ADB[@]}" connect "$dev" >/dev/null
    auto=$(api_get /api/autonomous)
    auto_enabled=$(echo "$auto" | json ".get('enabled')")
    auto_active=$(echo "$auto" | json ".get('active')")
    [ "$auto_enabled" = True ] && curl -s -XPOST "$API/api/autonomous/disable" >/dev/null
    t0=$(date -u '+%Y-%m-%d %H:%M:%S')
    "${ADB[@]}" -s "$dev" shell input keyevent KEYCODE_SLEEP
    lost=""; for i in $(seq 1 12); do sleep 5; [ "$(hdmi)" = False ] && { lost=$((i*5)); break; }; done
    if [ -z "$lost" ]; then
      echo "signal never dropped after KEYCODE_SLEEP"; FAIL+=("source cycle: sleep")
    else
      echo "NO SIGNAL after ${lost}s"; sleep 45
      [ "$(hdmi)" = False ] || { echo "signal came back on its own while asleep"; FAIL+=("source cycle: stayed asleep"); }
    fi
    t1=$(date -u '+%Y-%m-%d %H:%M:%S')
    "${ADB[@]}" -s "$dev" shell input keyevent KEYCODE_WAKEUP
    back=""; for i in $(seq 1 24); do sleep 5; [ "$(hdmi)" = True ] && { back=$((i*5)); break; }; done
    if [ -z "$back" ]; then
      echo "signal not back 120 s after KEYCODE_WAKEUP"; FAIL+=("source cycle: wake")
    else
      echo "signal back ${back}s after wake"; sleep 30
      log=$("${SUDO[@]}" journalctl -u minus --since "$t1")
      quiet=$(api_get /api/status | json ".get('audio_quiet_s')")
      echo "audio_quiet_s=$quiet  OCR reads since wake=$(echo "$log" | grep -c 'OCR #')  VLM=$(echo "$log" | grep -c 'VLM #')"
      echo "$log" | grep -q 'OCR #' || FAIL+=("source cycle: OCR not running after wake")
      echo "$log" | grep -q 'VLM #' || FAIL+=("source cycle: VLM not running after wake")
      python3 -c "import sys; q=sys.argv[1]; sys.exit(0 if q not in ('','None') and float(q) < 10 else 1)" "$quiet" \
          || echo "note: audio quiet ${quiet}s (fine if the source is on a silent screen)"
    fi
    tb=$("${SUDO[@]}" journalctl -u minus --since "$t0" | grep -c Traceback)
    [ "$tb" = 0 ] || FAIL+=("source cycle: $tb Traceback(s)")
    # Put autonomous mode back.
    if [ "$auto_enabled" = True ]; then
      curl -s -XPOST "$API/api/autonomous/enable" >/dev/null
      [ "$auto_active" = True ] && curl -s -XPOST "$API/api/autonomous/start" >/dev/null
    fi
  fi
fi

echo
if [ ${#FAIL[@]} = 0 ]; then echo "E2E: ALL PASSED"; exit 0; fi
echo "E2E: FAILED -> ${FAIL[*]}"; exit 1
