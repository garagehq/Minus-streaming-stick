"""Thermal monitoring + adaptive-degradation governor for Minus.

The RK3588 rides its ~85°C trip point during 4K passthrough and the kernel
throttles the CPU clocks (big cores 2304 -> ~1416MHz observed). The CPU-bound
half of the display pipeline (souphttpsrc / jpegparse) then can't sustain
60fps and the display oscillates around ~40fps, which looks worse than a
steady lower rate.

This module detects the throttled state and drives "thermal degraded" mode:
- display capped to a stable 30fps (frame gate in ad_blocker)
- blocking decision engine made stickier (longer min block duration, more
  consecutive no-ad frames to unblock) so slower OCR/VLM cadence under
  throttle doesn't flip-flop the overlay

Detection is primarily the kernel's own verdict: a `cpufreq-*` cooling
device with cur_state > 0 means the kernel IS actively capping CPU clocks —
no temperature guesswork. Temperature thresholds are the secondary signal
and drive the recovery hysteresis (states drop to 0 quickly once load
shrinks, but the SoC is still hot; we only restore full speed once it has
genuinely cooled).

Hysteresis (all env-overridable):
- enter: throttled OR temp >= MINUS_THERMAL_ENTER_C (83), sustained
  MINUS_THERMAL_ENTER_SUSTAIN (15s)
- exit: NOT throttled AND temp <= MINUS_THERMAL_EXIT_C (75), sustained
  MINUS_THERMAL_EXIT_SUSTAIN (60s)
"""

import os
import glob
import logging
import threading
import time

logger = logging.getLogger(__name__)

ENTER_TEMP_C = float(os.environ.get('MINUS_THERMAL_ENTER_C', '83'))
EXIT_TEMP_C = float(os.environ.get('MINUS_THERMAL_EXIT_C', '75'))
# A cpufreq cooling device reporting cur_state > 0 is the kernel's own
# "I am capping clocks" signal, but on RK3588 it can sit non-zero while the
# SoC is merely warm — observed live entering DEGRADED at 65.6°C and 67.5°C,
# which is nowhere near heat-limited and just churned the fps cap and the
# blocking thresholds. Require the throttle signal to be corroborated by a
# temperature floor; the standalone ENTER_TEMP_C path still catches a hot SoC
# whose cooling state we failed to read.
THROTTLE_MIN_TEMP_C = float(os.environ.get('MINUS_THERMAL_THROTTLE_MIN_C', '78'))
ENTER_SUSTAIN_S = float(os.environ.get('MINUS_THERMAL_ENTER_SUSTAIN', '15'))
EXIT_SUSTAIN_S = float(os.environ.get('MINUS_THERMAL_EXIT_SUSTAIN', '60'))
CHECK_INTERVAL_S = float(os.environ.get('MINUS_THERMAL_CHECK_INTERVAL', '5'))

_THERMAL_ZONE_GLOB = '/sys/class/thermal/thermal_zone*'
_COOLING_DEVICE_GLOB = '/sys/class/thermal/cooling_device*'


def read_max_temp_c():
    """Max temperature across all SoC thermal zones, in °C (None if unreadable)."""
    temps = []
    for path in glob.glob(_THERMAL_ZONE_GLOB + '/temp'):
        try:
            with open(path) as f:
                temps.append(int(f.read().strip()) / 1000.0)
        except (OSError, ValueError):
            continue
    return max(temps) if temps else None


def read_cpu_throttled():
    """True if the kernel is actively capping CPU clocks for thermal reasons.

    A `cpufreq-cpuN` cooling device with cur_state > 0 is the kernel's own
    statement that it is throttling — far more reliable than inferring from
    frequencies (which also drop under light load with schedutil).
    """
    for dev in glob.glob(_COOLING_DEVICE_GLOB):
        try:
            with open(dev + '/type') as f:
                ctype = f.read().strip()
            if not ctype.startswith('cpufreq'):
                continue
            with open(dev + '/cur_state') as f:
                if int(f.read().strip()) > 0:
                    return True
        except (OSError, ValueError):
            continue
    return False


class ThermalGovernor:
    """Pure hysteresis state machine — no sysfs, no threads (unit-testable).

    update() is fed (temp, throttled, now) samples and returns True whenever
    the degraded state changed. Sustain windows on both edges stop a
    borderline SoC from flapping the mode itself.
    """

    def __init__(self, enter_temp=None, exit_temp=None,
                 enter_sustain=None, exit_sustain=None, throttle_min_temp=None):
        self.enter_temp = ENTER_TEMP_C if enter_temp is None else enter_temp
        self.exit_temp = EXIT_TEMP_C if exit_temp is None else exit_temp
        self.throttle_min_temp = (THROTTLE_MIN_TEMP_C if throttle_min_temp is None
                                  else throttle_min_temp)
        self.enter_sustain = ENTER_SUSTAIN_S if enter_sustain is None else enter_sustain
        self.exit_sustain = EXIT_SUSTAIN_S if exit_sustain is None else exit_sustain
        self.degraded = False
        self._hot_since = None
        self._cool_since = None

    def update(self, temp_c, throttled, now):
        """Feed one sample. Returns True if self.degraded changed."""
        # Throttling only counts as "hot" when corroborated by temperature.
        # If temp is unreadable we trust the kernel's throttle signal alone
        # (better to degrade than to ignore a real thermal cap).
        throttled_hot = bool(throttled) and (
            temp_c is None or temp_c >= self.throttle_min_temp)
        hot = throttled_hot or (temp_c is not None and temp_c >= self.enter_temp)
        # Exit still requires the throttle signal to be fully clear, so we
        # never drop the cap while the kernel is actively capping clocks.
        cool = (not throttled) and (temp_c is None or temp_c <= self.exit_temp)

        if not self.degraded:
            self._cool_since = None
            if hot:
                if self._hot_since is None:
                    self._hot_since = now
                if now - self._hot_since >= self.enter_sustain:
                    self.degraded = True
                    self._hot_since = None
                    return True
            else:
                self._hot_since = None
        else:
            self._hot_since = None
            if cool:
                if self._cool_since is None:
                    self._cool_since = now
                if now - self._cool_since >= self.exit_sustain:
                    self.degraded = False
                    self._cool_since = None
                    return True
            else:
                self._cool_since = None
        return False


class ThermalMonitor:
    """Background thread sampling sysfs and driving the governor.

    on_change(degraded: bool, status: dict) fires on every transition.
    `force` (True/False/None) pins the mode for testing via
    POST /api/test/thermal; None resumes normal governing.
    """

    def __init__(self, on_change=None):
        self.governor = ThermalGovernor()
        self.force = None
        self.last_temp_c = None
        self.last_throttled = False
        self._on_change = on_change
        self._running = False
        self._thread = None

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name='thermal-monitor')
        self._thread.start()
        logger.info(f"[Thermal] Monitor started (enter: throttled at "
                    f">={self.governor.throttle_min_temp:.0f}°C, or "
                    f">={self.governor.enter_temp:.0f}°C alone, for "
                    f"{self.governor.enter_sustain:.0f}s; exit: unthrottled and "
                    f"<={self.governor.exit_temp:.0f}°C for "
                    f"{self.governor.exit_sustain:.0f}s)")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=CHECK_INTERVAL_S + 2)

    def check_now(self):
        """Run one synchronous sample/decide cycle (used by the test API)."""
        self._check()

    def get_status(self):
        return {
            'degraded': self.governor.degraded,
            'temp_c': self.last_temp_c,
            'throttled': self.last_throttled,
            'forced': self.force,
        }

    def _check(self):
        self.last_temp_c = read_max_temp_c()
        self.last_throttled = read_cpu_throttled()

        if self.force is not None:
            changed = (self.force != self.governor.degraded)
            self.governor.degraded = self.force
            # Reset edge timers so un-forcing re-evaluates cleanly
            self.governor._hot_since = None
            self.governor._cool_since = None
        else:
            changed = self.governor.update(self.last_temp_c, self.last_throttled,
                                           time.monotonic())

        if changed:
            logger.warning(f"[Thermal] Mode change -> "
                           f"{'DEGRADED' if self.governor.degraded else 'NORMAL'} "
                           f"(temp={self.last_temp_c}, throttled={self.last_throttled}, "
                           f"forced={self.force})")
            if self._on_change:
                try:
                    self._on_change(self.governor.degraded, self.get_status())
                except Exception as e:
                    logger.error(f"[Thermal] on_change callback failed: {e}")

    def _loop(self):
        while self._running:
            try:
                self._check()
            except Exception as e:
                logger.error(f"[Thermal] Monitor error: {e}")
            time.sleep(CHECK_INTERVAL_S)
