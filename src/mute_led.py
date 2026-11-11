"""Mute LED: a red LED on the 40-pin header that flashes while a name is muted.

Wiring (ROCK 5B+): Pin 40 (GPIO3_B3, gpiochip3 line 11) -> 1 kOhm resistor
-> LED anode (+), LED cathode (-) -> Pin 39 (GND).

Driven through sysfs GPIO so it needs no extra Python packages (works the
same from source and from the frozen executable). Must run as root or as a
member of the `gpio` group. If the GPIO can't be opened the LED is simply
disabled; the muter never fails because of it.

`set_muted(True/False)` is called from the name-mute scheduler's
on_mute_change hook. While muted the LED blinks; every mute shows at least
MIN_FLASHES blinks so even a ~0.5 s mute is visible.

Env overrides: MINUS_MUTE_LED=0 disables it; MINUS_MUTE_LED_CHIP (sysfs
chip label, default "gpio3") and MINUS_MUTE_LED_LINE (default 11) move it.
"""

import logging
import os
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

SYSFS = Path('/sys/class/gpio')
BLINK_ON_S = 0.12
BLINK_OFF_S = 0.08
MIN_FLASHES = 3


def _global_gpio(chip_label: str, line: int):
    for chip in SYSFS.glob('gpiochip*'):
        try:
            if (chip / 'label').read_text().strip() == chip_label:
                return int((chip / 'base').read_text()) + line
        except OSError:
            continue
    return None


class MuteLED:
    def __init__(self, chip_label: str = None, line: int = None):
        self.chip_label = chip_label or os.environ.get('MINUS_MUTE_LED_CHIP', 'gpio3')
        self.line = int(line if line is not None else os.environ.get('MINUS_MUTE_LED_LINE', 11))
        self.available = False
        self._value_path = None
        self._muted = False
        self._flashes_left = 0
        self._cond = threading.Condition()
        self._stop = False
        self._thread = None
        if os.environ.get('MINUS_MUTE_LED', '1') == '0':
            logger.info("[MuteLED] disabled (MINUS_MUTE_LED=0)")
            return
        try:
            self._open()
            self.available = True
            logger.info(f"[MuteLED] ready on {self.chip_label} line {self.line}")
        except Exception as e:
            logger.info(f"[MuteLED] not available ({e}); mute LED disabled")

    def _open(self):
        num = _global_gpio(self.chip_label, self.line)
        if num is None:
            raise RuntimeError(f"no sysfs gpiochip labelled {self.chip_label!r}")
        gpio_dir = SYSFS / f'gpio{num}'
        if not gpio_dir.exists():
            (SYSFS / 'export').write_text(str(num))
            for _ in range(50):  # udev may take a moment to fix permissions
                if (gpio_dir / 'direction').exists():
                    break
                time.sleep(0.02)
        for attempt in range(20):
            try:
                (gpio_dir / 'direction').write_text('low')
                break
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(0.05)
        self._value_path = gpio_dir / 'value'

    def _write(self, on: bool):
        try:
            self._value_path.write_text('1' if on else '0')
        except OSError as e:
            logger.debug(f"[MuteLED] write failed: {e}")

    def start(self):
        if not self.available or self._thread is not None:
            return
        self._stop = False
        self._thread = threading.Thread(target=self._run, name='mute-led', daemon=True)
        self._thread.start()

    def stop(self):
        with self._cond:
            self._stop = True
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self.available:
            self._write(False)

    def set_muted(self, muted: bool):
        if not self.available:
            return
        with self._cond:
            if muted and not self._muted:
                self._flashes_left = MIN_FLASHES
            self._muted = bool(muted)
            self._cond.notify_all()

    def _run(self):
        while True:
            with self._cond:
                while not self._stop and not self._muted and self._flashes_left <= 0:
                    self._cond.wait()
                if self._stop:
                    return
                self._flashes_left -= 1
            self._write(True)
            time.sleep(BLINK_ON_S)
            self._write(False)
            time.sleep(BLINK_OFF_S)
