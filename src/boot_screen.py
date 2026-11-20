"""KingBlocker9000 boot splash hand-off and Minus's full-screen text screens.

  boot splash   Plymouth (plymouth/kingblocker): logo + "A Kalshi INVENTION"
                + white loading bar on #28CC95, from early boot until Minus
                draws. Its bar follows Plymouth's boot progress (learned from
                the previous boot's timing) and runs to 100% when Minus takes
                over (release_plymouth).
  INITIALIZING  Minus's loading display (DRMAdBlocker.start_loading_mode,
                while it starts up with an input) and
  NO SIGNAL     its no-input display (start_no_signal_mode): white heavy
                italic text centred on the same green, static.

The images are in assets/screens/ (tools/render_screens.py). They go
straight to kmssink (the TV), never through ustreamer, so OCR never sees
them: "KING" on screen cannot trigger the name muter. If an image is
missing, the callers fall back to the old text screens.
"""
import logging
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent.parent
ASSET_DIR = Path(os.environ.get('MINUS_SCREENS_DIR', str(REPO / 'assets' / 'screens')))
NO_SIGNAL_IMAGE = ASSET_DIR / 'no_signal.png'
INITIALIZING_IMAGE = ASSET_DIR / 'initializing.png'

WIDTH, HEIGHT = 1920, 1080

# Status kingblocker.script reacts to (runs the bar to 100%), and how long
# that takes before Plymouth is told to quit.
PLYMOUTH_FINISH = 'kingblocker-finish'
PLYMOUTH_END_ANIM_S = 0.6

_plymouth_released = False


def release_plymouth():
    """Run the boot splash's bar to 100% and quit Plymouth, once, right
    before Minus's first screen.

    tools/install_boot_screen.sh turns plymouth-quit.service into a no-op, so
    the splash stays up past the end of boot (~9 s) until Minus is ready to
    draw (~25-30 s) instead of leaving the TV black in between. Plymouth holds
    DRM master while it runs, so this must happen before any kmssink or
    modetest -w. A no-op when Plymouth isn't running.
    """
    global _plymouth_released
    if _plymouth_released:
        return
    _plymouth_released = True
    import shutil
    import subprocess
    plymouth = shutil.which('plymouth')
    if not plymouth:
        return
    try:
        if subprocess.run([plymouth, '--ping'], timeout=3).returncode != 0:
            return
        subprocess.run([plymouth, 'update', f'--status={PLYMOUTH_FINISH}'], timeout=3)
        time.sleep(PLYMOUTH_END_ANIM_S)
        subprocess.run([plymouth, 'quit', '--retain-splash'], timeout=10)
        logger.info("[BootScreen] Plymouth splash released")
    except Exception as e:
        logger.warning(f"[BootScreen] Could not quit Plymouth: {e}")


def screen_image(name: str):
    """Path of the 'no_signal' / 'initializing' image, or None if it can't
    be used (missing, or MINUS_BOOT_SCREEN=0)."""
    if os.environ.get('MINUS_BOOT_SCREEN', '1') == '0':
        return None
    path = {'no_signal': NO_SIGNAL_IMAGE, 'initializing': INITIALIZING_IMAGE}[name]
    return path if path.exists() else None
