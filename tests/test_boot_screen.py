"""
Boot splash hand-off and the green INITIALIZING / NO SIGNAL screens
(src/boot_screen.py, DRMAdBlocker's loading / no-signal pipelines,
plymouth/kingblocker, tools/install_boot_screen.sh).

Run: python3 -m unittest tests.test_boot_screen
"""
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'src'))

import boot_screen as bs  # noqa: E402

GREEN_RGB = (0x28, 0xCC, 0x95)
SPLASH_FILES = ('boot_background.png', 'bar_track.png', 'bar_fill_left.png',
                'bar_fill_mid.png', 'bar_fill_right.png')


class TestAssets(unittest.TestCase):

    def test_screens_present_green_and_full_hd(self):
        from PIL import Image
        for name in ('no_signal', 'initializing'):
            path = bs.screen_image(name)
            self.assertIsNotNone(path, name)
            img = Image.open(path).convert('RGB')
            self.assertEqual(img.size, (bs.WIDTH, bs.HEIGHT))
            self.assertEqual(img.getpixel((5, 5)), GREEN_RGB)
            self.assertEqual(img.getpixel((bs.WIDTH - 5, bs.HEIGHT - 5)), GREEN_RGB)
            # White text in the middle band, nothing else on the screen.
            mid = img.crop((0, 400, bs.WIDTH, 680))
            self.assertTrue(any(px == (255, 255, 255) for px in mid.getdata()), name)

    def test_disable_switch(self):
        with patch.dict('os.environ', {'MINUS_BOOT_SCREEN': '0'}):
            self.assertIsNone(bs.screen_image('no_signal'))
            self.assertIsNone(bs.screen_image('initializing'))

    def test_splash_background_has_no_bar(self):
        from PIL import Image
        img = Image.open(bs.ASSET_DIR / 'boot_background.png').convert('RGB')
        self.assertEqual(img.size, (bs.WIDTH, bs.HEIGHT))
        self.assertEqual({img.getpixel((x, 955)) for x in range(670, 1250)}, {GREEN_RGB})

    def test_plymouth_theme_files(self):
        script = (REPO / 'plymouth' / 'kingblocker' / 'kingblocker.script').read_text()
        for name in SPLASH_FILES:
            self.assertIn(name, script)
            self.assertTrue((bs.ASSET_DIR / name).exists(), name)
        # The bar finishes on Minus's status, from the same string Minus sends.
        self.assertIn(f'status == "{bs.PLYMOUTH_FINISH}"', script)
        sh = (REPO / 'tools' / 'install_boot_screen.sh').read_text()
        for name in SPLASH_FILES:
            self.assertIn(name, sh)


class TestReleasePlymouth(unittest.TestCase):
    """Minus finishes the splash's bar and quits it, once, before its first
    screen."""

    def setUp(self):
        bs._plymouth_released = False

    def tearDown(self):
        bs._plymouth_released = True

    def _run(self, ping_rc):
        runs = []

        def run(args, timeout):
            runs.append(args[1:])
            return MagicMock(returncode=ping_rc)
        return runs, run

    def test_finishes_the_bar_then_quits_once(self):
        runs, run = self._run(0)
        with patch('shutil.which', return_value='/usr/bin/plymouth'), \
                patch('subprocess.run', side_effect=run), \
                patch.object(bs.time, 'sleep') as sleep:
            bs.release_plymouth()
            bs.release_plymouth()
        self.assertEqual(runs, [['--ping'], ['update', f'--status={bs.PLYMOUTH_FINISH}'],
                                ['quit', '--retain-splash']])
        sleep.assert_called_once_with(bs.PLYMOUTH_END_ANIM_S)   # bar runs to 100% first

    def test_nothing_when_not_running(self):
        runs, run = self._run(1)
        with patch('shutil.which', return_value='/usr/bin/plymouth'), \
                patch('subprocess.run', side_effect=run):
            bs.release_plymouth()
        self.assertEqual(runs, [['--ping']])

    def test_no_plymouth_installed(self):
        with patch('shutil.which', return_value=None), patch('subprocess.run') as r:
            bs.release_plymouth()
        r.assert_not_called()

    def test_installer_keeps_splash_until_minus(self):
        sh = (REPO / 'tools' / 'install_boot_screen.sh').read_text()
        self.assertIn('dropin plymouth-quit-wait.service "-/bin/true"', sh)
        self.assertIn('--on-active=180', sh)          # failsafe if Minus never draws
        self.assertIn('plymouth.ignore-serial-consoles', sh)


class TestPipelines(unittest.TestCase):
    """The pipelines DRMAdBlocker builds for its two screens."""

    def setUp(self):
        bs._plymouth_released = True     # never run the real plymouth here

    def _blocker(self):
        import ad_blocker
        b = object.__new__(ad_blocker.DRMAdBlocker)
        b.pipeline = None; b.bus = None; b.connector_id = 215; b.plane_id = 192
        b._stop_watchdog_thread = lambda: None
        b._stop_loading_animation = lambda: None
        b._stop_no_signal_animation = lambda: None
        b._start_loading_animation = MagicMock()
        b._start_no_signal_animation = MagicMock()
        b._set_led_state = lambda *_: None
        return ad_blocker, b

    def _start(self, method, available=True):
        ab, b = self._blocker()
        with patch.object(ab, 'probe_drm_output', return_value={'connector_id': 215}), \
                patch.dict('os.environ', {'MINUS_BOOT_SCREEN': '1' if available else '0'}), \
                patch.object(ab, 'Gst') as G:
            G.StateChangeReturn.FAILURE = 'fail'
            G.parse_launch.return_value.set_state.return_value = 'ok'
            if method == 'no_signal':
                self.assertTrue(b.start_no_signal_mode(skip_dpms=True))
            else:
                self.assertTrue(b.start_loading_mode())
        return b, G.parse_launch.call_args[0][0]

    def test_no_signal_is_the_static_image(self):
        b, desc = self._start('no_signal')
        self.assertIn('no_signal.png', desc)
        self.assertIn('imagefreeze is-live=true', desc)
        self.assertNotIn('textoverlay', desc)
        b._start_no_signal_animation.assert_not_called()     # no bouncing

    def test_initializing_is_the_static_image(self):
        b, desc = self._start('loading')
        self.assertIn('initializing.png', desc)
        self.assertIn('imagefreeze is-live=true', desc)
        self.assertNotIn('textoverlay', desc)
        b._start_loading_animation.assert_not_called()

    def test_fall_back_to_text(self):
        b, desc = self._start('no_signal', available=False)
        self.assertIn('textoverlay', desc)
        b._start_no_signal_animation.assert_called_once()
        b, desc = self._start('loading', available=False)
        self.assertIn('INITIALIZING', desc)
        b._start_loading_animation.assert_called_once()


if __name__ == '__main__':
    unittest.main()
