"""Playwright tests for the Status LEDs panel in the Settings tab.

The Minus service must be running at http://localhost:80. These tests
exercise the live API and assume the SPI overlay is enabled (so the
toggle isn't disabled). Tests reset the service to a known state
(enabled=False) before and after the suite so they don't leave the
strip animating after the run.

Run with:
    python3 tests/test_status_leds_ui.py
"""

import os
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from playwright.sync_api import sync_playwright
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False
    print("WARNING: playwright not installed, skipping UI tests")

BASE_URL = "http://localhost:80"

# Live test: only with MINUS_LIVE_UI_TESTS=1 (tests/live_guard.py explains why).
sys.path.insert(0, str(Path(__file__).parent))
from live_guard import require_live_opt_in  # noqa: E402
require_live_opt_in('test_status_leds_ui.py (drives the real status LEDs)')
SCREENSHOT_DIR = Path(__file__).parent / "screenshots" / "test_outputs"


def _shot(page, name):
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(SCREENSHOT_DIR / f"leds_{name}.png"))


def _open_settings(page):
    page.click("text=Settings")
    page.wait_for_timeout(500)


def _wait_panel_visible(page, expect_visible, timeout_ms=2000):
    deadline = time.monotonic() + timeout_ms / 1000.0
    while time.monotonic() < deadline:
        hidden = page.evaluate(
            "document.getElementById('leds-state-panel').classList.contains('hidden')"
        )
        if hidden != expect_visible:
            return True
        page.wait_for_timeout(80)
    return False


@unittest.skipUnless(HAS_PLAYWRIGHT, "playwright not installed")
class TestStatusLEDsUI(unittest.TestCase):
    """End-to-end UI tests against the running Minus service."""

    @classmethod
    def setUpClass(cls):
        cls.pw = sync_playwright().start()
        cls.browser = cls.pw.chromium.launch()
        cls._reset_disabled(cls.browser)

    @classmethod
    def tearDownClass(cls):
        cls._reset_disabled(cls.browser)
        cls.browser.close()
        cls.pw.stop()

    @staticmethod
    def _reset_disabled(browser):
        ctx = browser.new_context()
        page = ctx.new_page()
        try:
            page.request.post(f"{BASE_URL}/api/leds/disable")
        finally:
            page.close()
            ctx.close()

    def setUp(self):
        self.page = self.browser.new_page(
            viewport={"width": 1280, "height": 900})
        self.page.goto(BASE_URL)
        self.page.wait_for_timeout(1500)

    def tearDown(self):
        try:
            self.page.request.post(f"{BASE_URL}/api/leds/disable")
        finally:
            self.page.close()

    # ---------------------------------------------- structural / discovery

    def test_toggle_present_in_autonomous_section(self):
        _open_settings(self.page)
        toggle = self.page.query_selector("#leds-toggle")
        self.assertIsNotNone(toggle, "LED toggle must exist in Settings")
        section_h2 = self.page.evaluate(
            """() => {
                const t = document.getElementById('leds-toggle');
                let n = t; while (n && n.tagName !== 'SECTION') n = n.parentElement;
                return n ? n.querySelector('h2').textContent : null;
            }"""
        )
        self.assertEqual(section_h2, "Autonomous Mode")
        _shot(self.page, "01_toggle_visible")

    def test_panel_hidden_by_default(self):
        _open_settings(self.page)
        hidden = self.page.evaluate(
            "document.getElementById('leds-state-panel').classList.contains('hidden')"
        )
        self.assertTrue(hidden, "Panel must start hidden when LEDs are disabled")

    def test_status_endpoint_reachable(self):
        resp = self.page.request.get(f"{BASE_URL}/api/leds/status")
        self.assertEqual(resp.status, 200)
        data = resp.json()
        for key in ("enabled", "available", "running", "state", "states"):
            self.assertIn(key, data)
        # At least the canonical states should be present
        for needed in ("off", "initializing", "idle", "blocking",
                       "no_signal", "autonomous", "error"):
            self.assertIn(needed, data["states"])

    # -------------------------------------------------------- toggle behavior

    def test_toggle_on_reveals_panel(self):
        _open_settings(self.page)
        self.page.evaluate("document.getElementById(\"leds-toggle\").click()")
        self.assertTrue(_wait_panel_visible(self.page, expect_visible=True))
        buttons = self.page.query_selector_all(
            "#leds-state-panel button[data-led-state]")
        self.assertEqual(len(buttons), 7)
        labels = [b.text_content().strip() for b in buttons]
        for needed in ("Off", "Init", "Idle", "Blocking",
                       "No Signal", "Autonomous", "Error"):
            self.assertTrue(any(needed in l for l in labels),
                            f"missing label {needed!r} in {labels!r}")
        _shot(self.page, "02_panel_visible")

    def test_toggle_off_hides_panel(self):
        _open_settings(self.page)
        self.page.evaluate("document.getElementById(\"leds-toggle\").click()")
        self.assertTrue(_wait_panel_visible(self.page, expect_visible=True))
        self.page.evaluate("document.getElementById(\"leds-toggle\").click()")
        self.assertTrue(_wait_panel_visible(self.page, expect_visible=False))

    def test_toggle_on_persists_across_reload(self):
        _open_settings(self.page)
        self.page.evaluate("document.getElementById(\"leds-toggle\").click()")
        self.assertTrue(_wait_panel_visible(self.page, expect_visible=True))
        self.page.reload()
        self.page.wait_for_timeout(1500)
        _open_settings(self.page)
        # After reload, loadStatusLEDs() should hydrate from the server. Wait
        # for it rather than a fixed delay: on a box busy blocking an ad the
        # status fetch can take longer than 1.5 s (flaked once, 2026-10-19).
        try:
            self.page.wait_for_function(
                "document.getElementById('leds-toggle').checked", timeout=8000)
            toggle_checked = True
        except Exception:
            toggle_checked = False
        self.assertTrue(toggle_checked, "Toggle should remember 'enabled'")
        self.assertTrue(_wait_panel_visible(self.page, expect_visible=True))

    # ------------------------------------------------------- state selection

    def test_clicking_state_button_calls_api(self):
        _open_settings(self.page)
        self.page.evaluate("document.getElementById(\"leds-toggle\").click()")
        self.assertTrue(_wait_panel_visible(self.page, expect_visible=True))

        recorded = {}
        def on_request(req):
            if req.url.endswith("/api/leds/state") and req.method == "POST":
                try:
                    body = req.post_data_json or {}
                except Exception:
                    body = {}
                recorded["state"] = body.get("state")
        self.page.on("request", on_request)

        self.page.click("button[data-led-state='blocking']")
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and "state" not in recorded:
            self.page.wait_for_timeout(80)
        self.assertEqual(recorded.get("state"), "blocking")

        # And the active class should be reflected in the DOM
        deadline = time.monotonic() + 2.0
        active_state = None
        while time.monotonic() < deadline:
            active_state = self.page.evaluate(
                "document.querySelector('button.active[data-led-state]')?.dataset.ledState"
            )
            if active_state == "blocking":
                break
            self.page.wait_for_timeout(80)
        self.assertEqual(active_state, "blocking")

    def test_state_endpoint_rejects_unknown(self):
        # Don't need the toggle on for this — we're hitting the API directly.
        self.page.request.post(f"{BASE_URL}/api/leds/enable")
        try:
            resp = self.page.request.post(
                f"{BASE_URL}/api/leds/state",
                data={"state": "rainbow_explosion"},
                headers={"Content-Type": "application/json"},
            )
            self.assertEqual(resp.status, 400)
            data = resp.json()
            self.assertFalse(data["success"])
            self.assertIn("states", data)
        finally:
            self.page.request.post(f"{BASE_URL}/api/leds/disable")

    def test_state_endpoint_403_when_disabled(self):
        # Service starts disabled (setUpClass), so this should get blocked.
        resp = self.page.request.post(
            f"{BASE_URL}/api/leds/state",
            data={"state": "idle"},
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(resp.status, 403)
        data = resp.json()
        self.assertFalse(data["success"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
