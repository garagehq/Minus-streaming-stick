"""Playwright tests for the IR Remote panel in the Settings tab.

The Minus service must be running at http://localhost:80. These tests
exercise the live API, so each successful "send" leaves a real cooldown
window — the suite paces itself accordingly.

Run with:
    python3 tests/test_ir_ui.py
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
require_live_opt_in('test_ir_ui.py (fires the real IR blaster)')
SCREENSHOT_DIR = Path(__file__).parent / "screenshots" / "test_outputs"


def _shot(page, name):
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(SCREENSHOT_DIR / f"ir_{name}.png"))


def _open_settings(page):
    page.click("text=Settings")
    page.wait_for_timeout(500)


def _wait_panel_visible(page, expect_visible, timeout_ms=2000):
    deadline = time.monotonic() + timeout_ms / 1000.0
    while time.monotonic() < deadline:
        hidden = page.evaluate(
            "document.getElementById('ir-remote-panel').classList.contains('hidden')"
        )
        if hidden != expect_visible:
            return True
        page.wait_for_timeout(80)
    return False


@unittest.skipUnless(HAS_PLAYWRIGHT, "playwright not installed")
class TestIRRemoteUI(unittest.TestCase):
    """End-to-end UI tests against the running Minus service."""

    @classmethod
    def setUpClass(cls):
        cls.pw = sync_playwright().start()
        cls.browser = cls.pw.chromium.launch()
        # Seed the service to a known state: IR disabled at the start of the run.
        cls._reset_ir_disabled(cls.browser)

    @classmethod
    def tearDownClass(cls):
        # Leave the service in a known state regardless of test outcome.
        cls._reset_ir_disabled(cls.browser)
        cls.browser.close()
        cls.pw.stop()

    @staticmethod
    def _reset_ir_disabled(browser):
        ctx = browser.new_context()
        page = ctx.new_page()
        try:
            page.request.post(f"{BASE_URL}/api/ir/disable")
        finally:
            page.close()
            ctx.close()

    def setUp(self):
        self.page = self.browser.new_page(
            viewport={"width": 1280, "height": 900})
        self.page.goto(BASE_URL)
        self.page.wait_for_timeout(1500)

    def tearDown(self):
        # Reset between tests so cooldowns from one test don't bleed into the next.
        try:
            self.page.request.post(f"{BASE_URL}/api/ir/disable")
        finally:
            self.page.close()

    # ---------- structural / discovery ----------

    def test_toggle_present_in_autonomous_section(self):
        _open_settings(self.page)
        toggle = self.page.query_selector("#ir-toggle")
        self.assertIsNotNone(toggle, "IR toggle must exist in Settings")
        # Belongs inside the Autonomous Mode section, not stand-alone.
        section_h2 = self.page.evaluate(
            """() => {
                const t = document.getElementById('ir-toggle');
                let n = t; while (n && n.tagName !== 'SECTION') n = n.parentElement;
                return n ? n.querySelector('h2').textContent : null;
            }"""
        )
        self.assertEqual(section_h2, "Autonomous Mode")
        _shot(self.page, "01_toggle_visible")

    def test_panel_hidden_by_default(self):
        _open_settings(self.page)
        hidden = self.page.evaluate(
            "document.getElementById('ir-remote-panel').classList.contains('hidden')"
        )
        self.assertTrue(hidden, "Panel must start hidden when IR is disabled")

    def test_status_endpoint_reachable(self):
        resp = self.page.request.get(f"{BASE_URL}/api/ir/status")
        self.assertEqual(resp.status, 200)
        data = resp.json()
        for key in ("enabled", "available", "initialized", "codes"):
            self.assertIn(key, data)
        for name in ("input_1", "input_2", "input_3",
                     "power", "next", "auto"):
            self.assertIn(name, data["codes"])

    # ---------- toggle behavior ----------

    def test_toggle_on_reveals_panel(self):
        _open_settings(self.page)
        self.page.evaluate("document.getElementById(\"ir-toggle\").click()")
        self.assertTrue(_wait_panel_visible(self.page, expect_visible=True))
        # All six buttons must be there.
        buttons = self.page.query_selector_all("#ir-remote-panel button")
        self.assertEqual(len(buttons), 6)
        labels = [b.text_content().strip() for b in buttons]
        for needed in ("Input 1", "Input 2", "Input 3"):
            self.assertTrue(any(needed in l for l in labels),
                            f"missing label {needed!r} in {labels!r}")
        self.assertTrue(any("Power" in l for l in labels))
        self.assertTrue(any("Next" in l for l in labels))
        self.assertTrue(any("Auto" in l for l in labels))
        _shot(self.page, "02_panel_visible")

    def test_toggle_off_hides_panel(self):
        _open_settings(self.page)
        self.page.evaluate("document.getElementById(\"ir-toggle\").click()")
        self.assertTrue(_wait_panel_visible(self.page, expect_visible=True))
        self.page.evaluate("document.getElementById(\"ir-toggle\").click()")
        self.assertTrue(_wait_panel_visible(self.page, expect_visible=False))

    def test_toggle_on_persists_across_reload(self):
        _open_settings(self.page)
        self.page.evaluate("document.getElementById(\"ir-toggle\").click()")
        self.assertTrue(_wait_panel_visible(self.page, expect_visible=True))
        self.page.reload()
        self.page.wait_for_timeout(1500)
        _open_settings(self.page)
        # After reload, loadIRStatus() should hydrate from the server.
        checked = self.page.evaluate(
            "document.getElementById('ir-toggle').checked"
        )
        self.assertTrue(checked, "Toggle should reflect persisted ir_enabled")
        hidden = self.page.evaluate(
            "document.getElementById('ir-remote-panel').classList.contains('hidden')"
        )
        self.assertFalse(hidden, "Panel should be shown after reload")

    # ---------- send + cooldown behavior ----------

    def test_button_click_calls_command_endpoint(self):
        _open_settings(self.page)
        self.page.evaluate("document.getElementById(\"ir-toggle\").click()")
        self.assertTrue(_wait_panel_visible(self.page, expect_visible=True))

        seen = {"requests": []}

        def _on_request(req):
            if "/api/ir/command" in req.url:
                seen["requests"].append(req)

        self.page.on("request", _on_request)
        # Click any button — Power is the most distinctive.
        self.page.click("#ir-remote-panel button:has-text('Power')")
        self.page.wait_for_timeout(500)

        self.assertEqual(len(seen["requests"]), 1,
                         "exactly one command request expected")
        body = seen["requests"][0].post_data
        self.assertIn("power", body)
        # Status line should have been updated to "sent power".
        status = self.page.text_content("#ir-status-line")
        self.assertTrue(status and "power" in status.lower(),
                        f"unexpected status text: {status!r}")
        _shot(self.page, "03_after_send")

    def test_cooldown_disables_buttons(self):
        _open_settings(self.page)
        self.page.evaluate("document.getElementById(\"ir-toggle\").click()")
        self.assertTrue(_wait_panel_visible(self.page, expect_visible=True))

        self.page.click("#ir-remote-panel button:has-text('Power')")
        # Right after the click, JS should disable all buttons until cooldown clears.
        self.page.wait_for_timeout(150)
        all_disabled = self.page.evaluate(
            """() => {
                const btns = document.querySelectorAll('#ir-remote-panel button');
                return Array.from(btns).every(b => b.disabled);
            }"""
        )
        self.assertTrue(all_disabled,
                        "buttons should be disabled during cooldown")
        # And re-enabled after the 1.5s window.
        self.page.wait_for_timeout(1700)
        any_disabled = self.page.evaluate(
            """() => {
                const btns = document.querySelectorAll('#ir-remote-panel button');
                return Array.from(btns).some(b => b.disabled);
            }"""
        )
        self.assertFalse(any_disabled,
                         "buttons should re-enable after cooldown")

    def test_command_endpoint_blocks_without_toggle(self):
        # Toggle off; calls should fail with 403 even though hardware is present.
        resp = self.page.request.post(
            f"{BASE_URL}/api/ir/command",
            data={"button": "power"},
        )
        self.assertEqual(resp.status, 403)


@unittest.skipUnless(HAS_PLAYWRIGHT, "playwright not installed")
class TestIRHomeRemoteUI(unittest.TestCase):
    """Home-tab IR remote.

    When IR is enabled in Settings, an IR (HDMI-switch) remote is exposed on
    the Home tab:
      - Desktop (>=768px): sits beside the main remote and follows its open
        state; there is NO separate IR toggle button.
      - Mobile (<768px): its own separate dropdown driven by a dedicated
        "HDMI Switch" toggle; opening the main remote does not reveal it.
    All of it is gated behind body.ir-home-enabled (set only when IR is
    available AND enabled).
    """

    DESKTOP = {"width": 1280, "height": 900}
    MOBILE = {"width": 390, "height": 844}

    @classmethod
    def setUpClass(cls):
        cls.pw = sync_playwright().start()
        cls.browser = cls.pw.chromium.launch()

    @classmethod
    def tearDownClass(cls):
        cls._set_ir(cls.browser, enable=False)
        cls.browser.close()
        cls.pw.stop()

    @staticmethod
    def _set_ir(browser, enable):
        ctx = browser.new_context()
        page = ctx.new_page()
        try:
            ep = "/api/ir/enable" if enable else "/api/ir/disable"
            page.request.post(f"{BASE_URL}{ep}")
        finally:
            page.close()
            ctx.close()

    def _load(self, viewport, ir_enabled):
        # Server-side state must be set BEFORE load so loadIRStatus() hydrates
        # body.ir-home-enabled correctly.
        self._set_ir(self.browser, enable=ir_enabled)
        page = self.browser.new_page(viewport=viewport)
        page.goto(BASE_URL)
        page.wait_for_timeout(1500)
        return page

    def tearDown(self):
        self._set_ir(self.browser, enable=False)

    def test_desktop_beside_main_when_enabled(self):
        page = self._load(self.DESKTOP, ir_enabled=True)
        try:
            self.assertTrue(page.evaluate(
                "document.body.classList.contains('ir-home-enabled')"))
            # No separate IR toggle button on desktop.
            self.assertFalse(page.is_visible("#inline-ir-toggle-btn"))
            # IR remote stays hidden until the main remote is opened.
            self.assertFalse(page.is_visible("#inline-ir-remote"))
            page.evaluate("toggleInlineRemote()")
            page.wait_for_timeout(250)
            self.assertTrue(page.is_visible("#inline-remote"))
            self.assertTrue(page.is_visible("#inline-ir-remote"))
            btns = page.query_selector_all("#inline-ir-remote button")
            self.assertEqual(len(btns), 6)
            _shot(page, "home_desktop_sidebyside")
        finally:
            page.close()

    def test_desktop_absent_when_disabled(self):
        page = self._load(self.DESKTOP, ir_enabled=False)
        try:
            self.assertFalse(page.evaluate(
                "document.body.classList.contains('ir-home-enabled')"))
            page.evaluate("toggleInlineRemote()")
            page.wait_for_timeout(250)
            self.assertTrue(page.is_visible("#inline-remote"))
            self.assertFalse(page.is_visible("#inline-ir-remote"))
            self.assertFalse(page.is_visible("#inline-ir-toggle-btn"))
        finally:
            page.close()

    def test_mobile_separate_dropdown(self):
        page = self._load(self.MOBILE, ir_enabled=True)
        try:
            # Separate toggle button is shown on mobile when enabled.
            self.assertTrue(page.is_visible("#inline-ir-toggle-btn"))
            # Opening the main remote must NOT reveal the IR remote on mobile.
            page.evaluate("toggleInlineRemote()")
            page.wait_for_timeout(250)
            self.assertTrue(page.is_visible("#inline-remote"))
            self.assertFalse(page.is_visible("#inline-ir-remote"))
            # The IR toggle reveals it independently.
            page.evaluate("toggleInlineIRRemote()")
            page.wait_for_timeout(250)
            self.assertTrue(page.is_visible("#inline-ir-remote"))
            btns = page.query_selector_all("#inline-ir-remote button")
            self.assertEqual(len(btns), 6)
            _shot(page, "home_mobile_dropdown")
        finally:
            page.close()

    def test_mobile_toggle_absent_when_disabled(self):
        page = self._load(self.MOBILE, ir_enabled=False)
        try:
            self.assertFalse(page.is_visible("#inline-ir-toggle-btn"))
            page.evaluate("toggleInlineIRRemote()")
            page.wait_for_timeout(250)
            self.assertFalse(page.is_visible("#inline-ir-remote"))
        finally:
            page.close()

    def test_home_ir_button_sends_command(self):
        page = self._load(self.DESKTOP, ir_enabled=True)
        try:
            page.evaluate("toggleInlineRemote()")
            page.wait_for_timeout(250)
            seen = []
            page.on("request", lambda r: (
                seen.append(r.url) if "/api/ir/command" in r.url else None))
            page.click("#inline-ir-remote button:has-text('Power')")
            page.wait_for_timeout(600)
            self.assertEqual(
                len([u for u in seen if "/api/ir/command" in u]), 1,
                "exactly one IR command request expected")
            # Status line under the home IR remote should reflect the send.
            status = page.text_content("#inline-ir-status-line")
            self.assertTrue(status and "power" in status.lower(),
                            f"unexpected status: {status!r}")
        finally:
            page.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
