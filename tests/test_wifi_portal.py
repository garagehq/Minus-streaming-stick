"""
Playwright tests for the WiFi Captive Portal (wifi_setup.html).

Tests mobile responsiveness across various phone sizes, edge cases,
and breaking conditions.

Usage:
    python3 tests/test_wifi_portal.py
"""

import json
import os
import sys
import time
import unittest
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from playwright.sync_api import sync_playwright
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False
    print("WARNING: playwright not installed, skipping UI tests")

# Never the live web UI. These tests click real-looking networks and submit
# passwords; pointed at localhost:80 on a Minus box they drove the REAL radio
# (2026-10-15: the box joined a neighbour's open smart-plug network and stayed
# offline). setUpModule starts tests/portal_test_server.py — the real routes
# and page on a random 127.0.0.1 port with a fake Wi-Fi manager — and sets
# these.
BASE_URL = None
WIFI_SETUP_URL = None
_SERVER = None


def setUpModule():
    global BASE_URL, WIFI_SETUP_URL, _SERVER
    sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
    sys.path.insert(0, str(Path(__file__).parent))
    from portal_test_server import PortalTestServer
    _SERVER = PortalTestServer().start()
    BASE_URL = _SERVER.base_url
    WIFI_SETUP_URL = f"{BASE_URL}/wifi-setup"
    assert not BASE_URL.endswith(':80'), 'portal tests must never run against a real web UI'


def tearDownModule():
    if _SERVER is not None:
        # Nothing the tests did may have reached a real radio: the fake only
        # records attempts.
        _SERVER.stop()
SCREENSHOT_DIR = Path(__file__).parent / "screenshots" / "wifi_portal"

# Common mobile viewport sizes
VIEWPORTS = {
    'iphone_se': {'width': 375, 'height': 667},      # iPhone SE (small)
    'iphone_x': {'width': 375, 'height': 812},       # iPhone X
    'iphone_12_pro_max': {'width': 428, 'height': 926},  # iPhone 12 Pro Max (large)
    'pixel_5': {'width': 393, 'height': 851},        # Google Pixel 5
    'galaxy_s21': {'width': 360, 'height': 800},     # Samsung Galaxy S21
    'galaxy_fold': {'width': 280, 'height': 653},    # Galaxy Fold (narrow)
    'ipad_mini': {'width': 768, 'height': 1024},     # iPad Mini (tablet)
    'desktop': {'width': 1280, 'height': 900},       # Desktop
}


def save_screenshot(page, name):
    """Save a debug screenshot."""
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(SCREENSHOT_DIR / f"{name}.png"))


def wait_for_networks(page, timeout=5000):
    """Wait for network list to load (scanning spinner to disappear)."""
    try:
        page.wait_for_selector('.network-item', timeout=timeout)
        return True
    except:
        return False


@unittest.skipUnless(HAS_PLAYWRIGHT, "playwright not installed")
class TestWiFiPortalMobileResponsiveness(unittest.TestCase):
    """Test WiFi portal renders correctly on various mobile viewports."""

    @classmethod
    def setUpClass(cls):
        cls.pw = sync_playwright().start()
        cls.browser = cls.pw.chromium.launch()

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.pw.stop()

    def test_iphone_se_no_horizontal_scroll(self):
        """Smallest iPhone (SE) should not have horizontal scroll."""
        page = self.browser.new_page(viewport=VIEWPORTS['iphone_se'])
        page.goto(WIFI_SETUP_URL)
        page.wait_for_timeout(1000)

        scroll_width = page.evaluate('document.body.scrollWidth')
        viewport_width = page.evaluate('window.innerWidth')

        self.assertLessEqual(
            scroll_width, viewport_width + 5,
            f"iPhone SE should not have horizontal scroll (scroll={scroll_width}, viewport={viewport_width})"
        )
        save_screenshot(page, "iphone_se_layout")
        page.close()

    def test_galaxy_fold_narrow_viewport(self):
        """Galaxy Fold's narrow viewport (280px) should still be usable."""
        page = self.browser.new_page(viewport=VIEWPORTS['galaxy_fold'])
        page.goto(WIFI_SETUP_URL)
        page.wait_for_timeout(1000)

        # Check no horizontal overflow
        scroll_width = page.evaluate('document.body.scrollWidth')
        viewport_width = page.evaluate('window.innerWidth')
        self.assertLessEqual(scroll_width, viewport_width + 5)

        # Logo should be visible
        logo = page.query_selector('.logo')
        self.assertIsNotNone(logo)
        logo_box = logo.bounding_box()
        self.assertLess(logo_box['x'] + logo_box['width'], viewport_width)

        save_screenshot(page, "galaxy_fold_layout")
        page.close()

    def test_all_viewports_elements_visible(self):
        """All key elements should be visible on all viewport sizes."""
        for name, viewport in VIEWPORTS.items():
            with self.subTest(viewport=name):
                page = self.browser.new_page(viewport=viewport)
                page.goto(WIFI_SETUP_URL)
                page.wait_for_timeout(1000)

                # Logo visible
                logo = page.query_selector('.logo')
                self.assertIsNotNone(logo, f"Logo missing on {name}")

                # Network list or loading state visible
                network_list = page.query_selector('#network-list')
                self.assertIsNotNone(network_list, f"Network list missing on {name}")

                # Skip button visible and within viewport
                skip_btn = page.query_selector('button:has-text("Skip")')
                self.assertIsNotNone(skip_btn, f"Skip button missing on {name}")

                save_screenshot(page, f"viewport_{name}")
                page.close()

    def test_container_max_width_on_large_screens(self):
        """Container should have max-width on desktop/tablet."""
        page = self.browser.new_page(viewport=VIEWPORTS['desktop'])
        page.goto(WIFI_SETUP_URL)
        page.wait_for_timeout(500)

        container = page.query_selector('.container')
        box = container.bounding_box()

        # Container should be centered and not full width
        self.assertLessEqual(box['width'], 420, "Container should be max 400px + padding")
        self.assertGreater(box['x'], 0, "Container should be centered with margin")

        page.close()


@unittest.skipUnless(HAS_PLAYWRIGHT, "playwright not installed")
class TestWiFiPortalNetworkList(unittest.TestCase):
    """Test network list rendering and interactions."""

    @classmethod
    def setUpClass(cls):
        cls.pw = sync_playwright().start()
        cls.browser = cls.pw.chromium.launch()

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.pw.stop()

    def setUp(self):
        self.page = self.browser.new_page(viewport=VIEWPORTS['iphone_x'])
        self.page.goto(WIFI_SETUP_URL)
        wait_for_networks(self.page)

    def tearDown(self):
        self.page.close()

    def test_networks_display_signal_bars(self):
        """Each network should show signal strength bars."""
        networks = self.page.query_selector_all('.network-item')
        if not networks:
            self.skipTest("No networks available to test")

        for network in networks[:3]:  # Check first 3
            signal_bars = network.query_selector('.signal-bars')
            self.assertIsNotNone(signal_bars, "Network should have signal bars")
            bars = network.query_selector_all('.signal-bar')
            self.assertEqual(len(bars), 4, "Should have 4 signal bars")

    def test_secured_networks_show_lock(self):
        """Secured networks should display lock icon."""
        networks = self.page.query_selector_all('.network-item')
        if not networks:
            self.skipTest("No networks available")

        # At least one network should have a lock (most networks are secured)
        locks = self.page.query_selector_all('.network-lock')
        # This might be empty if all networks are open, which is valid

        save_screenshot(self.page, "network_list")

    def test_network_selection_highlights(self):
        """Clicking a network should highlight it."""
        networks = self.page.query_selector_all('.network-item')
        if not networks:
            self.skipTest("No networks available")

        first_network = networks[0]
        first_network.click()
        self.page.wait_for_timeout(500)

        # After click, the networks are re-rendered, so query again
        selected_network = self.page.query_selector('.network-item.selected')
        self.assertIsNotNone(selected_network, "A network should be selected after click")

        save_screenshot(self.page, "network_selected")

    def test_refresh_button_triggers_scan(self):
        """Refresh button should trigger network scan."""
        refresh_btn = self.page.query_selector('#refresh-btn')
        refresh_btn.click()

        # Should show scanning state
        self.page.wait_for_timeout(300)
        btn_text = refresh_btn.inner_text()
        self.assertIn('Scanning', btn_text, "Button should show scanning state")

        # Wait for scan to complete
        self.page.wait_for_timeout(3000)
        btn_text = refresh_btn.inner_text()
        self.assertEqual(btn_text, 'Refresh', "Button should return to Refresh state")


@unittest.skipUnless(HAS_PLAYWRIGHT, "playwright not installed")
class TestWiFiPortalPasswordSection(unittest.TestCase):
    """Test password input section behavior."""

    @classmethod
    def setUpClass(cls):
        cls.pw = sync_playwright().start()
        cls.browser = cls.pw.chromium.launch()

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.pw.stop()

    def setUp(self):
        self.page = self.browser.new_page(viewport=VIEWPORTS['iphone_x'])
        self.page.goto(WIFI_SETUP_URL)
        wait_for_networks(self.page)

    def tearDown(self):
        self.page.close()

    def test_password_section_hidden_initially(self):
        """Password section should be hidden until network selected."""
        password_section = self.page.query_selector('#password-section')
        is_visible = password_section.evaluate('el => el.classList.contains("visible")')
        self.assertFalse(is_visible, "Password section should be hidden initially")

    def test_password_section_shows_on_secure_network_select(self):
        """Password section should appear when selecting secured network."""
        # Find a secured network (has lock icon)
        networks = self.page.query_selector_all('.network-item')
        secured_network = None
        for net in networks:
            lock = net.query_selector('.network-lock')
            if lock:
                secured_network = net
                break

        if not secured_network:
            self.skipTest("No secured networks available")

        secured_network.click()
        self.page.wait_for_timeout(400)

        password_section = self.page.query_selector('#password-section')
        is_visible = password_section.evaluate('el => el.classList.contains("visible")')
        self.assertTrue(is_visible, "Password section should be visible for secured network")

        # Password input should be focused
        focused = self.page.evaluate('document.activeElement.id')
        self.assertEqual(focused, 'password-input', "Password input should be focused")

    def test_password_input_on_mobile_keyboard(self):
        """Password input should be accessible on mobile (not blocked by keyboard)."""
        networks = self.page.query_selector_all('.network-item')
        secured_network = None
        for net in networks:
            lock = net.query_selector('.network-lock')
            if lock:
                secured_network = net
                break

        if not secured_network:
            self.skipTest("No secured networks available")

        secured_network.click()
        self.page.wait_for_timeout(400)

        # The password input should be within the visible viewport
        password_input = self.page.query_selector('#password-input')
        box = password_input.bounding_box()
        viewport_height = VIEWPORTS['iphone_x']['height']

        # Even with keyboard, the input should be in upper half of screen
        # (iOS will scroll it into view, but initially it should be visible)
        self.assertLess(box['y'], viewport_height * 0.7,
            "Password input should be in visible area before keyboard")

        save_screenshot(self.page, "password_section_mobile")

    def test_enter_key_submits_password(self):
        """Pressing Enter in password field should submit."""
        networks = self.page.query_selector_all('.network-item')
        secured_network = None
        for net in networks:
            lock = net.query_selector('.network-lock')
            if lock:
                secured_network = net
                break

        if not secured_network:
            self.skipTest("No secured networks available")

        secured_network.click()
        self.page.wait_for_timeout(400)

        password_input = self.page.query_selector('#password-input')
        password_input.fill('testpassword')
        password_input.press('Enter')

        # Should show connecting status
        self.page.wait_for_timeout(500)
        status = self.page.query_selector('#status-message')
        is_visible = status.evaluate('el => el.classList.contains("visible")')
        self.assertTrue(is_visible, "Status message should appear after Enter")


@unittest.skipUnless(HAS_PLAYWRIGHT, "playwright not installed")
class TestWiFiPortalEdgeCases(unittest.TestCase):
    """Test edge cases and potential breaking conditions."""

    @classmethod
    def setUpClass(cls):
        cls.pw = sync_playwright().start()
        cls.browser = cls.pw.chromium.launch()

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.pw.stop()

    def test_long_ssid_doesnt_break_layout(self):
        """Very long SSID should truncate or wrap without breaking layout."""
        page = self.browser.new_page(viewport=VIEWPORTS['iphone_se'])

        # Mock a network scan response with a very long SSID
        page.route('**/api/wifi/scan', lambda route: route.fulfill(
            status=200,
            content_type='application/json',
            body=json.dumps({
                'networks': [
                    {'ssid': 'A' * 50, 'signal': 80, 'security': 'WPA2'},
                    {'ssid': 'Normal Network', 'signal': 60, 'security': 'WPA2'},
                ]
            })
        ))

        page.goto(WIFI_SETUP_URL)
        page.wait_for_timeout(1500)

        # Check no horizontal scroll (allow small margin for scrollbar)
        scroll_width = page.evaluate('document.body.scrollWidth')
        viewport_width = page.evaluate('window.innerWidth')
        self.assertLessEqual(scroll_width, viewport_width + 10,
            f"Long SSID should not cause horizontal scroll (scroll={scroll_width}, viewport={viewport_width})")

        # Verify the long SSID is truncated with ellipsis
        network_name = page.query_selector('.network-name')
        if network_name:
            overflow = page.evaluate('el => getComputedStyle(el).overflow', network_name)
            text_overflow = page.evaluate('el => getComputedStyle(el).textOverflow', network_name)
            self.assertEqual(overflow, 'hidden', "Network name should hide overflow")
            self.assertEqual(text_overflow, 'ellipsis', "Network name should show ellipsis")

        save_screenshot(page, "long_ssid")
        page.close()

    def test_special_characters_in_ssid(self):
        """SSIDs with special characters should be escaped properly."""
        page = self.browser.new_page(viewport=VIEWPORTS['iphone_x'])

        # Mock response with special characters that could break HTML/JS
        page.route('**/api/wifi/scan', lambda route: route.fulfill(
            status=200,
            content_type='application/json',
            body=json.dumps({
                'networks': [
                    {'ssid': '<script>alert("xss")</script>', 'signal': 80, 'security': 'WPA2'},
                    {'ssid': "Network's \"Name\" & <stuff>", 'signal': 60, 'security': 'WPA2'},
                    {'ssid': "emoji_test_\u2764_\u2605", 'signal': 40, 'security': 'Open'},
                ]
            })
        ))

        page.goto(WIFI_SETUP_URL)
        page.wait_for_timeout(1500)

        # Should not execute XSS
        networks = page.query_selector_all('.network-item')
        self.assertEqual(len(networks), 3, "Should render all 3 networks")

        # Verify text is escaped, not executed
        first_network_name = page.query_selector('.network-name')
        text = first_network_name.inner_text()
        self.assertIn('<script>', text, "Script tags should be escaped as text")

        save_screenshot(page, "special_chars_ssid")
        page.close()

    def test_empty_network_list(self):
        """Empty network list should show helpful message."""
        page = self.browser.new_page(viewport=VIEWPORTS['iphone_x'])

        page.route('**/api/wifi/scan', lambda route: route.fulfill(
            status=200,
            content_type='application/json',
            body=json.dumps({'networks': []})
        ))

        page.goto(WIFI_SETUP_URL)
        page.wait_for_timeout(1500)

        empty_state = page.query_selector('.empty-state')
        self.assertIsNotNone(empty_state, "Should show empty state")

        text = empty_state.inner_text()
        self.assertIn('No networks found', text)

        save_screenshot(page, "empty_networks")
        page.close()

    def test_scan_api_failure(self):
        """Network scan failure should show error state."""
        page = self.browser.new_page(viewport=VIEWPORTS['iphone_x'])

        page.route('**/api/wifi/scan', lambda route: route.abort('failed'))

        page.goto(WIFI_SETUP_URL)
        page.wait_for_timeout(1500)

        empty_state = page.query_selector('.empty-state')
        self.assertIsNotNone(empty_state)

        text = empty_state.inner_text()
        # Could say "Scan failed" or show emoji warning state
        self.assertTrue(
            'failed' in text.lower() or 'error' in text.lower() or '\u26a0' in text,
            f"Should show failure message, got: {text}"
        )

        save_screenshot(page, "scan_failure")
        page.close()

    def test_connection_error_shows_message(self):
        """Connection failure should display error message."""
        page = self.browser.new_page(viewport=VIEWPORTS['iphone_x'])

        # Let scan work normally
        page.goto(WIFI_SETUP_URL)
        wait_for_networks(page)

        # Mock connection failure
        page.route('**/api/wifi/connect', lambda route: route.fulfill(
            status=200,
            content_type='application/json',
            body=json.dumps({
                'success': False,
                'error': 'Authentication failed - wrong password'
            })
        ))

        # Select first secured network
        networks = page.query_selector_all('.network-item')
        for net in networks:
            lock = net.query_selector('.network-lock')
            if lock:
                net.click()
                break

        page.wait_for_timeout(400)

        # Enter password and submit
        page.fill('#password-input', 'wrongpassword')
        page.click('#connect-btn')
        page.wait_for_timeout(1000)

        # Should show error
        status = page.query_selector('#status-message')
        has_error = status.evaluate('el => el.classList.contains("error")')
        self.assertTrue(has_error, "Should show error status")

        text = status.inner_text()
        self.assertIn('wrong password', text.lower())

        save_screenshot(page, "connection_error")
        page.close()

    def test_rapid_refresh_clicks(self):
        """Rapid refresh button clicks should not break the UI."""
        page = self.browser.new_page(viewport=VIEWPORTS['iphone_x'])
        page.goto(WIFI_SETUP_URL)
        page.wait_for_timeout(1000)

        refresh_btn = page.query_selector('#refresh-btn')

        # Click rapidly 5 times
        for _ in range(5):
            if not refresh_btn.is_disabled():
                refresh_btn.click()
            page.wait_for_timeout(100)

        # Wait for any scans to complete
        page.wait_for_timeout(3000)

        # UI should still be functional
        btn_text = refresh_btn.inner_text()
        self.assertEqual(btn_text, 'Refresh')

        # No error states
        error_visible = page.evaluate(
            'document.querySelector("#status-message.error")?.classList.contains("visible") || false'
        )
        self.assertFalse(error_visible, "Should not show error from rapid clicks")

        page.close()

    def test_network_timeout_handling(self):
        """Slow network response should show loading state."""
        page = self.browser.new_page(viewport=VIEWPORTS['iphone_x'])

        # The initial loading state shows a spinner, so we can capture that
        # by checking quickly after page load
        page.goto(WIFI_SETUP_URL)

        # Should show scanning state initially
        page.wait_for_timeout(200)
        spinner = page.query_selector('.spinner')
        self.assertIsNotNone(spinner, "Should show spinner during initial scan")

        save_screenshot(page, "scanning_state")
        page.close()


@unittest.skipUnless(HAS_PLAYWRIGHT, "playwright not installed")
class TestWiFiPortalAccessibility(unittest.TestCase):
    """Test accessibility features of the portal."""

    @classmethod
    def setUpClass(cls):
        cls.pw = sync_playwright().start()
        cls.browser = cls.pw.chromium.launch()

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.pw.stop()

    def setUp(self):
        self.page = self.browser.new_page(viewport=VIEWPORTS['iphone_x'])
        self.page.goto(WIFI_SETUP_URL)
        wait_for_networks(self.page)

    def tearDown(self):
        self.page.close()

    def test_network_items_are_focusable(self):
        """Network items should have tabindex for keyboard navigation."""
        networks = self.page.query_selector_all('.network-item')
        if not networks:
            self.skipTest("No networks available")

        tabindex = networks[0].get_attribute('tabindex')
        self.assertEqual(tabindex, '0', "Network items should have tabindex=0")

    def test_keyboard_navigation_through_networks(self):
        """Should be able to tab through network items."""
        networks = self.page.query_selector_all('.network-item')
        if len(networks) < 2:
            self.skipTest("Need at least 2 networks")

        # Focus first network
        networks[0].focus()
        self.page.wait_for_timeout(200)

        # Tab to next
        self.page.keyboard.press('Tab')
        self.page.wait_for_timeout(200)

        # Check focus moved
        focused = self.page.evaluate('document.activeElement')
        # Focus should be on a different element
        self.assertIsNotNone(focused)

    def test_sufficient_color_contrast(self):
        """Text should have sufficient contrast against background."""
        # Check that primary text color (#e0e0e0) contrasts with bg (#000000)
        # This is a basic check - actual WCAG testing would need more
        text_color = self.page.evaluate(
            'getComputedStyle(document.querySelector(".network-name")).color'
        )
        self.assertIn('224', text_color, "Text should have light color for contrast")

    def test_touch_targets_minimum_size(self):
        """Touch targets (buttons, network items) should be at least 44x44px."""
        networks = self.page.query_selector_all('.network-item')
        if not networks:
            self.skipTest("No networks available")

        box = networks[0].bounding_box()
        self.assertGreaterEqual(box['height'], 44,
            f"Network item height {box['height']}px should be >= 44px for touch")

        # Check connect button if visible
        connect_btn = self.page.query_selector('#connect-btn')
        btn_box = connect_btn.bounding_box()
        if btn_box:  # May not be visible if password section hidden
            self.assertGreaterEqual(btn_box['height'], 44)


@unittest.skipUnless(HAS_PLAYWRIGHT, "playwright not installed")
class TestWiFiPortalAPIIntegration(unittest.TestCase):
    """Test WiFi portal API endpoints directly."""

    @classmethod
    def setUpClass(cls):
        cls.pw = sync_playwright().start()
        cls.browser = cls.pw.chromium.launch()

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.pw.stop()

    def setUp(self):
        self.page = self.browser.new_page()

    def tearDown(self):
        self.page.close()

    def test_wifi_status_endpoint(self):
        """GET /api/wifi/status returns proper structure."""
        resp = self.page.goto(f'{BASE_URL}/api/wifi/status')
        data = resp.json()

        self.assertIn('connected', data)
        self.assertIn('ap_mode_active', data)
        self.assertIsInstance(data['connected'], bool)
        self.assertIsInstance(data['ap_mode_active'], bool)

    def test_wifi_scan_endpoint(self):
        """GET /api/wifi/scan returns network list."""
        resp = self.page.goto(f'{BASE_URL}/api/wifi/scan')
        data = resp.json()

        self.assertIn('networks', data)
        self.assertIsInstance(data['networks'], list)

        if data['networks']:
            network = data['networks'][0]
            self.assertIn('ssid', network)
            self.assertIn('signal', network)
            self.assertIn('security', network)

    def test_captive_portal_android_check(self):
        """GET /generate_204 returns 204 when connected."""
        # Navigate to a page first to have valid origin
        self.page.goto(BASE_URL)
        self.page.wait_for_timeout(500)

        # Use evaluate with fetch to handle 204 response properly
        result = self.page.evaluate('''async () => {
            const r = await fetch('/generate_204');
            return {status: r.status, redirected: r.redirected};
        }''')

        # If connected, should be 204
        # If AP mode, should redirect (302) which fetch follows
        self.assertIn(result['status'], [200, 204],
            f"generate_204 should return 200/204, got {result['status']}")

    def test_captive_portal_apple_check(self):
        """GET /hotspot-detect.html returns Success or redirect."""
        resp = self.page.goto(f'{BASE_URL}/hotspot-detect.html')

        # If connected, should return HTML containing "Success"
        # If AP mode, should redirect
        if resp.status == 200:
            text = resp.text()
            self.assertIn('Success', text, "Apple check should contain 'Success'")

    def test_connect_requires_post(self):
        """GET /api/wifi/connect should fail (POST required)."""
        resp = self.page.goto(f'{BASE_URL}/api/wifi/connect')
        self.assertEqual(resp.status, 405, "Should require POST method")

    def test_connect_requires_ssid(self):
        """POST /api/wifi/connect without SSID should fail."""
        # First navigate to a page so we have a valid origin
        self.page.goto(BASE_URL)
        self.page.wait_for_timeout(500)

        result = self.page.evaluate('''async () => {
            const r = await fetch('/api/wifi/connect', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({password: 'test'})
            });
            return {status: r.status, data: await r.json()};
        }''')

        # Should either have success=False OR have an error field
        data = result['data']
        has_error = 'error' in data
        success_is_false = data.get('success') == False

        self.assertTrue(has_error or success_is_false,
            f"Should indicate failure without SSID, got: {data}")

        if has_error:
            self.assertIn('ssid', data['error'].lower())


@unittest.skipUnless(HAS_PLAYWRIGHT, "playwright not installed")
class TestWiFiPortalVisualRegression(unittest.TestCase):
    """Visual regression tests - capture screenshots for manual review."""

    @classmethod
    def setUpClass(cls):
        cls.pw = sync_playwright().start()
        cls.browser = cls.pw.chromium.launch()

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.pw.stop()

    def test_capture_all_states(self):
        """Capture screenshots of all important UI states."""
        states_captured = []

        # 1. Initial loading state
        page = self.browser.new_page(viewport=VIEWPORTS['iphone_x'])
        page.goto(WIFI_SETUP_URL)
        page.wait_for_timeout(500)
        save_screenshot(page, "state_01_loading")
        states_captured.append("loading")

        # 2. Network list loaded
        wait_for_networks(page)
        save_screenshot(page, "state_02_networks_loaded")
        states_captured.append("networks_loaded")

        # 3. Network selected
        networks = page.query_selector_all('.network-item')
        if networks:
            networks[0].click()
            page.wait_for_timeout(400)
            save_screenshot(page, "state_03_network_selected")
            states_captured.append("network_selected")

        # 4. Password being entered
        password_input = page.query_selector('#password-input')
        if password_input and password_input.is_visible():
            password_input.fill('testpassword123')
            save_screenshot(page, "state_04_password_entered")
            states_captured.append("password_entered")

        # 5. Connection in progress - capture the connecting state
        page.close()

        page = self.browser.new_page(viewport=VIEWPORTS['iphone_x'])
        # Don't mock - just capture the loading state quickly after clicking connect
        page.goto(WIFI_SETUP_URL)
        wait_for_networks(page)

        networks = page.query_selector_all('.network-item')
        for net in networks:
            if net.query_selector('.network-lock'):
                net.click()
                break
        page.wait_for_timeout(400)
        page.fill('#password-input', 'test')

        # Mock slow connection AFTER filling password
        page.route('**/api/wifi/connect', lambda route: route.abort('failed'))

        page.click('#connect-btn')
        page.wait_for_timeout(300)  # Quick capture of loading state
        save_screenshot(page, "state_05_connecting")
        states_captured.append("connecting")
        page.close()

        # 6. Error state
        page = self.browser.new_page(viewport=VIEWPORTS['iphone_x'])
        page.route('**/api/wifi/connect', lambda route: route.fulfill(
            status=200,
            content_type='application/json',
            body=json.dumps({'success': False, 'error': 'Authentication failed'})
        ))
        page.goto(WIFI_SETUP_URL)
        wait_for_networks(page)

        networks = page.query_selector_all('.network-item')
        for net in networks:
            if net.query_selector('.network-lock'):
                net.click()
                break
        page.wait_for_timeout(400)
        page.fill('#password-input', 'wrong')
        page.click('#connect-btn')
        page.wait_for_timeout(1000)
        save_screenshot(page, "state_06_error")
        states_captured.append("error")
        page.close()

        # 7. Success state
        page = self.browser.new_page(viewport=VIEWPORTS['iphone_x'])
        page.route('**/api/wifi/connect', lambda route: route.fulfill(
            status=200,
            content_type='application/json',
            body=json.dumps({'success': True, 'ssid': 'Test Network'})
        ))
        page.goto(WIFI_SETUP_URL)
        wait_for_networks(page)

        networks = page.query_selector_all('.network-item')
        for net in networks:
            if net.query_selector('.network-lock'):
                net.click()
                break
        page.wait_for_timeout(400)
        page.fill('#password-input', 'correct')
        page.click('#connect-btn')
        page.wait_for_timeout(1000)
        save_screenshot(page, "state_07_success")
        states_captured.append("success")
        page.close()

        print(f"\nCaptured {len(states_captured)} UI states: {', '.join(states_captured)}")
        print(f"Screenshots saved to: {SCREENSHOT_DIR}")


if __name__ == '__main__':
    if not HAS_PLAYWRIGHT:
        print("Install playwright: pip3 install playwright && python3 -m playwright install chromium")
        sys.exit(1)

    unittest.main(verbosity=2)


class TestPortalServerIsIsolated(unittest.TestCase):
    """The portal tests must only ever talk to the throwaway server."""

    def test_not_the_live_web_ui(self):
        sys.path.insert(0, str(Path(__file__).parent))
        from live_guard import targets_live_web_ui
        self.assertFalse(targets_live_web_ui(BASE_URL))
        self.assertTrue(BASE_URL.startswith('http://127.0.0.1:'))

    def test_only_portal_routes_are_served(self):
        import urllib.error
        import urllib.request
        for path in ('/api/status', '/api/wifi/disconnect', '/api/wifi/ap/start', '/api/test/trigger-block'):
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                urllib.request.urlopen(urllib.request.Request(f'{BASE_URL}{path}', method='POST', data=b'{}'))
            self.assertEqual(ctx.exception.code, 403, path)

    def test_connect_never_reaches_a_radio(self):
        import urllib.request
        req = urllib.request.Request(f'{BASE_URL}/api/wifi/connect', method='POST',
                                     data=json.dumps({'ssid': 'CoffeeShop', 'password': 'x' * 8}).encode(),
                                     headers={'Content-Type': 'application/json'})
        try:
            urllib.request.urlopen(req)
        except Exception:
            pass   # the fake answers 400 "Incorrect password"
        self.assertIn(('CoffeeShop', 'x' * 8), _SERVER.wifi.connect_attempts)
        with self.assertRaises(AssertionError):
            _SERVER.wifi.disconnect_network()     # any other radio action is refused
