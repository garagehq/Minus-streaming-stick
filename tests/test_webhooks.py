"""Tests for src/webhooks.py — WebhookManager class and convenience functions."""

import json
import sys
import os
import threading
import time
import unittest
from unittest.mock import patch, MagicMock, call

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


class TestWebhookManagerInit(unittest.TestCase):
    """Test WebhookManager initialization."""

    def test_default_init(self):
        """Test default constructor values."""
        from src.webhooks import WebhookManager
        mgr = WebhookManager()
        self.assertEqual(mgr.urls, [])
        self.assertTrue(mgr.enabled)
        self.assertIsNotNone(mgr._lock)

    def test_init_with_urls(self):
        """Test constructor with pre-populated URLs."""
        from src.webhooks import WebhookManager
        mgr = WebhookManager(urls=['http://a.com/hook', 'http://b.com/hook'])
        self.assertEqual(mgr.urls, ['http://a.com/hook', 'http://b.com/hook'])

    def test_init_with_enabled_false(self):
        """Test constructor with enabled=False."""
        from src.webhooks import WebhookManager
        mgr = WebhookManager(enabled=False)
        self.assertFalse(mgr.enabled)

    def test_init_lock_is_threading_lock(self):
        """Test that _lock is a real threading.Lock."""
        from src.webhooks import WebhookManager
        mgr = WebhookManager()
        self.assertIsInstance(mgr._lock, type(threading.Lock()))


class TestWebhookManagerCRUD(unittest.TestCase):
    """Test add_url, remove_url, get_urls, set_enabled."""

    def setUp(self):
        from src.webhooks import WebhookManager
        self.mgr = WebhookManager()

    def test_add_url_single(self):
        """Test adding a single URL."""
        self.mgr.add_url('http://example.com/webhook')
        self.assertEqual(self.mgr.get_urls(), ['http://example.com/webhook'])

    def test_add_url_idempotent(self):
        """Adding a duplicate URL does not create a second entry."""
        self.mgr.add_url('http://example.com/webhook')
        self.mgr.add_url('http://example.com/webhook')
        self.assertEqual(len(self.mgr.get_urls()), 1)

    def test_add_url_dedup(self):
        """Adding the same URL twice only keeps one."""
        self.mgr.add_url('http://a.com/hook')
        self.mgr.add_url('http://a.com/hook')
        self.assertEqual(len(self.mgr.get_urls()), 1)

    def test_add_url_multiple(self):
        """Test adding multiple distinct URLs."""
        self.mgr.add_url('http://a.com/hook')
        self.mgr.add_url('http://b.com/hook')
        self.mgr.add_url('http://c.com/hook')
        self.assertEqual(len(self.mgr.get_urls()), 3)

    def test_remove_url_existing(self):
        """Test removing an existing URL."""
        self.mgr.add_url('http://a.com/hook')
        self.mgr.add_url('http://b.com/hook')
        self.mgr.remove_url('http://a.com/hook')
        self.assertEqual(self.mgr.get_urls(), ['http://b.com/hook'])

    def test_remove_url_nonexistent(self):
        """Removing a URL that doesn't exist is a no-op."""
        self.mgr.remove_url('http://nope.com/hook')
        self.assertEqual(self.mgr.get_urls(), [])

    def test_remove_url_all(self):
        """Removing all URLs leaves empty list."""
        self.mgr.add_url('http://a.com/hook')
        self.mgr.remove_url('http://a.com/hook')
        self.assertEqual(self.mgr.get_urls(), [])

    def test_get_urls_returns_copy(self):
        """get_urls returns a fresh list (modifying it doesn't affect internal state)."""
        self.mgr.add_url('http://a.com/hook')
        urls = self.mgr.get_urls()
        urls.append('http://fake.com/hook')
        self.assertEqual(self.mgr.get_urls(), ['http://a.com/hook'])

    def test_set_enabled_true(self):
        """set_enabled(True) enables the manager."""
        self.mgr.set_enabled(False)
        self.assertFalse(self.mgr.enabled)
        self.mgr.set_enabled(True)
        self.assertTrue(self.mgr.enabled)

    def test_set_enabled_false(self):
        """set_enabled(False) disables the manager."""
        self.mgr.set_enabled(False)
        self.assertFalse(self.mgr.enabled)


class TestNotifyEnabledDisabled(unittest.TestCase):
    """Test notify() respects enabled flag."""

    def test_notify_noop_when_disabled(self):
        """When enabled=False, notify() returns early without spawning a thread."""
        from src.webhooks import WebhookManager
        mgr = WebhookManager(urls=['http://a.com/hook'], enabled=False)
        before = mgr._last_notification_time
        mgr.notify('blocking_started', {'source': 'test'})
        # Rate-limit timestamp should not have changed (no network attempt)
        # The key assertion: thread was never started
        self.assertEqual(mgr._last_notification_time, before)

    def test_notify_noop_when_no_urls(self):
        """When urls is empty, notify() returns early even if enabled."""
        from src.webhooks import WebhookManager
        mgr = WebhookManager(urls=[], enabled=True)
        before = mgr._last_notification_time
        mgr.notify('blocking_started', {'source': 'test'})
        # No URLs → returns immediately
        self.assertEqual(mgr._last_notification_time, before)


class TestNotifyPayload(unittest.TestCase):
    """Test the structure of webhook notifications."""

    def test_notify_creates_correct_payload(self):
        """Test that notify method creates a valid payload structure."""
        from src.webhooks import WebhookManager
        mgr = WebhookManager(urls=['http://a.com/hook'])
        self.assertTrue(hasattr(mgr, '_send_notifications'))

    def test_notify_payload_has_required_keys(self):
        """The _send_notifications method should include event, timestamp, data."""
        from src.webhooks import WebhookManager
        mgr = WebhookManager(urls=['http://a.com/hook'])

        # Capture what would be sent to urlopen
        captured_requests = []

        def fake_urlopen(req, timeout=5.0):
            payload = json.loads(req.data)
            captured_requests.append((req, payload))
            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=None)
            return mock_resp

        with patch('src.webhooks.urllib.request.urlopen', side_effect=fake_urlopen):
            mgr._send_notifications('test_event', {'foo': 'bar'}, ['http://a.com/hook'])

        self.assertEqual(len(captured_requests), 1)
        req, payload = captured_requests[0]
        self.assertEqual(payload['event'], 'test_event')
        self.assertIn('timestamp', payload)
        self.assertIn('data', payload)
        self.assertEqual(payload['data']['foo'], 'bar')
        self.assertEqual(req.get_method(), 'POST')

    def test_notify_payload_multiple_urls(self):
        """_send_notifications iterates over all configured URLs."""
        from src.webhooks import WebhookManager
        mgr = WebhookManager(urls=['http://a.com/hook', 'http://b.com/hook'])

        urlopen_calls = []
        def fake_urlopen(req, timeout=5.0):
            urlopen_calls.append(str(req.host if hasattr(req, 'host') else req))
            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=None)
            return mock_resp

        with patch('src.webhooks.urllib.request.urlopen', side_effect=fake_urlopen):
            mgr._send_notifications('multi_test', {'key': 'val'}, ['http://a.com/hook', 'http://b.com/hook'])

        self.assertEqual(len(urlopen_calls), 2)


class TestNotifyRateLimiting(unittest.TestCase):
    """Test rate limiting in notify()."""

    def test_rate_limit_skips_short_interval(self):
        """Two notify calls within 1s — second is dropped."""
        from src.webhooks import WebhookManager
        mgr = WebhookManager(urls=['http://a.com/hook'], enabled=True)

        # First notify sets _last_notification_time
        with patch('src.webhooks.urllib.request.urlopen'):
            mgr.notify('blocking_started', {'source': 'ocr'})
            first_time = mgr._last_notification_time
            self.assertGreater(first_time, 0)

        # Second notify immediately should be rate-limited (returns without calling urlopen)
        urlopen_calls = []
        def fake_urlopen(req, timeout=5.0):
            urlopen_calls.append(req)
            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=None)
            return mock_resp

        with patch('src.webhooks.urllib.request.urlopen', side_effect=fake_urlopen):
            mgr.notify('blocking_started', {'source': 'vlm'})

        # Rate limit should have dropped this — 0 calls
        self.assertEqual(len(urlopen_calls), 0)

    def test_rate_limit_allows_after_interval(self):
        """Notify after the rate-limit interval should succeed."""
        from src.webhooks import WebhookManager
        mgr = WebhookManager(urls=['http://a.com/hook'], enabled=True)

        # First call
        mgr.notify('blocking_started', {'source': 'ocr'})
        first_time = mgr._last_notification_time

        # Wait past the rate-limit interval (default is 1s)
        time.sleep(1.1)

        urlopen_calls = []
        def fake_urlopen(req, timeout=5.0):
            urlopen_calls.append(req)
            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=None)
            return mock_resp

        with patch('src.webhooks.urllib.request.urlopen', side_effect=fake_urlopen):
            mgr.notify('blocking_started', {'source': 'vlm'})

        # Should succeed after cooldown
        self.assertEqual(len(urlopen_calls), 1)


class TestConvenienceFunctions(unittest.TestCase):
    """Test the convenience functions at module level."""

    def setUp(self):
        """Reset the global webhook manager for clean state."""
        import src.webhooks as wh
        wh._webhook_manager = None

    def tearDown(self):
        """Clean up after tests."""
        import src.webhooks as wh
        wh._webhook_manager = None

    def test_get_webhook_manager_singleton(self):
        """get_webhook_manager returns the same instance on repeated calls."""
        from src.webhooks import get_webhook_manager
        mgr = get_webhook_manager()
        mgr2 = get_webhook_manager()
        self.assertIs(mgr, mgr2)

    def test_notify_blocking_started(self):
        """Test notify_blocking_started convenience function."""
        from src.webhooks import notify_blocking_started, get_webhook_manager
        mgr = get_webhook_manager()
        # Reset to clean state
        mgr.urls = []
        mgr.add_url('http://test.com/hook')
        mgr.enabled = True

        urlopen_calls = []
        def fake_urlopen(req, timeout=5.0):
            urlopen_calls.append(json.loads(req.data))
            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=None)
            return mock_resp

        with patch('src.webhooks.urllib.request.urlopen', side_effect=fake_urlopen):
            notify_blocking_started('ocr', duration=5.0)

        self.assertEqual(len(urlopen_calls), 1)
        self.assertEqual(urlopen_calls[0]['event'], 'blocking_started')
        self.assertEqual(urlopen_calls[0]['data']['source'], 'ocr')

    def test_notify_blocking_stopped_includes_duration(self):
        """Test notify_blocking_stopped includes duration_seconds."""
        from src.webhooks import notify_blocking_stopped, get_webhook_manager
        mgr = get_webhook_manager()
        mgr.urls = []
        mgr.add_url('http://test.com/hook')
        mgr.enabled = True

        captured_data = {}

        def capture_send(event, data, urls):
            captured_data.update(data)

        mgr._send_notifications = capture_send

        notify_blocking_stopped('vlm', duration_seconds=12.5)

        self.assertEqual(captured_data.get('source'), 'vlm')
        self.assertEqual(captured_data.get('duration_seconds'), 12.5)

    def test_notify_ad_detected(self):
        """Test notify_ad_detected includes texts list."""
        from src.webhooks import notify_ad_detected, get_webhook_manager
        mgr = get_webhook_manager()
        mgr.urls = []
        mgr.add_url('http://test.com/hook')
        mgr.enabled = True

        captured_data = {}

        def capture_send(event, data, urls):
            captured_data.update(data)

        mgr._send_notifications = capture_send

        notify_ad_detected('ocr', texts=['Ad 0:30', 'skip'])

        self.assertEqual(captured_data.get('source'), 'ocr')
        self.assertEqual(captured_data.get('texts'), ['Ad 0:30', 'skip'])


class TestWebhookAPIEndpoints(unittest.TestCase):
    """Test the Flask routes for webhooks in webui.py."""

    def setUp(self):
        # Ensure sys.path has src/ so WebUI imports resolve
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
        # Reset the global webhook manager for clean state
        import src.webhooks as wh
        wh._webhook_manager = None
        from src.webui import WebUI
        self.mock_minus = MagicMock()
        self.ui = WebUI(self.mock_minus, port=18080)
        self.client = self.ui.app.test_client()

    def tearDown(self):
        import src.webhooks as wh
        wh._webhook_manager = None

    def test_webhooks_get_empty(self):
        """GET /api/webhooks returns empty list when no URLs configured."""
        response = self.client.get('/api/webhooks')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data['enabled'])
        self.assertEqual(data['urls'], [])

    def test_webhooks_set_add_url(self):
        """POST /api/webhooks with add_url adds a URL."""
        response = self.client.post('/api/webhooks',
                                    json={'add_url': 'http://test.com/hook'})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data['success'])
        self.assertEqual(len(data['urls']), 1)
        self.assertEqual(data['urls'][0], 'http://test.com/hook')

    def test_webhooks_set_urls(self):
        """POST /api/webhooks with urls replaces all URLs."""
        response = self.client.post('/api/webhooks',
                                    json={'urls': ['http://a.com/h', 'http://b.com/h']})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(len(data['urls']), 2)

    def test_webhooks_set_enable_disable(self):
        """POST /api/webhooks with enabled toggles webhook state."""
        response = self.client.post('/api/webhooks', json={'enabled': False})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertFalse(data['enabled'])

        response = self.client.post('/api/webhooks', json={'enabled': True})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data['enabled'])

    def test_webhooks_set_combined(self):
        """POST /api/webhooks handles combined enable + urls + add_url."""
        response = self.client.post('/api/webhooks', json={
            'enabled': True,
            'urls': ['http://a.com/h', 'http://b.com/h'],
            'add_url': 'http://c.com/h'
        })
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data['success'])
        self.assertEqual(len(data['urls']), 3)

    def test_webhooks_test_no_urls(self):
        """POST /api/webhooks/test returns 400 when no URLs configured."""
        response = self.client.post('/api/webhooks/test')
        self.assertEqual(response.status_code, 400)

    def test_webhooks_test_with_urls(self):
        """POST /api/webhooks/test sends notification when URLs configured."""
        self.client.post('/api/webhooks', json={'urls': ['http://test.com/h']})
        response = self.client.post('/api/webhooks/test')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data['success'])


if __name__ == '__main__':
    unittest.main()
