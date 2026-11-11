"""
Webhook notifications for Minus.

Sends HTTP notifications to configured endpoints when events occur.
"""

import logging
import threading
import time
import json
from typing import Optional, List
import urllib.request
import urllib.error

logger = logging.getLogger(__name__)


class WebhookManager:
    """
    Manages webhook notifications for blocking state changes.

    Events:
    - blocking_started: When ad blocking begins
    - blocking_stopped: When ad blocking ends
    - ad_detected: When an ad is detected (may not trigger blocking)
    """

    def __init__(self, urls: Optional[List[str]] = None, enabled: bool = True):
        """
        Initialize webhook manager.

        Args:
            urls: List of webhook URLs to notify
            enabled: Whether webhooks are enabled
        """
        self.urls = urls or []
        self.enabled = enabled
        self._lock = threading.Lock()
        self._last_notification_time = 0
        self._min_notification_interval = 1.0  # Rate limit to 1 per second

    def add_url(self, url: str):
        """Add a webhook URL."""
        with self._lock:
            if url not in self.urls:
                self.urls.append(url)
                logger.info(f"[Webhook] Added URL: {url}")

    def remove_url(self, url: str):
        """Remove a webhook URL."""
        with self._lock:
            if url in self.urls:
                self.urls.remove(url)
                logger.info(f"[Webhook] Removed URL: {url}")

    def get_urls(self) -> List[str]:
        """Get list of webhook URLs."""
        with self._lock:
            return list(self.urls)

    def set_enabled(self, enabled: bool):
        """Enable or disable webhooks."""
        self.enabled = enabled
        logger.info(f"[Webhook] {'Enabled' if enabled else 'Disabled'}")

    def notify(self, event: str, data: dict = None):
        """
        Send notification to all webhook URLs.

        Args:
            event: Event type (blocking_started, blocking_stopped, ad_detected)
            data: Additional event data
        """
        if not self.enabled:
            return

        with self._lock:
            if not self.urls:
                return

            # Rate limiting
            now = time.time()
            if now - self._last_notification_time < self._min_notification_interval:
                return
            self._last_notification_time = now

            urls = list(self.urls)

        # Send notifications in background thread to avoid blocking
        threading.Thread(
            target=self._send_notifications,
            args=(event, data or {}, urls),
            daemon=True
        ).start()

    def _send_notifications(self, event: str, data: dict, urls: List[str]):
        """Send notifications to all URLs (runs in background thread)."""
        payload = {
            'event': event,
            'timestamp': time.time(),
            'data': data
        }

        json_data = json.dumps(payload).encode('utf-8')

        for url in urls:
            try:
                req = urllib.request.Request(
                    url,
                    data=json_data,
                    headers={
                        'Content-Type': 'application/json',
                        'User-Agent': 'Kingblocker9000/1.0'
                    },
                    method='POST'
                )
                with urllib.request.urlopen(req, timeout=5.0) as response:
                    if response.status < 300:
                        logger.debug(f"[Webhook] Sent {event} to {url}")
                    else:
                        logger.warning(f"[Webhook] Failed to send to {url}: HTTP {response.status}")
            except urllib.error.URLError as e:
                logger.debug(f"[Webhook] Network error sending to {url}: {e}")
            except Exception as e:
                logger.debug(f"[Webhook] Error sending to {url}: {e}")


# Global instance for easy access
_webhook_manager: Optional[WebhookManager] = None


def get_webhook_manager() -> WebhookManager:
    """Get the global webhook manager instance."""
    global _webhook_manager
    if _webhook_manager is None:
        _webhook_manager = WebhookManager()
    return _webhook_manager


def notify_blocking_started(source: str, **kwargs):
    """Notify that blocking has started."""
    get_webhook_manager().notify('blocking_started', {
        'source': source,
        **kwargs
    })


def notify_blocking_stopped(source: str, duration_seconds: float = 0, **kwargs):
    """Notify that blocking has stopped."""
    get_webhook_manager().notify('blocking_stopped', {
        'source': source,
        'duration_seconds': duration_seconds,
        **kwargs
    })


def notify_ad_detected(source: str, texts: List[str] = None, **kwargs):
    """Notify that an ad was detected."""
    get_webhook_manager().notify('ad_detected', {
        'source': source,
        'texts': texts or [],
        **kwargs
    })
