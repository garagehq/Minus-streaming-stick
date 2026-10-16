"""Opt-in gate for tests that drive the LIVE Minus service on this machine.

Some UI tests are deliberately live: they talk to the running web UI on
http://localhost:80 and make it act for real (fire the IR blaster, drive the
status LEDs, move screenshot files). Run by accident on a box someone is
watching, a test like that changes the real device. On 2026-10-15 a full
`pytest tests` on a Minus box drove the live Wi-Fi API (via the old
test_wifi_portal.py) and took the box off the network until someone plugged in
ethernet. Live tests therefore only run when asked for explicitly:

    MINUS_LIVE_UI_TESTS=1 python3 -m pytest tests/test_ir_ui.py

`require_live_opt_in()` at module level skips the whole file otherwise (under
pytest), or exits cleanly when the file is run as a script. tests/conftest.py
applies the same rule to any test module whose BASE_URL points at the local
web UI, so a new test copying the pattern is covered too.
"""
import os
import sys
from urllib.parse import urlparse

ENV = 'MINUS_LIVE_UI_TESTS'
LOCAL_HOSTS = {'localhost', '127.0.0.1', '0.0.0.0', '::1', ''}


def live_ui_tests_enabled() -> bool:
    return os.environ.get(ENV, '').strip() == '1'


def targets_live_web_ui(base_url) -> bool:
    """True if a test's BASE_URL points at the web UI a Minus box serves (port 80 here)."""
    if not base_url:
        return False
    u = urlparse(str(base_url))
    port = u.port or (443 if u.scheme == 'https' else 80)
    return (u.hostname or '') in LOCAL_HOSTS and port == 80


def skip_reason(what: str = 'this test module') -> str:
    return (f'{what} drives the LIVE Minus service on this machine; '
            f'set {ENV}=1 to run it on purpose')


def require_live_opt_in(what: str = 'this test module'):
    if live_ui_tests_enabled():
        return
    reason = skip_reason(what)
    if 'pytest' in sys.modules:
        import pytest
        pytest.skip(reason, allow_module_level=True)
    print(f'SKIPPED: {reason}')
    sys.exit(0)
