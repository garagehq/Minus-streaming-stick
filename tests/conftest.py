"""Session-wide safety net: never let a test reach the live Minus service by accident.

Any test module whose BASE_URL points at the local web UI (http://localhost:80,
what a Minus box serves) is skipped unless MINUS_LIVE_UI_TESTS=1 is set — see
tests/live_guard.py for why. Tests that need the portal UI without a real
device use tests/portal_test_server.py instead.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from live_guard import live_ui_tests_enabled, skip_reason, targets_live_web_ui  # noqa: E402


def pytest_collection_modifyitems(config, items):
    if live_ui_tests_enabled():
        return
    for item in items:
        mod = getattr(item, 'module', None)
        if mod is not None and targets_live_web_ui(getattr(mod, 'BASE_URL', None)):
            item.add_marker(pytest.mark.skip(reason=skip_reason(Path(mod.__file__).name)))
