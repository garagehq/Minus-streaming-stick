"""Session-wide safety net: never let a test reach the live Minus service by accident.

Any test module whose BASE_URL points at the local web UI (http://localhost:80,
what a Minus box serves) is skipped unless MINUS_LIVE_UI_TESTS=1 is set — see
tests/live_guard.py for why. Tests that need the portal UI without a real
device use tests/portal_test_server.py instead.
"""
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from live_guard import live_ui_tests_enabled, skip_reason, targets_live_web_ui  # noqa: E402


# The process-wide clocks as they were before any test module was imported.
_REAL_CLOCKS = {name: getattr(time, name) for name in ('time', 'monotonic', 'sleep')}


def _replaced_clocks():
    return [name for name, fn in _REAL_CLOCKS.items() if getattr(time, name) is not fn]


def pytest_collection_finish(session):
    """Importing a test module must not rewrite the global `time` module.

    tests/test_vlm_decision_sim.py once did `H.time.time = ...` at import (H.time
    is the real time module), which froze time.time() for the whole run: 23
    unrelated tests failed, only on a Minus box where that module loads. Patch
    a module's own reference (or use unittest.mock) instead.
    """
    bad = _replaced_clocks()
    if bad:
        raise pytest.UsageError(
            f"a test module replaced time.{'/'.join(bad)} at import time; "
            f"patch the module's own reference instead (see tests/conftest.py)")


def pytest_collection_modifyitems(config, items):
    if live_ui_tests_enabled():
        return
    for item in items:
        mod = getattr(item, 'module', None)
        if mod is not None and targets_live_web_ui(getattr(mod, 'BASE_URL', None)):
            item.add_marker(pytest.mark.skip(reason=skip_reason(Path(mod.__file__).name)))
