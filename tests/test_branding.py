"""User-visible branding: KingBlocker9000, never "Minus", never a name-mute hit.

The box shows its name on the TV (READY banner, hotspot instructions), in
the web UI and on the Wi-Fi setup page. The name muter targets the word
"king", so the brand must stay one token ("KingBlocker9000"): "King
Blocker" on screen would mute the TV whenever it is shown or read out.
"""
import os
import re
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(ROOT, 'src'))

from name_mute import NameMatcher  # noqa: E402
import wifi_manager  # noqa: E402

BRAND = 'KingBlocker9000'


def _read(rel):
    with open(os.path.join(ROOT, rel), encoding='utf-8') as f:
        return f.read()


def test_hotspot_ssid():
    assert wifi_manager.AP_SSID == BRAND


def test_brand_is_not_a_name_mute_hit():
    m = NameMatcher()
    for text in (BRAND, BRAND.upper(), f'[ {BRAND.upper()} READY ]',
                 f'Connect to WiFi: {BRAND}'):
        assert m.find(text) == [], text


def test_pages_say_kingblocker_not_minus():
    for rel in ('src/templates/index.html', 'src/templates/wifi_setup.html'):
        html = _read(rel)
        # Visible text only: drop scripts' identifiers like minusInstance,
        # API paths and CSS classes by looking at text between tags and at
        # string literals shown to the user (alert/confirm/Notification).
        text = re.sub(r'<script.*?</script>', ' ', html, flags=re.S)
        text = re.sub(r'<[^>]+>', ' ', text)
        assert not re.search(r'\bMinus\b', text), rel
        for call in re.findall(r'(?:alert|confirm|Notification)\(\s*([`\'"].*?[`\'"])',
                               html, flags=re.S):
            assert 'Minus' not in call, (rel, call)
        assert BRAND in html or BRAND.upper() in html, rel


def test_hotspot_password_has_no_minus():
    src = _read('src/wifi_manager.py')
    assert 'minus123' not in src
