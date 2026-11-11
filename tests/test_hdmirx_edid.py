"""The HDMI-RX EDID caps the input at 4K30 / 1080p60.

Sources pick the highest refresh the EDID offers. The old 24-30 Hz-only
EDID left a source that needs 60 Hz sending nothing (TV: No Signal); 4K60
runs the SoC at 80-85 C. So: 4K 24/25/30, 1080p 24-60, never 4K60.
"""
import os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
EDID = os.path.join(ROOT, 'hdmirx', 'hdmirx-edid-4k30-1080p60.bin')

VIC_4K60 = {96, 97, 106, 107, 117, 118}          # 3840/4096 x2160 @50/60
VIC_1080P60 = 16


def _edid():
    with open(EDID, 'rb') as f:
        return f.read()


def _vics(cta):
    out, i = [], 4
    while i < cta[2]:
        tag, ln = cta[i] >> 5, cta[i] & 31
        if tag == 2:
            out += [b & 0x7f if 1 <= (b & 0x7f) <= 64 else b for b in cta[i + 1:i + 1 + ln]]
        i += 1 + ln
    return out


def _dtd_pixclk_khz(block):
    clks = []
    for off in (54, 72, 90, 108):
        clk = int.from_bytes(block[off:off + 2], 'little')
        if clk:
            clks.append(clk * 10)
    return clks


def test_checksums():
    e = _edid()
    assert len(e) == 256
    assert sum(e[:128]) % 256 == 0
    assert sum(e[128:]) % 256 == 0


def test_offers_1080p60_and_4k30_not_4k60():
    e = _edid()
    vics = _vics(e[128:])
    assert VIC_1080P60 in vics
    assert 95 in vics                   # 3840x2160@30
    assert not VIC_4K60 & set(vics)


def test_offers_50hz_and_sd_modes():
    vics = set(_vics(_edid()[128:]))
    # 720p50, 576p 4:3/16:9, 480p 4:3/16:9, 1080p50, 640x480
    assert {19, 17, 18, 2, 3, 31, 1} <= vics


def test_no_detailed_timing_above_4k30():
    e = _edid()
    clks = _dtd_pixclk_khz(e[:128])
    assert max(clks) <= 297000          # 4K30 = 297 MHz; 4K60 needs 594


def test_range_limits_stop_at_300mhz():
    base = _edid()[:128]
    for off in (54, 72, 90, 108):
        d = base[off:off + 18]
        if d[:4] == b'\0\0\0\xfd':
            assert d[6] <= 61               # max vertical Hz
            assert d[9] * 10 <= 300         # max pixel clock MHz
            assert d[10] == 0x01            # range limits only, no GTF modes
            return
    raise AssertionError('no range-limits descriptor')


def test_service_and_installer_use_it():
    with open(os.path.join(ROOT, 'minus-hdmirx-edid.service')) as f:
        assert 'hdmirx-edid-4k30-1080p60.bin' in f.read()
    with open(os.path.join(ROOT, 'install.sh')) as f:
        s = f.read()
    assert 'hdmirx-edid-4k30-1080p60.bin' in s
    assert 'MINUS_STREAM_FPS=0' in s
