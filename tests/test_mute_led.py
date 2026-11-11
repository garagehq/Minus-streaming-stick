"""MuteLED against a fake sysfs GPIO tree: flashes while muted, minimum blinks, off on stop."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
import mute_led


def _fake_sysfs(tmp_path, monkeypatch):
    chip = tmp_path / 'gpiochip96'
    chip.mkdir()
    (chip / 'label').write_text('gpio3\n')
    (chip / 'base').write_text('96\n')
    g = tmp_path / 'gpio107'
    g.mkdir()
    (g / 'direction').write_text('in')
    (g / 'value').write_text('0')
    monkeypatch.setattr(mute_led, 'SYSFS', tmp_path)
    return g / 'value'


def _watch(path, seconds):
    seen, end = [], time.time() + seconds
    while time.time() < end:
        v = path.read_text().strip()
        if not seen or seen[-1] != v:
            seen.append(v)
        time.sleep(0.005)
    return seen


def test_short_mute_still_flashes(tmp_path, monkeypatch):
    value = _fake_sysfs(tmp_path, monkeypatch)
    led = mute_led.MuteLED()
    assert led.available
    led.start()
    led.set_muted(True)
    led.set_muted(False)
    seen = _watch(value, 0.8)
    led.stop()
    assert seen.count('1') >= mute_led.MIN_FLASHES
    assert value.read_text() == '0'


def test_disabled_by_env(monkeypatch, tmp_path):
    _fake_sysfs(tmp_path, monkeypatch)
    monkeypatch.setenv('MINUS_MUTE_LED', '0')
    led = mute_led.MuteLED()
    assert not led.available
    led.set_muted(True)  # no-op, no crash


def test_missing_chip_is_harmless(monkeypatch, tmp_path):
    monkeypatch.setattr(mute_led, 'SYSFS', tmp_path)
    led = mute_led.MuteLED()
    assert not led.available
    led.start(); led.set_muted(True); led.stop()
