"""Tests for the opt-in anonymous ad-snapshot contribution (src/contribute.py).

Every upload goes to a local fake ingest server — nothing here touches the
real endpoint.
"""
import base64
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
import contribute  # noqa: E402
from contribute import SnapshotContributor  # noqa: E402


@pytest.fixture(autouse=True)
def real_clock(monkeypatch):
    """tests/test_vlm_decision_sim.py freezes the GLOBAL time.time at import
    (`H.time.time = lambda: ...`), which leaks into every test collected with
    it. These tests need a moving clock, so take one that doesn't go through
    time.time."""
    from datetime import datetime
    monkeypatch.setattr(time, 'time', lambda: datetime.now().timestamp())


class FakeIngest:
    """Minimal stand-in for server/ingest-server.mjs from the extension repo."""

    def __init__(self, status=200):
        self.status = status
        self.posts = []
        owner = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                body = self.rfile.read(int(self.headers['Content-Length']))
                owner.posts.append({'headers': dict(self.headers), 'body': json.loads(body)})
                self.send_response(owner.status)
                self.end_headers()
                self.wfile.write(b'{"ok":true}')

            def log_message(self, *a):
                pass

        self.httpd = HTTPServer(('127.0.0.1', 0), H)
        self.url = f'http://127.0.0.1:{self.httpd.server_port}/ingest'
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()


@pytest.fixture
def ingest():
    srv = FakeIngest()
    yield srv
    srv.close()


def frame(seed=0, w=1920, h=1080):
    """A textured frame; different seeds give frames dHash sees as different."""
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 255, (9, 16, 3), dtype=np.uint8)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def make(tmp_path, ingest_url, enabled=True, cooldown=0.0):
    return SnapshotContributor(enabled=enabled, queue_dir=tmp_path / 'q', ingest_url=ingest_url,
                               ingest_key='test-key', cooldown_s=cooldown, start_thread=False)


def test_off_by_default_queues_and_sends_nothing(tmp_path, ingest):
    c = SnapshotContributor(queue_dir=tmp_path / 'q', ingest_url=ingest.url, cooldown_s=0,
                            start_thread=False)
    assert c.enabled is False
    assert c.offer(frame(1), block_id=1.0, source='vlm', p_ad=0.9) is False
    assert c.upload_due() == 0
    assert ingest.posts == []
    assert not (tmp_path / 'q').exists()


def test_settings_default_is_off():
    src = (Path(__file__).resolve().parent.parent / 'minus.py').read_text()
    assert "'contribute_snapshots': False" in src


def test_payload_is_anonymous_and_matches_extension_schema(tmp_path, ingest):
    c = make(tmp_path, ingest.url)
    assert c.offer(frame(1), block_id=111.5, source='vlm', p_ad=0.93456) is True
    assert c.upload_due() == 1
    assert len(ingest.posts) == 1
    post = ingest.posts[0]
    hdr = {k.lower(): v for k, v in post['headers'].items()}
    assert hdr.get('x-minus-key') == 'test-key'
    assert hdr.get('user-agent') == 'Minus-TV'      # fixed string, no version/identifier
    body = post['body']
    assert body['v'] == 1 and len(body['samples']) == 1
    s = body['samples'][0]
    # exactly the extension's fields, nothing else (no device/block/time info)
    assert set(s) == {'key', 'img', 'p_ad', 'verdict', 'host', 'w', 'h', 'engine'}
    assert s['verdict'] == 'ad' and s['host'] == 'tv' and s['engine'] == 'minus-tv/vlm'
    assert s['p_ad'] == pytest.approx(0.9346, abs=1e-4)
    assert len(s['key']) == 32
    # downscaled to MAX_SIDE, real PNG
    assert max(s['w'], s['h']) == contribute.MAX_SIDE
    png = base64.b64decode(s['img'].split(',', 1)[1])
    img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
    assert img.shape[1] == s['w'] and img.shape[0] == s['h']
    # uploaded samples leave the local queue
    assert c.status()['queued'] == 0


def test_keys_are_not_linkable(tmp_path, ingest):
    c = make(tmp_path, ingest.url)
    contribute.MIN_SPACING_S, old = 0.0, contribute.MIN_SPACING_S
    try:
        assert c.offer(frame(1), block_id=1.0, source='ocr')
        assert c.offer(frame(2), block_id=1.0, source='ocr')
    finally:
        contribute.MIN_SPACING_S = old
    c.upload_due()
    keys = [s['key'] for s in ingest.posts[0]['body']['samples']]
    assert len(set(keys)) == 2
    assert all(s['p_ad'] is None for s in ingest.posts[0]['body']['samples'])  # OCR-only: no VLM score


def test_cooldown_holds_samples_locally(tmp_path, ingest):
    c = make(tmp_path, ingest.url, cooldown=600)
    assert c.offer(frame(1), block_id=1.0, source='vlm', p_ad=0.8)
    assert c.upload_due() == 0
    assert ingest.posts == []
    assert c.status()['queued'] == 1


def test_pause_retracts_the_block_before_upload(tmp_path, ingest):
    c = make(tmp_path, ingest.url, cooldown=600)
    assert c.offer(frame(1), block_id=1.0, source='vlm', p_ad=0.8)
    assert c.offer(frame(2), block_id=2.0, source='vlm', p_ad=0.8)
    assert c.retract_block(1.0) == 1
    # nothing more is taken from a retracted block
    assert c.offer(frame(3), block_id=1.0, source='vlm', p_ad=0.8) is False
    c.cooldown_s = 0
    assert c.upload_due() == 1
    sent = ingest.posts[0]['body']['samples']
    assert len(sent) == 1
    assert c.status()['retracted_this_session'] == 1


def test_per_block_cap_spacing_and_dedup(tmp_path, ingest):
    c = make(tmp_path, ingest.url, cooldown=600)
    old = contribute.MIN_SPACING_S
    try:
        assert c.offer(frame(1), block_id=5.0, source='vlm')
        # too soon after the last one
        assert c.offer(frame(2), block_id=5.0, source='vlm') is False
        contribute.MIN_SPACING_S = 0.0
        # near-identical frame in the same block
        assert c.offer(frame(1), block_id=5.0, source='vlm') is False
        assert c.offer(frame(2), block_id=5.0, source='vlm')
        assert c.offer(frame(3), block_id=5.0, source='vlm')
        # cap reached
        assert c.offer(frame(4), block_id=5.0, source='vlm') is False
    finally:
        contribute.MIN_SPACING_S = old
    assert c.status()['queued'] == contribute.MAX_PER_BLOCK


def test_opt_out_deletes_queue_and_stops_uploads(tmp_path, ingest):
    c = make(tmp_path, ingest.url, cooldown=600)
    assert c.offer(frame(1), block_id=1.0, source='vlm')
    c.set_enabled(False)
    assert not (tmp_path / 'q').exists()
    c.cooldown_s = 0
    assert c.upload_due() == 0
    assert ingest.posts == []
    assert c.offer(frame(2), block_id=2.0, source='vlm') is False


def test_dead_server_backs_off_and_keeps_samples(tmp_path):
    c = make(tmp_path, 'http://127.0.0.1:9/ingest')    # nothing listens on port 9
    assert c.offer(frame(1), block_id=1.0, source='vlm')
    assert c.upload_due() == 0
    assert c.status()['queued'] == 1
    assert c.last_error
    first = c._next_upload_at - time.time()
    assert 0 < first <= contribute.BACKOFF_BASE_S + 1
    # in backoff: no attempt at all
    assert c.upload_due() == 0
    c._next_upload_at = 0
    c.upload_due()
    assert c._next_upload_at - time.time() > first          # grows


def test_server_error_is_a_failure(tmp_path):
    srv = FakeIngest(status=500)
    try:
        c = make(tmp_path, srv.url)
        assert c.offer(frame(1), block_id=1.0, source='vlm')
        assert c.upload_due() == 0
        assert c.status()['queued'] == 1
    finally:
        srv.close()


def test_queue_caps(tmp_path, ingest, monkeypatch):
    monkeypatch.setattr(contribute, 'MAX_QUEUE', 2)
    monkeypatch.setattr(contribute, 'MIN_SPACING_S', 0.0)
    c = make(tmp_path, ingest.url, cooldown=600)
    for i in range(4):
        assert c.offer(frame(10 + i), block_id=float(i), source='vlm')
    assert c.status()['queued'] == 2
    # TTL: an old sample is dropped
    meta_files = list((tmp_path / 'q').glob('*.json'))
    m = json.loads(meta_files[0].read_text())
    m['_queued_at'] = time.time() - contribute.SAMPLE_TTL_S - 1
    meta_files[0].write_text(json.dumps(m))
    c._enforce_caps()
    assert c.status()['queued'] == 1


def test_offer_never_raises(tmp_path, ingest):
    c = make(tmp_path, ingest.url)
    assert c.offer(None, block_id=1.0, source='vlm') is False
    assert c.offer(np.zeros((0, 0, 3), np.uint8), block_id=1.0, source='vlm') is False
    assert c.offer('not a frame', block_id=1.0, source='vlm') is False
    assert c.offer(frame(1), block_id=None, source='vlm') is False


# --------------------------------------------------------------- minus.py hooks
def _minus_stub(contributor, **state):
    """Bind the real Minus contribution helpers to a light stub object."""
    import importlib
    minus_mod = importlib.import_module('minus')
    st = SimpleNamespace(
        contributor=contributor, ad_detected=True, blocking_source='vlm', blocking_start_time=42.0,
        static_blocking_suppressed=False, config=SimpleNamespace(no_blocking=False),
        _contrib_last_block=(None, 0.0), paused=False)
    st.__dict__.update(state)
    st.is_blocking_paused = lambda: st.paused
    st._contrib_offer = minus_mod.Minus._contrib_offer.__get__(st)
    st._contrib_retract_on_pause = minus_mod.Minus._contrib_retract_on_pause.__get__(st)
    return st


@pytest.fixture
def minus_importable():
    try:
        import minus  # noqa: F401
    except Exception as e:  # GStreamer bindings etc. missing off-device
        pytest.skip(f'minus.py not importable here: {e}')


def test_hooks_offer_only_while_a_block_is_on_screen(tmp_path, ingest, minus_importable):
    c = make(tmp_path, ingest.url, cooldown=600)
    st = _minus_stub(c)
    st._contrib_offer(frame(1), 'vlm', 0.9)
    assert c.status()['queued'] == 1
    for k, v in (('paused', True), ('static_blocking_suppressed', True), ('ad_detected', False)):
        st2 = _minus_stub(c, blocking_start_time=float(hash(k)), **{k: v})
        st2._contrib_offer(frame(2), 'vlm', 0.9)
    assert c.status()['queued'] == 1


def test_hooks_pause_retracts_current_and_just_ended_block(tmp_path, ingest, minus_importable):
    c = make(tmp_path, ingest.url, cooldown=600)
    st = _minus_stub(c, blocking_start_time=1.0)
    st._contrib_offer(frame(1), 'vlm', 0.9)
    st._contrib_retract_on_pause()
    assert c.status()['queued'] == 0
    # a block that ended 10 s ago is still retracted by a pause
    st = _minus_stub(c, blocking_start_time=2.0)
    st._contrib_offer(frame(2), 'vlm', 0.9)
    st.ad_detected = False
    st._contrib_last_block = (2.0, time.time() - 10)
    st._contrib_retract_on_pause()
    assert c.status()['queued'] == 0
    # but not one that ended long ago
    st = _minus_stub(c, blocking_start_time=3.0)
    st._contrib_offer(frame(3), 'vlm', 0.9)
    st.ad_detected = False
    st._contrib_last_block = (3.0, time.time() - 3600)
    st._contrib_retract_on_pause()
    assert c.status()['queued'] == 1


# ------------------------------------------------------------------- web UI
def test_webui_requires_explicit_consent_and_opt_out_deletes(tmp_path, minus_importable):
    import minus as minus_mod
    import webui

    class FakeMinus:
        def __init__(self):
            self._system_settings = {'contribute_snapshots': False}
            self.contributor = SnapshotContributor(queue_dir=tmp_path / 'q', ingest_url='http://127.0.0.1:9/x',
                                                   cooldown_s=600, start_thread=False)

        def _save_system_settings(self):
            pass

        set_contribute_snapshots = minus_mod.Minus.set_contribute_snapshots

    m = FakeMinus()
    client = webui.WebUI(m).app.test_client()
    assert client.get('/api/contribute/status').json['enabled'] is False
    # a bare POST, or anything but consent: true, does not opt in
    assert client.post('/api/contribute/enable').status_code == 400
    assert client.post('/api/contribute/enable', json={'consent': 'yes'}).status_code == 400
    assert m.contributor.enabled is False and m._system_settings['contribute_snapshots'] is False
    r = client.post('/api/contribute/enable', json={'consent': True})
    assert r.status_code == 200 and r.json['enabled'] is True
    assert m._system_settings['contribute_snapshots'] is True
    assert m.contributor.offer(frame(1), block_id=1.0, source='vlm', p_ad=0.9)
    assert client.get('/api/contribute/status').json['queued'] == 1
    r = client.post('/api/contribute/disable')
    assert r.status_code == 200 and r.json['enabled'] is False
    assert m._system_settings['contribute_snapshots'] is False
    assert not (tmp_path / 'q').exists()


def test_settings_page_has_the_consent_dialog():
    html = (Path(__file__).resolve().parent.parent / 'src' / 'templates' / 'index.html').read_text()
    assert 'id="contribute-toggle"' in html and 'id="contribute-modal"' in html
    # the toggle starts unchecked and only the dialog's accept button sends consent
    assert 'id="contribute-toggle" onchange' in html and 'id="contribute-toggle" checked' not in html
    assert "JSON.stringify(enabled ? { consent: true } : {})" in html
