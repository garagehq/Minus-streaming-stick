"""/snapshot/raw staleness: a stale raw frame is re-requested once."""
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import capture  # noqa: E402


class _Resp:
    def __init__(self, age=None, status=200, body=None):
        self.status_code = status
        self.headers = {} if age is None else {'X-Raw-Age': age}
        self.content = body if body is not None else b''


def _jpeg(val):
    ok, buf = cv2.imencode('.jpg', np.full((8, 8, 3), val, np.uint8))
    return buf.tobytes()


def test_stale_detection():
    assert capture.raw_frame_is_stale(_Resp('312.004'))
    assert capture.raw_frame_is_stale(_Resp('1.2'))
    assert not capture.raw_frame_is_stale(_Resp('0.35'))
    assert not capture.raw_frame_is_stale(_Resp(None))        # old ustreamer
    assert not capture.raw_frame_is_stale(_Resp('garbage'))
    assert not capture.raw_frame_is_stale(_Resp('5', status=503))


def _run(responses):
    sess = mock.Mock()
    sess.get.side_effect = responses
    with mock.patch.object(capture, '_get_http_session', return_value=sess), \
         mock.patch.object(capture, '_is_blocking_active', return_value=True), \
         mock.patch.object(capture.time, 'sleep'):
        capture._last_capture_time = 0.0
        img = capture.UstreamerCapture().capture()
    return img, sess.get.call_count


def test_stale_frame_is_requested_again():
    img, calls = _run([_Resp('400.0', body=_jpeg(10)), _Resp('0.1', body=_jpeg(200))])
    assert calls == 2
    assert int(img.mean()) > 150   # the fresh frame, not the stale one


def test_fresh_frame_one_request():
    img, calls = _run([_Resp('0.2', body=_jpeg(200))])
    assert calls == 1 and img is not None


def test_old_ustreamer_no_header_one_request():
    img, calls = _run([_Resp(None, body=_jpeg(200))])
    assert calls == 1 and img is not None
