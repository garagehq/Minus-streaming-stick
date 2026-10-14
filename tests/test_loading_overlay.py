"""Loading spinner (src/loading_overlay.py): frames and plane choice."""
import os
import sys
import time
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import loading_overlay as lo  # noqa: E402

# Trimmed `modetest -M rockchip -p` from minus-2 (RK3588, no TV attached).
PLANES = """Planes:
id\tcrtc\tfb\tCRTC x,y\tx,y\tgamma size\tpossible crtcs
56\t0\t0\t0,0\t\t0,0\t0       \t0x00000007
  formats: XR30 XB30 XR24 AR24 XB24 AB24 RG24 BG24 RG16 BG16 YU08 YU10 YUYV Y210
  props:
\t8 type:
\t\tflags: immutable enum
\t\tenums: Overlay=0 Primary=1 Cursor=2
\t\tvalue: 1
\t63 zpos:
\t\tflags: range
\t\tvalues: 0 11
\t\tvalue: 0
176\t0\t0\t0,0\t\t0,0\t0       \t0x00000007
  formats: XR30 XB30 XR24 AR24 XB24 AB24 RG24 BG24 RG16 BG16 YU08 YU10 YUYV Y210
  props:
\t8 type:
\t\tflags: immutable enum
\t\tenums: Overlay=0 Primary=1 Cursor=2
\t\tvalue: 0
\t183 zpos:
\t\tflags: range
\t\tvalues: 0 11
\t\tvalue: 9
152\t0\t0\t0,0\t\t0,0\t0       \t0x00000007
  formats: XR24 AR24 XB24 AB24 RG24 BG24 RG16 BG16 NV12 NV21
  props:
\t8 type:
\t\tflags: immutable enum
\t\tenums: Overlay=0 Primary=1 Cursor=2
\t\tvalue: 2
\t159 zpos:
\t\tflags: range
\t\tvalues: 0 11
\t\tvalue: 11
192\t0\t0\t0,0\t\t0,0\t0       \t0x00000007
  formats: XR24 AR24 XB24 AB24 RG24 BG24 RG16 BG16 NV12 NV21
  props:
\t8 type:
\t\tflags: immutable enum
\t\tenums: Overlay=0 Primary=1 Cursor=2
\t\tvalue: 0
\t199 zpos:
\t\tflags: range
\t\tvalues: 0 11
\t\tvalue: 14
"""


class TestPlanes(unittest.TestCase):
    def test_parse(self):
        p = lo.parse_planes(PLANES)
        self.assertEqual(sorted(p), [56, 152, 176, 192])
        self.assertEqual(p[192]['type'], 0)
        self.assertEqual(p[192]['zpos'], 14)
        self.assertEqual(p[176]['zpos_range'], (0, 11))
        self.assertTrue(p[176]['zpos_mutable'])
        self.assertIn('AR24', p[56]['formats'])

    def test_pick_prefers_spare_overlay_plane(self):
        p = lo.parse_planes(PLANES)
        self.assertEqual(lo.pick_plane(p, 192), 176)      # not the cursor, not primary
        self.assertEqual(lo.pick_plane(p, 176), 192)      # another spare overlay plane

    def test_pick_skips_planes_in_use(self):
        p = lo.parse_planes(PLANES)
        p[176]['crtc'] = 88
        self.assertEqual(lo.pick_plane(p, 192), 56)
        self.assertIsNone(lo.pick_plane(p, 999))


class TestFrames(unittest.TestCase):
    def setUp(self):
        self.spin = lo.render_spinner(n=6, size=60, h=180)
        self.badge = lo.render_badge()

    def test_spinner_frames(self):
        self.assertEqual(len(self.spin), 6)
        self.assertEqual(self.spin[0].shape, (60, 60, 4))
        self.assertFalse(np.array_equal(self.spin[0], self.spin[1]))    # it moves

    def test_spinner_layer(self):
        x = lo.compose(self.spin[0], None, w=320, h=180)
        # Premultiplied: no colour channel exceeds alpha.
        self.assertTrue((x[..., :3].max(axis=2) <= x[..., 3]).all())
        # Corners are the plain dim layer.
        self.assertEqual(int(x[0, 0, 3]), round(lo.DIM_ALPHA * 255))
        self.assertEqual(int(x[0, 0, :3].max()), 0)
        # The spinner is white and brighter than the dim layer.
        self.assertGreater(int(x[60:120, 130:190, 0].max()), 200)

    def test_clear_layer(self):
        self.assertFalse(lo.compose(None, None, w=320, h=180).any())

    def test_badge_top_right(self):
        bh, bw = self.badge.shape[:2]
        self.assertEqual(bh, lo.BADGE_H)
        x = lo.compose(None, self.badge)
        self.assertTrue((x[..., :3].max(axis=2) <= x[..., 3]).all())
        m = lo.BADGE_MARGIN
        box = x[m:m + bh, lo.W - m - bw:lo.W - m]
        # Opaque red bubble (BGRA), white text inside, nothing elsewhere.
        r, g, b, a = lo.BADGE_RGBA
        self.assertEqual(tuple(int(v) for v in box[bh // 2, 8]), (
            round(b * a / 255), round(g * a / 255), round(r * a / 255), a))
        self.assertTrue((box[..., :3].min(axis=2) > 240).any())
        self.assertEqual(int(box[0, 0, 3]), 0)             # rounded corner
        rest = x.copy()
        rest[m:m + bh, lo.W - m - bw:lo.W - m] = 0
        self.assertFalse(rest.any())

    def test_badge_over_spinner_stays_opaque(self):
        x = lo.compose(lo.render_spinner(n=1)[0], self.badge)
        m = lo.BADGE_MARGIN
        self.assertGreaterEqual(int(x[m + lo.BADGE_H // 2, lo.W - m - 20, 3]), lo.BADGE_RGBA[3])


class TestHeadless(unittest.TestCase):
    class Video:
        pipeline = None
        plane_id = 192
        connector_id = 215

    def test_no_display_pipeline_is_a_no_op(self):
        ov = lo.ScreenOverlay(self.Video())
        ov.show(lambda: False, max_s=0.2)
        ov.set_blocked(True)
        time.sleep(0.3)
        st = ov.get_status()
        self.assertEqual((st['shown'], st['badge_shown'], st['blocked_now']), (0, 0, False))
        self.assertEqual(st['last_error'], '')

    def test_one_overlay_per_video(self):
        v = self.Video()
        self.assertIs(lo.for_video(v), lo.for_video(v))


if __name__ == '__main__':
    unittest.main()
