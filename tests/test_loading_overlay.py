"""Loading spinner (src/loading_overlay.py): frames and plane choice."""
import os
import sys
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
    def test_frames(self):
        frames, clear = lo.render_frames(w=320, h=180, n=6)
        self.assertEqual(len(frames), 6)
        self.assertEqual(clear, bytes(320 * 180 * 4))
        px = [np.frombuffer(f, np.uint8).reshape(180, 320, 4) for f in frames]
        for x in px:
            # Premultiplied: no colour channel exceeds alpha.
            self.assertTrue((x[..., :3].max(axis=2) <= x[..., 3]).all())
            # Corners are the plain dim layer.
            self.assertEqual(int(x[0, 0, 3]), round(lo.DIM_ALPHA * 255))
            self.assertEqual(int(x[0, 0, :3].max()), 0)
            # The spinner is white and brighter than the dim layer.
            self.assertGreater(int(x[60:120, 130:190, 0].max()), 200)
        self.assertFalse(np.array_equal(px[0], px[1]))    # it moves


class TestHeadless(unittest.TestCase):
    def test_no_display_pipeline_is_a_no_op(self):
        class Video:
            pipeline = None
            plane_id = 192
            connector_id = 215
        ov = lo.LoadingOverlay(Video())
        ov.show(lambda: False, max_s=0.2)
        ov._thread.join(2)
        self.assertEqual(ov.shown_count, 0)
        self.assertEqual(ov.last_error, '')


if __name__ == '__main__':
    unittest.main()
