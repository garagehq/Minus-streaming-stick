"""A block's mute must survive the audio pipeline being rebuilt.

Seen live 2026-10-29 03:00 UTC: an ad was blocked (audio MUTED), the TV re-hotplugged
twice, DRMAdBlocker restarted audio for the HDMI reconnect (AudioPassthrough.stop() +
start()), and the new pipeline came up with `volume mute=false` while `is_muted`
stayed True - so the ad played with sound for the rest of the block and nothing
noticed (every check read is_muted). The watchdog restart paths re-applied the
mute; start() did not. The pipeline is now built with the current mute state, so
every rebuild path (start, restart, reconnect) comes up muted mid-block.
"""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def _launch_string(is_muted, name_muted=False):
    from audio import AudioPassthrough
    captured = {}

    def fake_parse_launch(s):
        captured['str'] = s
        raise RuntimeError("fake-parse-launch (test)")
    with patch('audio.Gst.parse_launch', side_effect=fake_parse_launch), \
         patch('audio.detect_hdmi_capture_device', return_value='hw:4,0'), \
         patch.object(AudioPassthrough, '_hdmi_tx_connected', return_value=True):
        ap = AudioPassthrough(capture_device='hw:4,0', playback_device='hw:0,0')
        ap.is_muted = is_muted
        ap._name_muted = name_muted
        try:
            ap._init_pipeline()
        except RuntimeError as e:
            if "fake-parse-launch" not in str(e):
                raise
    return captured.get('str', '')


class TestMuteSurvivesRebuild(unittest.TestCase):
    def test_pipeline_built_while_muted_starts_muted(self):
        s = _launch_string(is_muted=True)
        self.assertIn('volume name=vol', s)
        self.assertIn('mute=true', s)
        self.assertNotIn('mute=false', s)

    def test_pipeline_built_while_unmuted_starts_unmuted(self):
        s = _launch_string(is_muted=False)
        self.assertIn('mute=false', s)

    def test_pipeline_built_mid_name_mute_starts_muted(self):
        """asr-first: a LeBron name mute in progress survives a rebuild too."""
        s = _launch_string(is_muted=False, name_muted=True)
        self.assertIn('mute=true', s)
        self.assertNotIn('mute=false', s)

    def test_volume_element_mute_reasserted_after_build(self):
        """Belt and braces: the volume element is set muted right after it is found."""
        from audio import AudioPassthrough
        vol = MagicMock()
        pipe = MagicMock()
        pipe.get_by_name.side_effect = lambda n: vol if n == 'vol' else MagicMock()
        with patch('audio.Gst.parse_launch', return_value=pipe), \
             patch('audio.detect_hdmi_capture_device', return_value='hw:4,0'), \
             patch.object(AudioPassthrough, '_hdmi_tx_connected', return_value=True):
            ap = AudioPassthrough(capture_device='hw:4,0', playback_device='hw:0,0')
            ap.is_muted = True
            ap._init_pipeline()
        vol.set_property.assert_any_call('mute', True)


if __name__ == '__main__':
    unittest.main()
