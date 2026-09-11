"""
Configuration classes for Minus.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path


def _get_env_path(env_var: str, default: str) -> str:
    """Get path from environment variable or use default."""
    return os.environ.get(env_var, default)


def _get_env_float(env_var: str, default: float) -> float:
    """Get float from environment variable or use default."""
    val = os.environ.get(env_var)
    if val is not None:
        try:
            return float(val)
        except ValueError:
            pass
    return default


def _get_env_int(env_var: str, default: int) -> int:
    """Get int from environment variable or use default."""
    val = os.environ.get(env_var)
    if val is not None:
        try:
            return int(val)
        except ValueError:
            pass
    return default


@dataclass
class MinusConfig:
    """Configuration for the Minus pipeline."""
    # Device and paths
    device: str = "/dev/video0"
    screenshot_dir: str = "screenshots"

    # Timeouts and thresholds
    ocr_timeout: float = 1.5
    vlm_timeout: float = 3.0  # Hard timeout for VLM inference (seconds)

    # ustreamer settings
    ustreamer_port: int = 9090

    # Screenshot management
    max_screenshots: int = 0  # 0 = unlimited (keep all for training)

    # DRM output (auto-detected if not specified)
    drm_connector_id: int = None  # Auto-detect HDMI output connector
    drm_plane_id: int = None  # Auto-detect NV12-capable overlay plane
    output_width: int = None  # Auto-detect from display EDID
    output_height: int = None  # Auto-detect from display EDID

    # Audio settings (auto-detected if not specified)
    audio_capture_device: str = "auto"  # HDMI-RX audio input - auto-detected from /proc/asound/cards
    audio_playback_device: str = None  # Auto-detect based on connected HDMI output

    # Web UI
    webui_port: int = 80  # Web UI port (port 80 requires root)

    # Feature toggles
    no_ocr: bool = False  # Disable OCR processing (for testing)
    no_vlm: bool = False  # Disable VLM processing (for testing)
    no_blocking: bool = False  # Disable blocking overlays (for testing)

    # Animation durations (seconds)
    animation_start_duration: float = field(
        default_factory=lambda: _get_env_float('MINUS_ANIMATION_START', 0.3)
    )
    animation_end_duration: float = field(
        default_factory=lambda: _get_env_float('MINUS_ANIMATION_END', 0.25)
    )

    # Health check thresholds
    frame_stale_threshold: float = field(
        default_factory=lambda: _get_env_float('MINUS_FRAME_STALE_THRESHOLD', 5.0)
    )

    # Detection thresholds
    vlm_alone_threshold: int = field(
        default_factory=lambda: _get_env_int('MINUS_VLM_ALONE_THRESHOLD', 5)
    )
    scene_change_threshold: float = field(
        # 0.001 — measured min/p25/p50 inter-frame mean-abs-diff on real
        # video content (BBB) at the OCR sample cadence: p5≈0.002, p50≈0.017,
        # max=0.31 (scene cuts). Frozen frames sit at 0. The previous 0.01
        # default classified ~26% of natural low-motion frames as "static"
        # and tripped STATIC_OCR_THRESHOLD on long static-suppression runs
        # mid-content. 0.001 only flags genuinely-frozen frames (~1.7% of
        # BBB), correctly distinguishing user-pause from slow content.
        # Tuned via tests/block_latency_harness.py.
        default_factory=lambda: _get_env_float('MINUS_SCENE_CHANGE_THRESHOLD', 0.001)
    )
    dynamic_cooldown: float = field(
        # 1.5s — long enough for screen content to actually settle into the
        # post-pause/post-static state. The previous 0.5s was too short:
        # the first OCR cycle after cooldown often still saw frames that
        # were transitioning off the ad and immediately re-triggered blocking.
        default_factory=lambda: _get_env_float('MINUS_DYNAMIC_COOLDOWN', 1.5)
    )


# External paths (configurable via environment variables)
USTREAMER_PATH = _get_env_path('MINUS_USTREAMER_PATH', '/home/radxa/ustreamer-patched')
# minus-v0.1 ad-classifier — fine-tuned LFM2.5-VL-450M (iter28) compiled for
# fused-layer NPU3 prefill with logit-argmax decisions.
# Published at https://huggingface.co/TheGarageDev/Minus-v0.1
# (benchmarks: ~/axera_models/BENCHMARKS.md). detect_ad AND autonomous-mode
# query_image both share this single model — there is no FastVLM dependency
# anymore. Override with MINUS_VLM_MODEL_DIR.
VLM_MODEL_DIR = _get_env_path('MINUS_VLM_MODEL_DIR', '/home/radxa/axera_models/minus-v0.1')
# PaddleOCR det/rec RKNN models. They ship in the repository
# (models/paddleocr/, see its README) so a fresh clone needs no download;
# the legacy rknn-llm demo path is only used when the repo copy is absent.
# Override with MINUS_OCR_MODEL_DIR.
_REPO_OCR_MODEL_DIR = Path(__file__).resolve().parent.parent / 'models' / 'paddleocr'
_LEGACY_OCR_MODEL_DIR = '/home/radxa/rknn-llm/examples/multimodal_model_demo/deploy/install/demo_Linux_aarch64/models/paddleocr'
OCR_MODEL_DIR = _get_env_path(
    'MINUS_OCR_MODEL_DIR',
    str(_REPO_OCR_MODEL_DIR) if any(_REPO_OCR_MODEL_DIR.glob('ppocrv3_det_*.rknn')) else _LEGACY_OCR_MODEL_DIR,
)

# Which PP-OCR generation to load. Both sets of weights ship side by side in
# models/paddleocr/ and are selected here rather than by renaming files.
#   'v6'   — PP-OCRv6 tiny (default since Sep 2026). Chosen on latency, not
#            accuracy: a 5.67h production soak measured OCR inference p50
#            195ms / p90 257ms against v3's 231ms idle / 330ms in-block and
#            p90 550ms, and ZERO hard timeouts against 3 in 36h of v3. The
#            tighter tail is what keeps text-dense ad frames under the 1.5s
#            hard timeout. Accuracy is a wash (40 identical ad frames through
#            the production keyword matcher: v3 39/40, v6 38/40).
#   'v3'   — PP-OCRv3, kept as a one-env-var rollback
#   'auto' — v6 when present, else v3
# Each generation needs its OWN dictionary and its OWN DB post-process
# thresholds; mixing them silently degrades OCR (a v3 dict against v6 weights
# decodes to wrong characters entirely), so they are bound together below.
OCR_MODEL_VERSION = os.environ.get('MINUS_OCR_MODEL_VERSION', 'v6').strip().lower()

OCR_MODEL_GENERATIONS = {
    # DB thresholds come from each release's inference.yml.
    'v3': {
        'det_glob': 'ppocrv3_det_*.rknn',
        'rec_glob': 'ppocrv3_rec_*.rknn',
        'dict_name': 'ppocr_keys_v1.txt',
        'db_params': {'thresh': 0.3, 'box_thresh': 0.5, 'unclip_ratio': 1.5},
    },
    'v6': {
        'det_glob': 'ppocrv6_det_*.rknn',
        'rec_glob': 'ppocrv6_rec_*.rknn',
        'dict_name': 'ppocrv6_keys.txt',
        'db_params': {'thresh': 0.2, 'box_thresh': 0.4, 'unclip_ratio': 1.4},
    },
}


def resolve_ocr_models(base_dir=None, version=None):
    """Resolve the OCR model set to load.

    Returns a dict with det/rec/dict paths, db_params and the resolved
    version, or None when no complete set is present. Never raises.
    """
    base = Path(base_dir or OCR_MODEL_DIR)
    requested = (version or OCR_MODEL_VERSION or 'v3').strip().lower()
    order = (['v6', 'v3'] if requested == 'auto'
             else [requested] if requested in OCR_MODEL_GENERATIONS
             else ['v6', 'v3'])

    for name in order:
        gen = OCR_MODEL_GENERATIONS[name]
        try:
            det = sorted(base.glob(gen['det_glob']))
            rec = sorted(base.glob(gen['rec_glob']))
            dictionary = base / gen['dict_name']
            if det and rec and dictionary.exists():
                return {
                    'version': name,
                    'det': str(det[0]),
                    'rec': str(rec[0]),
                    'dict': str(dictionary),
                    'db_params': dict(gen['db_params']),
                    'base_dir': str(base),
                }
        except OSError:
            continue
    return None
