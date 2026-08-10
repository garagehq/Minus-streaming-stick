"""
VLM (Vision Language Model) integration for Minus.

Uses minus-v0.1 (fine-tuned LFM2.5-VL-450M, iter28 — published at
https://huggingface.co/TheGarageDev/Minus-v0.1) on the Axera LLM 8850
NPU for ad detection AND autonomous-mode screen-state classification.

Both `detect_ad()` and `query_image()` are prefill-only on the 16
fused decoder layers — no autoregressive decode, no KV cache state
to manage between inferences. Total per-image latency ~370ms
deterministic (vision ~185ms + prefill ~185ms).

- `detect_ad`: argmax of max(YES_logits) vs max(NO_logits) from the
  last-token vocab logits. Also exposes `p_yes_norm` for an optional
  tunable threshold (env `MINUS_VLM_AD_THRESHOLD`); default 0.5 ≡
  argmax. Holdout (800 images): 97.0% acc, 94.8% ad-recall, 99.2%
  non-ad-recall.

- `query_image`: prefill-only multi-class logit lookup over the first
  token of each screen-state class (PLAYING / PAUSED / DIALOG / MENU /
  SCREENSAVER). Returns the class name with the highest first-token
  logit (max over no-leading-space and leading-space variants). No
  autoregressive decode — the per-layer decode axmodels are not
  shipped with this v2 build; only `post_d.axmodel` is. Single prefill
  takes ~370ms, same as detect_ad.

Migration notes vs FastVLM-0.5B iter4 (the previous model):
  - Tokenizer is the LFM custom one (NOT Qwen2). All token IDs are
    different.
  - 256 image tokens (16×16 grid) instead of 64 (8×8).
  - Vision preprocessing is direct bilinear resize → patchify into
    (1, 1024, 768); not `expand2square` + CLIPImageProcessor.
  - All axmodel I/O is FP32 (no bfloat16 ceremony).
  - 16 fused-layer axmodels (10 conv + 6 attn) instead of 24 separate.
  - Per-call conv state allocated fresh in `detect_ad`/`query_image`
    — there is no persistent state to reset between calls. The
    `_reset_kv_cache()` shim is kept as a no-op for backward compat.
  - Prompt format is byte-exact per the fine-tune, including the BOS
    + IM_START + system + IM_END skeleton — do NOT edit the prompt
    strings in `_build_prompt_ids()` without recalibrating.

Reference: /home/radxa/axera_models/LFM2/MINUS_INTEGRATION_GUIDE.md
Working standalone: /home/radxa/axera_models/minus-v0.1/infer_vlm_fused.py

Model is loaded ONCE at startup (~8s) and kept running.
"""

import os
import sys
import time
import logging
import threading
from pathlib import Path

import numpy as np
from PIL import Image

from config import VLM_MODEL_DIR

logger = logging.getLogger('Minus.VLM')

# --- Model file layout ---
# Default points at the LFM2.5-VL fused-v2 build; override the base
# dir via MINUS_VLM_MODEL_DIR.
LFM_MODEL_DIR = Path(VLM_MODEL_DIR)
FUSED_DIR     = LFM_MODEL_DIR / "fused_models"
VISION_PATH   = LFM_MODEL_DIR / "vision_encoder_512.axmodel"
POST_PATH     = LFM_MODEL_DIR / "decode_models" / "post_d.axmodel"
EMBEDS_PATH   = LFM_MODEL_DIR / "embed.npy"
TOKENIZER_FILE = LFM_MODEL_DIR / "tokenizer.json"

# Backwards-compat alias — older tests / external callers still import
# `FASTVLM_MODEL_DIR`. Resolves to the current VLM model dir regardless
# of which family is configured via MINUS_VLM_MODEL_DIR.
FASTVLM_MODEL_DIR = LFM_MODEL_DIR

# Decoder layer execution order. 10 conv + 6 full-attention = 16.
# Matches the on-disk axmodel filenames (l{i}_{conv|attn}_fused.axmodel).
# DO NOT REORDER — this is the model architecture itself.
LAYER_TYPES = (
    "conv", "conv", "attn",
    "conv", "conv", "attn",
    "conv", "conv", "attn",
    "conv",         "attn",
    "conv",         "attn",
    "conv",         "attn",
    "conv",
)
assert len(LAYER_TYPES) == 16


class VLMManager:
    """
    LFM2.5-VL-450M ad-classifier on Axera LLM 8850 NPU.

    Loaded once at init; both inference paths share the same prefill loop.
    """

    # --- Image / prompt config (LFM-specific) ---
    INPUT_SIZE     = 512
    NUM_IMG_TOKENS = 256             # 16×16 grid
    PREFILL_LEN    = 320             # padded buffer (37 text + 256 image = 293)
    HIDDEN_SIZE    = 1024
    PATCH_SIZE     = 16
    GRID_SIZE      = 32              # 512 / PATCH_SIZE
    VISION_HIDDEN  = 768
    IMG_TOKEN_ID   = 396
    CONV_L_CACHE   = 3               # per-layer conv state depth

    # LFM tokenizer special token IDs (probed from tokenizer.json)
    BOS_TOKEN_ID  = 1
    IM_START_ID   = 6
    IM_END_ID     = 7
    IMG_START_ID  = 498
    IMG_END_ID    = 499

    # Yes/No token IDs in the LFM tokenizer.
    # ("Yes","yes"," Yes"," yes") / ("No","no"," No"," no")
    YES_TOKEN_IDS = (12948, 12184, 18051, 18672)
    NO_TOKEN_IDS  = (5048, 2744, 3253, 1295)

    # First-token IDs for each autonomous-mode screen-state class.
    # We use the leading-space-vs-no-space PAIR for each — max over both
    # — so the lookup matches whether the model emits with or without a
    # leading space after "assistant\n". Two-token classes ("M","PL"...)
    # are sufficient for argmax-based class selection; we never need to
    # decode past the first emitted token.
    SCREEN_CLASS_TOKEN_IDS = (
        ("PLAYING",     (9436,  60595)),   # "PL",   " PLA"
        ("PAUSED",      (7055,  16137)),   # "PA",   " PA"
        ("DIALOG",      (20546, 23238)),   # "DI",   " DI"
        ("MENU",        (554,   857)),     # "M",    " M"
        ("SCREENSAVER", (8421,  14151)),   # "SC",   " SC"
    )

    # Decision threshold on p_yes_norm. Default 0.5 == argmax (equivalent
    # to max(YES_logits) > max(NO_logits)). LFM holdout-calibrated argmax
    # gives 97.0% accuracy / 94.8% ad-recall / 99.2% non-ad-recall — the
    # gain from threshold tuning is small. Override via env to bias
    # ad-recall vs non-ad-recall without re-running inference.
    AD_THRESHOLD = float(os.environ.get('MINUS_VLM_AD_THRESHOLD', '0.5'))

    # Ad-detection prompt. MUST be byte-exact — the fine-tune was trained
    # on this exact prompt string with the LFM chat template.
    AD_PROMPT_TEXT = "Is this an advertisement? Answer Yes or No."

    # Compatibility shims for old callers (FastVLM legacy):
    AD_SYSTEM_PROMPT = "You are a helpful multimodal assistant by Liquid AI."
    AD_PROMPT        = AD_PROMPT_TEXT
    TOKEN_LENGTH     = NUM_IMG_TOKENS

    def __init__(self):
        """Initialize VLM manager."""
        self.is_ready = False
        self._lock = threading.Lock()

        # Model components (populated by load_model)
        self.tokenizer = None
        self.vision = None
        self.fused = None
        self.post = None
        self.embeds = None
        self._ad_prompt_ids = None   # Cached: prompt for detect_ad
        self._screen_prompt_ids = None  # Cached: prompt for query_image

        # Validate paths up front
        for label, p in (("LFM model dir", LFM_MODEL_DIR),
                          ("fused dir", FUSED_DIR),
                          ("vision encoder", VISION_PATH),
                          ("post", POST_PATH),
                          ("embeds", EMBEDS_PATH),
                          ("tokenizer", TOKENIZER_FILE)):
            if not p.exists():
                logger.error(f"{label} not found: {p}")
                return

        logger.info(f"VLM using minus-v0.1 (LFM2.5-VL) at: {LFM_MODEL_DIR}")

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_model(self):
        """Load model — 17 axmodel sessions + embeds + tokenizer (~8s)."""
        if self.is_ready:
            logger.info("Model already loaded")
            return True

        try:
            logger.info("Starting LFM2.5-VL model...")
            t0 = time.time()

            try:
                import axengine as ax
                from transformers import PreTrainedTokenizerFast
            except ImportError as e:
                logger.error(f"Missing dependency: {e}")
                logger.error("Install: pip3 install axengine transformers")
                return False

            # 1× vision encoder
            logger.info("  Loading vision encoder...")
            self.vision = ax.InferenceSession(str(VISION_PATH))

            # 16× fused decoder layers, in execution order
            logger.info("  Loading 16 fused decoder layers...")
            self.fused = []
            for i, lt in enumerate(LAYER_TYPES):
                fname = f"l{i}_{lt}_fused.axmodel"
                self.fused.append(ax.InferenceSession(str(FUSED_DIR / fname)))

            # 1× post (LM head projection)
            logger.info("  Loading post (LM head)...")
            self.post = ax.InferenceSession(str(POST_PATH))

            # Embedding table — mmap so we don't resident-load 256MB
            logger.info("  Loading embedding table (mmap)...")
            self.embeds = np.load(str(EMBEDS_PATH), mmap_mode='r')

            # Tokenizer — pure-Python tokenizer.json, no chat template
            logger.info("  Loading tokenizer...")
            self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(TOKENIZER_FILE))

            # Pre-build the two prompts we use (ad-detection + screen
            # classification). Their token IDs are identical across all
            # calls — only the image features change. Caching saves a
            # few ms per inference and avoids tokenizer thread issues.
            self._ad_prompt_ids = self._build_prompt_ids(self.AD_PROMPT_TEXT)
            # The screen prompt is identical to autonomous_mode's
            # SCREEN_QUERY_PROMPT. We don't import that to avoid a
            # circular dependency; this string MUST match it byte-exact.
            # KEEP IT SHORT — the full prompt (text + 256 image + chat
            # template) MUST fit in PREFILL_LEN=320 tokens or the
            # [IM_START] assistant\n suffix gets truncated and the
            # last-position logits become garbage. The previous, longer
            # phrasing tokenised to 326 tokens — over by 6 — and silently
            # truncated.
            screen_prompt = (
                "Classify this TV screen: PLAYING, PAUSED, DIALOG, MENU, or SCREENSAVER?"
            )
            self._screen_prompt_ids = self._build_prompt_ids(screen_prompt)

            # Fail loud if any cached prompt overflows the fixed prefill
            # window. Future edits to the prompt strings will trip this
            # before they silently degrade accuracy.
            for label, ids in (("ad-prompt", self._ad_prompt_ids),
                                ("screen-prompt", self._screen_prompt_ids)):
                if len(ids) > self.PREFILL_LEN:
                    logger.error(
                        f"{label} is {len(ids)} tokens > PREFILL_LEN="
                        f"{self.PREFILL_LEN}. The [IM_START] assistant suffix "
                        f"will be truncated and inference will be garbage. "
                        f"Shorten the prompt."
                    )
                    return False

            load_time = time.time() - t0
            logger.info(
                f"LFM2.5-VL loaded in {load_time:.1f}s "
                f"(ad-prompt={len(self._ad_prompt_ids)}t, "
                f"screen-prompt={len(self._screen_prompt_ids)}t)"
            )
            self.is_ready = True
            return True

        except Exception as e:
            logger.error(f"Failed to load LFM2.5-VL: {e}")
            import traceback
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            return False

    # ------------------------------------------------------------------
    # Compat shims (FastVLM legacy callers)
    # ------------------------------------------------------------------

    def _reset_kv_cache(self):
        """No-op for LFM (no persistent state between inferences)."""
        pass

    def start_tokenizer_service(self):
        """Compatibility shim — alias for load_model()."""
        return self.load_model()

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _encode_image(self, image_path):
        """Run vision encoder on `image_path` → (1, 256, 1024) FP32.

        LFM uses direct bilinear resize to 512×512 + (x/255 - 0.5)/0.5
        normalization, then a patchify into (1, 1024, 768). DO NOT
        substitute FastVLM-style expand2square/CLIPImageProcessor —
        accuracy degrades.
        """
        img = Image.open(image_path).convert("RGB")
        if img.size != (self.INPUT_SIZE, self.INPUT_SIZE):
            img = img.resize((self.INPUT_SIZE, self.INPUT_SIZE), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32)
        arr = (arr / 255.0 - 0.5) / 0.5    # → [-1, 1]
        # (512, 512, 3) → (32, 16, 32, 16, 3) → (1024, 16, 16, 3) → (1024, 768)
        x = arr.reshape(self.GRID_SIZE, self.PATCH_SIZE,
                        self.GRID_SIZE, self.PATCH_SIZE, 3)
        x = x.transpose(0, 2, 1, 3, 4).reshape(
            self.GRID_SIZE * self.GRID_SIZE, -1)
        patches = x.reshape(1, self.GRID_SIZE * self.GRID_SIZE,
                            self.VISION_HIDDEN)
        out = self.vision.run(None, {"pixel_values": patches})[0]
        return out.astype(np.float32, copy=False)

    def _build_prompt_ids(self, user_question_text):
        """Build the full prompt token sequence for a given user question.

        Format (must be byte-exact for the fine-tune to apply):

          [BOS, IM_START] system "system\\n..." [IM_END] \\n
          [IM_START] user "user\\n" [IMG_START] [IMG_TOKEN]×256 [IMG_END]
          <user_question> [IM_END] \\n
          [IM_START] assistant "assistant\\n"

        ~37 text tokens + 256 image tokens = ~293 total (well under PREFILL_LEN=320).
        """
        BOS      = self.BOS_TOKEN_ID
        IM_START = self.IM_START_ID
        IM_END   = self.IM_END_ID
        IMG_START = self.IMG_START_ID
        IMG_END   = self.IMG_END_ID

        def enc_strip_bos(s):
            ids = self.tokenizer.encode(s)
            return ids[1:] if ids and ids[0] == BOS else ids

        system_text = "system\nYou are a helpful multimodal assistant by Liquid AI."
        system_toks = enc_strip_bos(system_text)
        user_toks   = enc_strip_bos("user\n")
        prompt_toks = enc_strip_bos(user_question_text)
        asst_toks   = enc_strip_bos("assistant\n")
        nl_tok      = enc_strip_bos("\n")

        return ([BOS, IM_START] + system_toks + [IM_END] + nl_tok +
                [IM_START] + user_toks + [IMG_START] +
                [self.IMG_TOKEN_ID] * self.NUM_IMG_TOKENS +
                [IMG_END] + prompt_toks + [IM_END] + nl_tok +
                [IM_START] + asst_toks)

    # ------------------------------------------------------------------
    # Prefill (shared by detect_ad + query_image)
    # ------------------------------------------------------------------

    def _prefill_last_logits(self, vision_out, prompt_ids):
        """Run vision-feature-spliced prefill, return last-real-token logits.

        Args:
            vision_out: (1, 256, 1024) FP32 — from `_encode_image`.
            prompt_ids: list[int] — from `_build_prompt_ids`.

        Returns:
            np.ndarray, (vocab_size,) FP32 — logits at the last real
            (non-padding) prompt position.
        """
        n_tokens = min(len(prompt_ids), self.PREFILL_LEN)
        prompt_arr = np.array(prompt_ids[:n_tokens], dtype=np.int64)

        # Locate image-token slots
        img_positions = np.where(prompt_arr == self.IMG_TOKEN_ID)[0]
        img_start_pos = int(img_positions[0]) if len(img_positions) else -1

        # Build the (1, 320, 1024) prefill input: text embeddings + vision features
        prefill_data = np.zeros(
            (1, self.PREFILL_LEN, self.HIDDEN_SIZE), dtype=np.float32)
        prefill_data[0, :n_tokens, :] = self.embeds[prompt_arr].astype(
            np.float32, copy=False)
        if img_start_pos >= 0:
            n_v = min(len(img_positions), vision_out.shape[1])
            prefill_data[0, img_start_pos:img_start_pos + n_v, :] = \
                vision_out[0, :n_v, :]

        # Causal mask, vectorized. -65536 outside the live window, causal
        # triangle inside the [0:n_tokens, 0:n_tokens] block.
        mask = np.full((1, self.PREFILL_LEN, self.PREFILL_LEN),
                       -65536.0, dtype=np.float32)
        causal = np.triu(np.ones((n_tokens, n_tokens), dtype=np.float32), k=1)
        mask[0, :n_tokens, :n_tokens] = causal * -65536.0
        indices = np.arange(self.PREFILL_LEN, dtype=np.int32).reshape(
            1, self.PREFILL_LEN)

        # Per-layer conv state, allocated fresh (no persistent state)
        conv_states = {}
        for i, lt in enumerate(LAYER_TYPES):
            if lt == "conv":
                conv_states[i] = np.zeros(
                    (1, self.HIDDEN_SIZE, self.CONV_L_CACHE), dtype=np.float32)

        # Run all 16 fused layers sequentially
        data = prefill_data
        for i, lt in enumerate(LAYER_TYPES):
            if lt == "conv":
                outs = self.fused[i].run(None, {
                    "hidden": data,
                    "conv_state_in": conv_states[i]})
                data = outs[0]
                conv_states[i] = outs[1]
            else:  # attn
                outs = self.fused[i].run(None, {
                    "hidden": data,
                    "mask": mask,
                    "indices": indices})
                data = outs[0]

        # Post: project last-real-token hidden state → vocab logits
        last_hidden = data[:, n_tokens - 1:n_tokens, :]
        logits = self.post.run(None, {"input": last_hidden})[0].flatten()
        return logits, n_tokens

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_ad(self, image_path):
        """Ad/not-ad classification.

        Returns:
            (is_ad: bool, response_text: str, elapsed: float, confidence: float)

        confidence ∈ [0,1] is `p_yes_norm` for ads, `1 - p_yes_norm` for
        non-ads — feeds the existing sliding-window decision engine.
        """
        if not self.is_ready:
            return False, "VLM not ready", 0, 0.0
        if not os.path.exists(image_path):
            return False, f"Image not found: {image_path}", 0, 0.0

        with self._lock:
            try:
                t0 = time.time()

                vision_out = self._encode_image(image_path)
                logits, _ = self._prefill_last_logits(
                    vision_out, self._ad_prompt_ids)

                # Argmax-of-spelling decision (matches reference script)
                p_yes_logit = float(max(logits[t] for t in self.YES_TOKEN_IDS))
                p_no_logit  = float(max(logits[t] for t in self.NO_TOKEN_IDS))

                # Also compute p_yes_norm in the {YES, NO} subspace for
                # tunable thresholding. Softmax-normalize over only the
                # 8 yes/no tokens for numerical stability with raw logits.
                yn_logits = np.array(
                    [logits[t] for t in self.YES_TOKEN_IDS] +
                    [logits[t] for t in self.NO_TOKEN_IDS], dtype=np.float32)
                yn_logits = yn_logits - yn_logits.max()
                yn_probs = np.exp(yn_logits)
                yn_probs /= yn_probs.sum()
                p_yes_sub = float(yn_probs[:len(self.YES_TOKEN_IDS)].sum())
                p_no_sub  = float(yn_probs[len(self.YES_TOKEN_IDS):].sum())
                p_yes_norm = p_yes_sub / (p_yes_sub + p_no_sub + 1e-9)

                if abs(self.AD_THRESHOLD - 0.5) < 1e-6:
                    # Pure argmax — matches eval_fused.py exactly
                    is_ad = p_yes_logit > p_no_logit
                else:
                    is_ad = p_yes_norm > self.AD_THRESHOLD

                confidence = p_yes_norm if is_ad else (1.0 - p_yes_norm)
                response = f"{'Yes' if is_ad else 'No'} (p={p_yes_norm:.4f})"

                elapsed = time.time() - t0
                logger.info(
                    f"VLM(LFM2): {'AD' if is_ad else 'NO-AD'} "
                    f"p_yes={p_yes_norm:.4f} "
                    f"y_logit={p_yes_logit:.3f} n_logit={p_no_logit:.3f} "
                    f"T={self.AD_THRESHOLD} lat={elapsed:.3f}s"
                )
                return is_ad, response, elapsed, confidence

            except Exception as e:
                logger.error(f"VLM inference error: {e}")
                import traceback
                logger.error(f"Traceback:\n{traceback.format_exc()}")
                return False, str(e), time.time() - t0, 0.0

    def query_image(self, image_path, prompt, max_new_tokens=8):
        """Multi-class screen-state classification.

        Specialized for autonomous_mode.SCREEN_QUERY_PROMPT. Returns the
        class name (PLAYING/PAUSED/DIALOG/MENU/SCREENSAVER) whose
        first-token logit is highest. `max_new_tokens` is ignored (no
        decode loop) but kept for API compatibility.

        Returns:
            (response_text: str, elapsed: float)
        """
        if not self.is_ready:
            return "VLM not ready", 0.0
        if not os.path.exists(image_path):
            return f"Image not found: {image_path}", 0.0

        with self._lock:
            t0 = time.time()
            try:
                vision_out = self._encode_image(image_path)

                # If the caller passed the canonical screen prompt, reuse
                # the cached token IDs. Otherwise build on the fly (rare
                # in production — only autonomous_mode uses query_image)
                # and length-check against the fixed prefill window.
                if (prompt and prompt.startswith("Classify this TV screen")):
                    prompt_ids = self._screen_prompt_ids
                else:
                    prompt_ids = self._build_prompt_ids(prompt or self.AD_PROMPT_TEXT)
                    if len(prompt_ids) > self.PREFILL_LEN:
                        logger.warning(
                            f"query_image prompt is {len(prompt_ids)} tokens > "
                            f"PREFILL_LEN={self.PREFILL_LEN} — shorten the "
                            f"prompt; returning PROMPT_TOO_LONG"
                        )
                        return "PROMPT_TOO_LONG", time.time() - t0

                logits, _ = self._prefill_last_logits(vision_out, prompt_ids)

                # Pick the class with the highest first-token logit (max
                # over no-leading-space and leading-space spellings).
                best_class = None
                best_score = -float('inf')
                scores = {}
                for name, token_ids in self.SCREEN_CLASS_TOKEN_IDS:
                    s = float(max(logits[t] for t in token_ids))
                    scores[name] = s
                    if s > best_score:
                        best_score = s
                        best_class = name

                elapsed = time.time() - t0

                # Format scores compactly for the log; autonomous_mode
                # only consumes the leading class name via `startswith`.
                score_str = " ".join(
                    f"{n}={scores[n]:.2f}" for n, _ in self.SCREEN_CLASS_TOKEN_IDS)
                logger.info(
                    f"VLM(LFM2) query: {best_class} ({score_str}) "
                    f"lat={elapsed:.3f}s"
                )
                return best_class, elapsed

            except Exception as e:
                import traceback
                logger.error(f"VLM query error: {e}\n{traceback.format_exc()}")
                return f"Error: {e}", time.time() - t0

    # ------------------------------------------------------------------
    # Confidence parsing (kept for backwards compat with callers that
    # used the FastVLM response-text path; LFM responses are now
    # structured ("Yes (p=…)" / class names) so these are mostly dead
    # code now, but harmless to keep around).
    # ------------------------------------------------------------------

    def _parse_confidence(self, response):
        r = (response or "").lower()
        for word in ('definitely', 'clearly', 'certainly', 'absolutely',
                     '100%', 'sure', 'obvious', 'without doubt', 'no doubt'):
            if word in r:
                return 0.95
        for word in ('maybe', 'possibly', 'might', 'could be', 'perhaps',
                     'probably', 'likely', 'appears to', 'seems to', 'looks like'):
            if word in r:
                return 0.5
        for word in ('not sure', 'uncertain', 'unclear', 'hard to tell',
                     'difficult to say', 'cannot determine', "can't tell"):
            if word in r:
                return 0.3
        return 0.75

    def _is_ad_response(self, response):
        r = (response or "").lower().strip()
        confidence = self._parse_confidence(response)
        if r.startswith('no') or r == 'n':
            return False, confidence
        if r.startswith('yes') or r == 'y':
            return True, confidence
        return False, 0.3

    # ------------------------------------------------------------------
    # Release
    # ------------------------------------------------------------------

    def release(self):
        """Release NPU sessions and embeddings."""
        import gc
        with self._lock:
            for attr in ('vision', 'post'):
                sess = getattr(self, attr, None)
                if sess is not None:
                    try:
                        if hasattr(sess, 'release'):
                            sess.release()
                    except Exception as e:
                        logger.debug(f"Error releasing {attr}: {e}")
                    setattr(self, attr, None)

            if self.fused:
                for sess in self.fused:
                    try:
                        if sess is not None and hasattr(sess, 'release'):
                            sess.release()
                    except Exception as e:
                        logger.debug(f"Error releasing fused layer: {e}")
                self.fused = None

            self.tokenizer = None
            self.embeds = None
            self._ad_prompt_ids = None
            self._screen_prompt_ids = None
            self.is_ready = False

        gc.collect()
        logger.info("[VLM] LFM2.5-VL model unloaded and resources released")
