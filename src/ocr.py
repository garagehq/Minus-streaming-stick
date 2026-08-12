"""
PaddleOCR module for Minus using RKNN NPU.

Detects text in frames and checks for ad-related keywords.
"""

import os
import re
import time
import numpy as np
import cv2
from pathlib import Path

# RKNNLite (Rockchip NPU SDK) has a side effect at import: it globally
# overwrites `logging._nameToLevel` from the standard
# {'CRITICAL': 50, 'DEBUG': 10, 'ERROR': 40, 'FATAL': 50, 'INFO': 20,
#  'NOTSET': 0, 'WARN': 30, 'WARNING': 30}
# to single-letter shorthands
# {'C': 50, 'D': 10, 'E': 40, 'I': 20, 'W': 30}.
# This breaks any code that later calls `logger.setLevel('WARNING')`
# with a full name string — stdlib logging then rejects it as
# "Unknown level: 'WARNING'".
#
# Observed crash (2026-05-27): the faster-whisper ASR engine swap added
# torch to the indirect dep chain. torch.fx.passes.utils.matcher_utils
# ._init_logger reads PYTORCH_MATCHER_LOGLEVEL=WARNING (set by
# minus.service) and calls setLevel('WARNING') during torch import.
# Before faster-whisper landed, nothing in the import path triggered
# that call, so RKNNLite's mutation was harmless. After: minus crashed
# at every startup with "ValueError: Unknown level: 'WARNING'".
#
# Snapshot the standard table before importing rknnlite and restore it
# afterward. The restoration is idempotent — if a future rknnlite
# release stops mutating, this is a no-op.
import logging as _logging
_logging_levels_before = dict(_logging._nameToLevel)
from rknnlite.api import RKNNLite  # noqa: E402
if 'WARNING' not in _logging._nameToLevel:
    _logging._nameToLevel.clear()
    _logging._nameToLevel.update(_logging_levels_before)
    # Rebuild the reverse mapping (CRITICAL/FATAL and WARN/WARNING
    # alias to the same numeric levels — keep the canonical primary
    # name for each).
    _logging._levelToName.clear()
    for _name, _lvl in _logging_levels_before.items():
        if _name in ('FATAL', 'WARN'):
            continue
        _logging._levelToName[_lvl] = _name
del _logging_levels_before

try:
    import pyclipper
    from shapely.geometry import Polygon
    HAS_POSTPROCESS = True
except ImportError:
    HAS_POSTPROCESS = False
    print("[OCR] Warning: pyclipper/shapely not installed. Install with: pip3 install --break-system-packages pyclipper shapely")


class DBPostProcessor:
    """Post-processor for text detection using DB (Differentiable Binarization)."""

    def __init__(self, thresh=0.3, box_thresh=0.5, max_candidates=1000,
                 unclip_ratio=1.5, min_size=3):
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio
        self.min_size = min_size

    def __call__(self, pred, src_h, src_w):
        if len(pred.shape) == 3:
            pred = pred[0]

        segmentation = pred > self.thresh
        boxes, scores = self.boxes_from_bitmap(pred, segmentation, src_w, src_h)
        return boxes, scores

    def boxes_from_bitmap(self, pred, bitmap, dest_width, dest_height):
        height, width = bitmap.shape
        bitmap_uint8 = (bitmap * 255).astype(np.uint8)
        contours, _ = cv2.findContours(bitmap_uint8, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        num_contours = min(len(contours), self.max_candidates)
        boxes = []
        scores = []

        for i in range(num_contours):
            contour = contours[i]
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size:
                continue

            score = self.box_score_fast(pred, points.reshape(-1, 2))
            if self.box_thresh > score:
                continue

            box = self.unclip(points, self.unclip_ratio)
            if box is None:
                continue

            box, sside = self.get_mini_boxes(box.reshape(-1, 1, 2).astype(np.int32))
            if sside < self.min_size + 2:
                continue

            box[:, 0] = np.clip(box[:, 0] / width * dest_width, 0, dest_width)
            box[:, 1] = np.clip(box[:, 1] / height * dest_height, 0, dest_height)

            boxes.append(box.astype(np.int32))
            scores.append(score)

        return boxes, scores

    def get_mini_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = np.array([points[index_1], points[index_2],
                       points[index_3], points[index_4]])
        return box, min(bounding_box[1])

    def box_score_fast(self, bitmap, box):
        h, w = bitmap.shape
        box = box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int32), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int32), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int32), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int32), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    def unclip(self, box, unclip_ratio):
        try:
            poly = Polygon(box)
            distance = poly.area * unclip_ratio / poly.length
            offset = pyclipper.PyclipperOffset()
            offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
            expanded = offset.Execute(distance)
            if len(expanded) == 0:
                return None
            return np.array(expanded[0])
        except Exception:
            return None


class CTCLabelDecode:
    """CTC decoder for text recognition."""

    def __init__(self, character_dict_path):
        self.character = ['blank']
        with open(character_dict_path, 'r', encoding='utf-8') as f:
            for line in f:
                char = line.strip('\n')
                if char:
                    self.character.append(char)
        self.character.append(' ')

    def __call__(self, preds):
        if len(preds.shape) == 3:
            preds = preds[0]

        preds_idx = preds.argmax(axis=1)
        preds_prob = preds.max(axis=1)

        result = []
        prev_idx = -1
        conf_list = []

        for i, idx in enumerate(preds_idx):
            if idx != 0 and idx != prev_idx:
                if idx < len(self.character):
                    result.append(self.character[idx])
                    conf_list.append(preds_prob[i])
            prev_idx = idx

        text = ''.join(result)
        confidence = np.mean(conf_list) if conf_list else 0.0

        return text, float(confidence)


class PaddleOCR:
    """PaddleOCR using RKNN models for NPU acceleration."""

    # Ad-related keywords to detect (must be distinct/clear ad indicators).
    # THE single source of truth — OCRProcess (src/ocr_worker.py) delegates
    # here. These lists and the worker's used to be separate copies and
    # drifted BOTH ways (worker missing 'skip'/'sponsor'/Spanish; this list
    # missing 'skip in'/'learn more'/bare 'ad' that minus.py's strong/weak
    # keyword-name sets reference). Merged Aug 2026 — keep it that way.
    AD_KEYWORDS_EXACT = [
        'skip ad', 'skip ads', 'skipad', 'skipads', 'skip in',
        'video will play after ad',
        'sponsored', 'advertisement', 'ad break',
        'shop now', 'buy now', 'learn more',
        'download now', 'install now', 'get the app', 'free download',
        'limited time', 'offer ends', 'dont miss', "don't miss",
        'order now', 'sign up', 'subscribe now',
        'promoted',  # Twitter/social media promoted ads
        'visit advertiser', 'visitadvertiser',  # YouTube pre-roll CTA
        # Note: 'promo' removed - too broad, matches 'Promote' button
        # Spanish keywords
        'patrocinado', 'anuncio', 'publicidad',
        'comprar ahora', 'visita al anunciante',
        'contenido patrocinado',
    ]
    # Keywords that need word boundary matching (avoid matching inside words)
    AD_KEYWORDS_WORD = [
        'ad', 'ads',  # word-boundary so 'loading'/'adobe' cannot match
        'skip', 'sponsor',
        # Spanish word-boundary keywords
        'patroci',  # Catches patrocinado, patrocinador, etc.
    ]

    # Phrases that should NOT trigger ad detection (false positives)
    # These override keyword matches when the full phrase is detected
    AD_EXCLUSIONS = [
        'skip recap', 'skiprecap',  # Netflix "Skip Recap" button
        'skip intro', 'skipintro',  # Streaming "Skip Intro" button
        'skip credits', 'skip opening',  # end-credits / anime opening buttons
        'saltar intro', 'saltarintro',  # Spanish "Skip Intro"
        # 'Add ...' UI phrases that OCR can mangle toward bare 'ad'
        'add to', 'add it', 'already added', 'address', 'add new',
        'additionally', 'adaptive', 'advanced', 'advantage',
        # Minus overlay messages (prevent self-triggering)
        'ad skipping enabled', 'ad skipping', 'adskipping',
        'ad detection active', 'ad detection actiue',  # Status overlay (OCR misreads V as U)
        'addetection', 'detection active', 'detection actiue',
        'vlm ready', 'vlmready', 'ulmready',  # VLM status (OCR misreads V as U)
        'ocr ready', 'ocrready',  # OCR status
        'ready',  # Generic status word (too common to be an ad indicator)
    ]
    # Fuzzy match for "Skip Intro" with OCR character swaps. Covers
    # "Skip Intro", "Sk1p Intro", "Skip 1ntro", "Sk1p 1ntro", "Sk1p1ntro",
    # etc. Defined as a compiled regex rather than enumerating every
    # permutation in AD_EXCLUSIONS.
    SKIP_INTRO_FUZZY_RE = re.compile(r's[kK][i1lI]p\s*[i1lI]ntro', re.IGNORECASE)

    def __init__(self, det_model_path, rec_model_path, dict_path,
                 cls_model_path=None):
        self.det_model_path = det_model_path
        self.rec_model_path = rec_model_path
        self.cls_model_path = cls_model_path
        self.dict_path = dict_path

        self.det_rknn = None
        self.rec_rknn = None
        self.cls_rknn = None

        self.det_input_h = 960
        self.det_input_w = 960
        self.rec_input_h = 48
        self.rec_input_w = 320

        self.db_postprocess = DBPostProcessor() if HAS_POSTPROCESS else None
        self.ctc_decode = None
        self.initialized = False

    def load_models(self):
        """Load all RKNN models. Returns True on success, False on failure."""
        if not HAS_POSTPROCESS:
            print("[OCR] Cannot load models without pyclipper/shapely")
            return False

        try:
            print("[OCR] Loading PaddleOCR models...")

            # Load detection model
            print(f"[OCR]   Loading detection model...")
            self.det_rknn = RKNNLite()
            ret = self.det_rknn.load_rknn(self.det_model_path)
            if ret != 0:
                print(f"[OCR]   Failed to load detection model: {ret}")
                return False
            ret = self.det_rknn.init_runtime()
            if ret != 0:
                print(f"[OCR]   Failed to init detection runtime: {ret}")
                return False

            # Load recognition model
            print(f"[OCR]   Loading recognition model...")
            self.rec_rknn = RKNNLite()
            ret = self.rec_rknn.load_rknn(self.rec_model_path)
            if ret != 0:
                print(f"[OCR]   Failed to load recognition model: {ret}")
                return False
            ret = self.rec_rknn.init_runtime()
            if ret != 0:
                print(f"[OCR]   Failed to init recognition runtime: {ret}")
                return False

            # Initialize CTC decoder
            if os.path.exists(self.dict_path):
                self.ctc_decode = CTCLabelDecode(self.dict_path)
                print(f"[OCR]   Dictionary loaded: {len(self.ctc_decode.character)} characters")
            else:
                print(f"[OCR]   Dictionary not found: {self.dict_path}")
                return False

            self.initialized = True
            print("[OCR] Models loaded successfully")
            return True

        except Exception as e:
            print(f"[OCR] Failed to load models: {e}")
            import traceback
            traceback.print_exc()
            return False

    def preprocess_det(self, img):
        """Preprocess image for detection."""
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w = img.shape[:2]
        img_resized = cv2.resize(img, (self.det_input_w, self.det_input_h))
        img_input = np.expand_dims(img_resized, 0).astype(np.uint8)
        return img_input, h, w

    def preprocess_rec(self, img):
        """Preprocess cropped text region for recognition."""
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w = img.shape[:2]
        ratio = self.rec_input_h / h
        new_w = int(w * ratio)
        new_w = min(new_w, self.rec_input_w)

        img_resized = cv2.resize(img, (new_w, self.rec_input_h))

        if new_w < self.rec_input_w:
            pad_w = self.rec_input_w - new_w
            img_resized = np.pad(img_resized,
                                ((0, 0), (0, pad_w), (0, 0)),
                                mode='constant', constant_values=0)

        img_input = np.expand_dims(img_resized, 0).astype(np.uint8)
        return img_input

    def crop_text_region(self, img, box):
        """Crop text region from image using perspective transform."""
        box = np.array(box).astype(np.float32)

        width = int(max(
            np.linalg.norm(box[0] - box[1]),
            np.linalg.norm(box[2] - box[3])
        ))
        height = int(max(
            np.linalg.norm(box[0] - box[3]),
            np.linalg.norm(box[1] - box[2])
        ))

        if width < 3 or height < 3:
            return None

        dst = np.array([
            [0, 0],
            [width, 0],
            [width, height],
            [0, height]
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(box, dst)
        cropped = cv2.warpPerspective(img, M, (width, height))
        return cropped

    def detect(self, img):
        """Run text detection."""
        img_input, src_h, src_w = self.preprocess_det(img)

        start = time.time()
        outputs = self.det_rknn.inference(inputs=[img_input])
        det_time = (time.time() - start) * 1000

        # Copy output to release RKNN internal buffer reference
        pred = outputs[0].copy()
        del outputs  # Explicitly release RKNN output list

        if len(pred.shape) == 4:
            pred = pred[0, 0]
        elif len(pred.shape) == 3:
            pred = pred[0]

        boxes, scores = self.db_postprocess(pred, src_h, src_w)
        return boxes, scores, det_time

    def recognize(self, img_crop):
        """Run text recognition on cropped region."""
        img_input = self.preprocess_rec(img_crop)

        start = time.time()
        outputs = self.rec_rknn.inference(inputs=[img_input])
        rec_time = (time.time() - start) * 1000

        # Copy output to release RKNN internal buffer reference
        pred = outputs[0].copy()
        del outputs  # Explicitly release RKNN output list

        text, confidence = self.ctc_decode(pred)
        return text, confidence, rec_time

    def ocr(self, img):
        """
        Run full OCR pipeline on image.

        Returns:
            List of dicts with 'text', 'confidence', 'box'
        """
        if not self.initialized:
            return []

        results = []

        # Detection
        boxes, det_scores, det_time = self.detect(img)

        # Recognition for each box
        for box in boxes:
            cropped = self.crop_text_region(img, box)
            if cropped is None:
                continue

            text, confidence, rec_time = self.recognize(cropped)

            if text.strip():
                results.append({
                    'text': text,
                    'confidence': confidence,
                    'box': box.tolist()
                })

        return results

    # Patterns that indicate terminal/development content
    TERMINAL_INDICATORS = [
        # Shell/Terminal patterns
        r'\$\s*$',           # Shell prompt
        r'radxa@',           # Username prompt
        r'/home/',           # Unix paths
        r'\.py\b',           # Python files
        r'\.log\b',          # Log files
        r'\[I\]|\[W\]|\[E\]', # Log level indicators
        r'Exit code',        # Command exit
        r'ctrl\+',           # Keyboard shortcuts
        r'minus',            # Our own script
        r'OCR #\d+',         # Our log output
        r'^\d{4}-\d{2}-\d{2}', # Timestamps
        r'Error:|Warning:',  # Error messages
        r'python3?\s',       # Python command
        r'nohup|grep|cat|tail|cd\s', # Common commands

        # Claude Code / AI Assistant patterns (OCR may misread)
        r'dangerously.*skip.*perm',  # --dangerously-skip-permissions flag
        r'angerously.*skip.*perm',   # OCR misread without 'd'
        r'claude.*code',     # Claude Code UI
        r'anthropic',        # Company name
        r'opus|sonnet|haiku', # Model names
        r'welco[nm]e\s*back',  # Welcome back (Claude greeting)

        # Keyword list patterns (our own keywords displayed on screen)
        r"'skip.*sponsor",   # Keyword list showing
        r'skip.*promo.*sponsor', # Multiple keywords together
        r"skip\s*ad.*skip\s*ads", # Multiple ad keywords listed
        r'AD_KEYWORDS',      # Variable name
        r'TERMINAL_INDICATORS', # This variable name

        # Code patterns
        r'def\s+\w+\s*\(',   # Python function definitions
        r'class\s+\w+',      # Class definitions
        r'import\s+\w+',     # Import statements
        r'pip3?\s+install',  # pip commands
        r'sudo\s+',          # sudo commands
        r'git\s+(status|commit|push|pull)', # git commands
    ]

    @classmethod
    def is_terminal_content(cls, all_texts):
        """
        Check if the detected text appears to be terminal/development content.

        Returns:
            True if terminal content is detected, False otherwise
        """
        import re

        terminal_matches = 0
        total_texts = len(all_texts)

        if total_texts == 0:
            return False

        combined_text = ' '.join(all_texts)
        combined_lower = combined_text.lower()

        for pattern in cls.TERMINAL_INDICATORS:
            if re.search(pattern, combined_text, re.IGNORECASE):
                terminal_matches += 1

        # If we match 3+ terminal indicators, it's likely terminal content
        if terminal_matches >= 3:
            return True

        # Also check for high density of code-like characters
        code_chars = sum(1 for c in combined_text if c in '{}[]();:=></')
        if code_chars > len(combined_text) * 0.05 and total_texts > 20:
            return True

        # Check if multiple ad keywords appear together (likely showing our code)
        # Note: Don't double-count related keywords (e.g., "sponsored" contains "sponsor")
        matched_keywords = set()
        for kw in cls.AD_KEYWORDS_EXACT + cls.AD_KEYWORDS_WORD:
            if kw in combined_lower:
                # Skip if a longer keyword already matched this same text
                already_matched = any(kw in mk or mk in kw for mk in matched_keywords if mk != kw)
                if not already_matched:
                    matched_keywords.add(kw)
        # If 4+ distinct keywords visible, it's probably our code or documentation
        if len(matched_keywords) >= 4:
            return True

        # Check for Python-like syntax (OCR may mangle it)
        python_patterns = [
            r"'\w+',\s*['\"]",   # 'word', ' or 'word', "
            r'exit\s*code',      # exit code (lenient)
            r'step.*ed',         # stopped, stepped
            r'\[\s*\]',          # [] brackets
            r'renovedI|Added.*Lin', # OCR misreads of "removed" and "Added lines"
        ]
        python_matches = sum(1 for p in python_patterns
                            if re.search(p, combined_text, re.IGNORECASE))
        if python_matches >= 2:
            return True

        return False

    @classmethod
    def check_ad_keywords(cls, ocr_results):
        """
        Check OCR results for ad-related keywords.

        THE single source of truth for keyword matching. OCRProcess
        (src/ocr_worker.py) delegates here — production hit repeated
        "keyword-pattern drift" bugs while it kept its own copy (see
        CLAUDE.md, OCR Worker Keyword-Pattern Drift), so do NOT
        reintroduce a second implementation.

        Returns:
            Tuple of (found_ad, matched_keywords, all_texts, is_terminal)
        """
        import re
        matched = []
        all_texts = []

        for result in ocr_results:
            text = result['text']
            all_texts.append(text)

            text_lower = text.lower()
            text_clean = ''.join(c for c in text_lower if c.isalnum())

            # Check exclusions FIRST - skip ALL ad detection for excluded phrases
            # This prevents Minus overlay text from triggering false positives
            is_excluded = (
                any(excl in text_lower or excl.replace(' ', '') in text_clean
                    for excl in cls.AD_EXCLUSIONS)
                or cls.SKIP_INTRO_FUZZY_RE.search(text_lower) is not None
            )
            if is_excluded:
                continue  # Skip all pattern matching for this text element

            # Check exact phrase keywords (can appear anywhere)
            for keyword in cls.AD_KEYWORDS_EXACT:
                keyword_clean = ''.join(c for c in keyword if c.isalnum())
                if keyword in text_lower or keyword_clean in text_clean:
                    matched.append((keyword, text))
                    break

            # Check word-boundary keywords (must be whole word)
            for keyword in cls.AD_KEYWORDS_WORD:
                # Use word boundary regex
                pattern = r'\b' + re.escape(keyword) + r'\b'
                if re.search(pattern, text_lower):
                    matched.append((keyword, text))
                    break

            # Fuzzy matches for common OCR misreads of "Skip Ad"
            if 'skipad' in text_clean or 'skipads' in text_clean:
                if ('skipad', text) not in matched and ('skipads', text) not in matched:
                    matched.append(('skip ad (fuzzy)', text))
            # Common OCR misreads
            if 'spad' in text_clean and len(text_clean) < 10:  # Short text with spad
                matched.append(('skip ad (fuzzy-spad)', text))
            if 'foad' in text_clean and len(text_clean) < 10:  # Short text with foad
                matched.append(('skip ad (fuzzy-foad)', text))

            # Fuzzy "Sponsored" — OCR drops/mangles interior letters on
            # stylized ad chyrons ("Sponoed", "Sponsoed", "Sponsred").
            # Guards: (?<!re) rejects (cor)responded, (?!ged) rejects sponged.
            if ('sponsored' not in text_clean
                    and re.search(r'(?<!re)spon(?!ged)[a-z]{0,3}ed', text_clean)):
                matched.append(('sponsored (fuzzy)', text))

            # Fuzzy matches for "Shop now" - frequently misread
            if 'shopnow' in text_clean or 'shpnow' in text_clean:
                matched.append(('shop now (fuzzy)', text))
            # "Shan ngw", "Shon ngw", "Shap now" etc.
            if re.search(r'sh[ao][np]\s*n[gwo]w', text_lower):
                matched.append(('shop now (fuzzy-shan)', text))
            # "go to [site].io" or "go to [site].com" indicates ad CTA
            if re.search(r'go\s*to\s+\w+\.(io|com|net|org)', text_lower):
                matched.append(('go to site (ad CTA)', text))

            # "Ad 1 of 2", "Ad2of2", "ad 2 of 3" - video ad progress indicator
            if re.search(r'ad\s*\d+\s*of\s*\d+', text_lower) or re.search(r'ad\d+of\d+', text_clean):
                matched.append(('ad X of Y', text))

            # "Ad 10", "Ad 5", "Ad 30" - Netflix/streaming ad countdown timer
            # Matches "Ad" followed by a number (seconds remaining)
            if re.search(r'^ad\s*\d+$', text_lower.strip()):
                matched.append(('ad countdown', text))

            # "0:30 | Ad", "Ad | 0:30", "Ad 0:30", "Ad0:30", "Ad1:20", "Ado:30" - ad with timestamp
            # OCR common misreads:
            #   0 ↔ o ↔ O (zero vs letter o)
            #   1 ↔ l ↔ I ↔ i (one vs lowercase L vs uppercase i)
            #   : ↔ ; ↔ . (colon vs semicolon vs period)
            # Match "ad" at word boundary OR "ad" followed by digit-like char + separator
            # Digit-like: 0-9, o, O, l, I, i (common OCR confusions)
            # Separator: : ; . (colon misreads)
            has_ad = re.search(r'\bad\b', text_lower) or re.search(r'ad[0-9oOlIi][:;.]', text_lower)
            has_timestamp = re.search(r'[0-9oOlIi][:;.][0-9oOlIi][0-9oOlIi]', text_lower)
            if has_ad and has_timestamp:
                matched.append(('ad with timestamp', text))

        # Cross-element check: "Ad" in one element + timestamp in another (Hulu-style)
        # Only if we haven't already matched and there are few text elements (not noisy OCR)
        if not matched and len(all_texts) <= 5:
            combined = ' '.join(all_texts).lower()
            combined_clean = ''.join(c for c in combined if c.isalnum())
            # Check exclusions for combined text too
            is_combined_excluded = (
                any(excl in combined or excl.replace(' ', '') in combined_clean
                    for excl in cls.AD_EXCLUSIONS)
                or cls.SKIP_INTRO_FUZZY_RE.search(combined) is not None
            )
            if not is_combined_excluded:
                has_ad_word = re.search(r'\bad\b', combined) or re.search(r'ad[0-9oOlIi][:;.]', combined)
                has_timestamp = re.search(r'[0-9oOlIi][:;.][0-9oOlIi][0-9oOlIi]', combined)
                if has_ad_word and has_timestamp:
                    matched.append(('ad with timestamp (cross-element)', combined[:50]))

        # Check if this appears to be terminal content
        is_terminal = cls.is_terminal_content(all_texts)

        return len(matched) > 0, matched, all_texts, is_terminal

    def release(self):
        """Release all models."""
        if self.det_rknn:
            self.det_rknn.release()
        if self.rec_rknn:
            self.rec_rknn.release()
        if self.cls_rknn:
            self.cls_rknn.release()
        self.initialized = False
