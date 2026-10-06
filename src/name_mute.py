"""
Name mute: briefly cut the audio whenever a target name is spoken or
captioned.

Two detectors feed one scheduler:

  * ASR (Moonshine, word timestamps) hears the name in the audio tap.
  * OCR reads it in on-screen captions.

Both tap the signal before playback. When audio and video are run through
the same delay line (MINUS_AV_DELAY_S), the detectors get a head start on
what the TV is about to play, so the mute can land on the word itself
instead of after it.

Time bookkeeping: every detection is converted to a *capture* time (the
monotonic clock when that audio/frame entered the pipeline). The sample
captured at time c is heard at c + playback_delay, where playback_delay is
the live fill of the audio sync queue. The scheduler mutes
[c_start + delay - pad_before, c_end + delay + pad_after].
"""

import collections
import difflib
import logging
import os
import re
import threading
import time

logger = logging.getLogger('Minus.NameMute')


def _norm(text: str) -> str:
    text = text.lower().replace("’", "'")
    text = re.sub(r"[^a-z' ]+", ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


# Spellings Moonshine and OCR actually produce for "LeBron" / "LeBron James".
# Measured on NBC/ESPN commentary: "LeBron", "LeBron's", "Le Bron",
# "the Bron James", "Lebrun". "The Bronx" alone is NOT matched (it is a real
# place); "the Bronx James" is.
_FIRST_NAME_RE = re.compile(r"\b(?:le ?bron+|lebrun|la ?bron)(?:'?s)?\b")
_FULL_NAME_RE = re.compile(
    r"\b(?:(?:le|the|la) ?bron(?:x)?|bron|lebrun) james(?:'?s)?\b")
# "King" on its own counts (user request), and so does "King James"; the
# plural "Kings" (Sacramento) does not.
_KING_JAMES_RE = re.compile(r"\bking(?: james)?\b(?!s)")
_RAW_NAME_RE = re.compile(r"le ?bron+|lebrun", re.I)


def _is_graphic_name(line: str) -> bool:
    """Captions write "LeBron"/"lebron"; a name with 4+ capitals in a row
    ("LEBRON 2011 Finals", "LeBRON JAMES") is an on-screen graphic."""
    return any(re.search(r"[A-Z]{4}", m.group(0)) for m in _RAW_NAME_RE.finditer(line))
_SURNAME_RE = re.compile(r"\bjames(?:'?s)?\b")

# Word-level forms (single token as Moonshine emits it).
_WORD_FIRST_RE = re.compile(r"^(?:le ?bron+|lebrun|la ?bron)(?:'?s)?$")
# Only counts when followed by "James". Besides "Bron"/"Bronx" this takes
# the ASR's garbled first names ("amron james", "thebron james", "lebra
# james"), but not real first names ending in -ron.
_WORD_BRON_RE = re.compile(
    r"^(?!(?:aaron|cameron|byron|myron|ron|baron|darron|kendron)$)"
    r"\w{0,4}(?:bron|brun|bra|ron)x?$")
_WORD_JAMES_RE = re.compile(r"^james(?:'?s)?$")
_WORD_KING_RE = re.compile(r"^king(?:'s)?$")


class NameMatcher:
    """Finds a target name in free text and in timed word lists.

    include_surname: also match a bare "James". Off by default: in LeBron
    footage it is usually him, but it also matches every other James.
    """

    def __init__(self, include_surname: bool = False):
        self.include_surname = include_surname

    def find(self, text: str) -> list:
        """Return the matched name phrases in `text` (empty if none)."""
        t = _norm(text)
        if not t:
            return []
        hits = []
        for rx in (_FULL_NAME_RE, _FIRST_NAME_RE, _KING_JAMES_RE):
            hits.extend(m.group(0) for m in rx.finditer(t))
        if self.include_surname and not hits:
            hits.extend(m.group(0) for m in _SURNAME_RE.finditer(t))
        return hits

    def find_word_spans(self, words) -> list:
        """Return (start, end) spans of name words.

        words: iterable of (word, start_s, end_s). A "LeBron James" pair
        becomes one span; a lone "Bron"/"Bronx" counts only when followed
        by "James" (so "the Bronx" stays unmuted).
        """
        toks = [(_norm(w), s, e) for w, s, e in words]
        spans = []
        i = 0
        while i < len(toks):
            w, s, e = toks[i]
            nxt = toks[i + 1] if i + 1 < len(toks) else None
            nxt_is_james = nxt is not None and _WORD_JAMES_RE.match(nxt[0])
            if _WORD_FIRST_RE.match(w) or (_WORD_BRON_RE.match(w) and nxt_is_james):
                if nxt_is_james:
                    spans.append((s, nxt[2]))
                    i += 2
                    continue
                spans.append((s, e))
            elif _WORD_KING_RE.match(w):
                if nxt_is_james:
                    spans.append((s, nxt[2]))
                    i += 2
                    continue
                spans.append((s, e))
            elif self.include_surname and _WORD_JAMES_RE.match(w):
                spans.append((s, e))
            i += 1
        return spans


class NameMuteScheduler:
    """Mutes the audio over scheduled playback-time windows.

    `audio` must provide set_name_mute(bool). `delay_fn` returns the current
    capture-to-speaker delay in seconds.
    """

    # Default pads (used for ASR word spans), from live Google TV runs scored
    # against YouTube caption tracks (tests/name_mute_live_analyze.py).
    # Moonshine word starts land after the caption word by a median of
    # 0.22-0.40s depending on the broadcast (p90 0.38-0.64s).
    PAD_BEFORE_S = float(os.environ.get('MINUS_NAME_MUTE_PAD_BEFORE', '0.4'))
    PAD_AFTER_S = float(os.environ.get('MINUS_NAME_MUTE_PAD_AFTER', '0.35'))
    # A detection that arrives too late to cover the word still mutes this
    # long, so a missed head start still cuts the tail of the name.
    MIN_LATE_MUTE_S = float(os.environ.get('MINUS_NAME_MUTE_MIN_LATE', '0.8'))
    TICK_S = 0.01

    def __init__(self, audio, delay_fn):
        self._audio = audio
        self._delay_fn = delay_fn
        self._lock = threading.Lock()
        self._windows = []          # [(play_start, play_end, source, label)]
        self._muted = False
        self._stop = threading.Event()
        self._thread = None
        self.events = []            # recent (wall_time, source, label, on_time)
        self.mute_count = 0
        self.late_count = 0
        self.duplicate_count = 0
        # Every detection, for offline timing analysis (GET /api/name-mute/log).
        # Times are time.monotonic(): capture_* is when the audio/frame entered
        # the pipeline, play_* the scheduled mute (None if skipped).
        self.detection_log = collections.deque(maxlen=5000)

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name='NameMute', daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        self._set(False)

    def schedule(self, capture_start: float, capture_end: float,
                 source: str, label: str, meta: dict = None,
                 pad_before: float = None, pad_after: float = None,
                 replace_sources=()):
        """Mute the playback of audio captured in [capture_start, capture_end].

        replace_sources: pending (not yet started) windows from these sources
        that overlap this one are dropped first, so a precise detection
        replaces a coarse one for the same mention instead of widening it.
        """
        delay = max(0.0, float(self._delay_fn() or 0.0))
        pb = self.PAD_BEFORE_S if pad_before is None else pad_before
        pa = self.PAD_AFTER_S if pad_after is None else pad_after
        start = capture_start + delay - pb
        end = capture_end + delay + pa
        now = time.monotonic()
        if replace_sources:
            with self._lock:
                self._windows = [
                    w for w in self._windows
                    if not (w[2] in replace_sources and w[0] > now
                            and w[0] < end + 0.8 and w[1] > start - 0.8)]
        with self._lock:
            covered = any(w[0] <= start and end <= w[1] for w in self._windows)
        entry = {'source': source, 'label': label, 'detected': now,
                 'capture_start': capture_start, 'capture_end': capture_end,
                 'delay': delay, 'play_start': None, 'play_end': None,
                 'duplicate': covered, 'meta': meta or {}}
        self.detection_log.append(entry)
        if covered:
            # Already muting this stretch (e.g. ASR confirming a caption hit,
            # or two OCR misreads of the same caption line).
            self.duplicate_count += 1
            logger.debug(f"[NameMute] {source}: '{label}' already covered")
            return
        on_time = start >= now
        if not on_time:
            self.late_count += 1
            start = now
            end = max(end, now + self.MIN_LATE_MUTE_S)
        entry['play_start'], entry['play_end'] = start, end
        with self._lock:
            self._windows.append((start, end, source, label))
            self.mute_count += 1
            self.events.append((time.time(), source, label, on_time))
            del self.events[:-50]
        logger.info(f"[NameMute] {source}: '{label}' -> mute in {start - now:+.2f}s "
                    f"for {end - start:.2f}s (delay {delay:.2f}s, "
                    f"{'on time' if on_time else 'LATE'})")

    def _run(self):
        while not self._stop.is_set():
            now = time.monotonic()
            with self._lock:
                self._windows = [w for w in self._windows if w[1] > now]
                want = any(w[0] <= now < w[1] for w in self._windows)
            if want != self._muted:
                self._set(want)
            self._stop.wait(self.TICK_S)

    def _set(self, muted: bool):
        try:
            self._audio.set_name_mute(muted)
            self._muted = muted
        except Exception as e:
            logger.warning(f"[NameMute] set_name_mute failed: {e}")

    def get_status(self) -> dict:
        with self._lock:
            pending = len(self._windows)
            recent = [
                {'time': t, 'source': s, 'label': l, 'on_time': ok}
                for t, s, l, ok in self.events[-10:]
            ]
        return {
            'muted': self._muted,
            'pending_windows': pending,
            'mute_count': self.mute_count,
            'late_count': self.late_count,
            'duplicate_count': self.duplicate_count,
            'delay_s': round(float(self._delay_fn() or 0.0), 3),
            'recent': recent,
        }


class NameMuteController:
    """Turns ASR words and caption OCR text into scheduled mutes.

    De-duplicates the repeats both detectors produce: ASR windows overlap,
    so one spoken name is heard by 2-3 consecutive windows; a caption line
    stays on screen (and grows word by word) across many OCR frames.
    """

    # Two ASR hits whose capture times are this close are the same mention.
    ASR_DEDUP_S = 0.6
    # Word timings shorter than this are Moonshine repetition hallucinations
    # ("LeBron. LeBron. LeBron." with zero-length words).
    ASR_MIN_WORD_S = 0.08
    TRUNCATED_WORD_EXTRA_S = 0.4
    # ASR window length; set from ASRManager.WINDOW_SECONDS by the owner.
    window_s = 2.5
    # A caption mention is remembered this long (lines scroll up and stay
    # visible for several seconds).
    CAPTION_MEMORY_S = 10.0
    # Caption text appears a median 0.66s after the word starts (live, Google
    # TV YouTube auto-captions); mute this much around the frame that first
    # shows it (capture time). Tuned with tests/name_mute_live_sim.py.
    CAPTION_BEFORE_S = 0.7
    CAPTION_AFTER_S = 0.7
    # A line that ends on the first name ("... and LeBron") usually continues
    # with "James" on the next caption; the plain window ends before it
    # (2 of 23 live mentions were cut short that way with ASR missing).
    CAPTION_LINE_END_EXTRA_S = 0.6
    # OCR misreads one caption line differently frame to frame ("they oking
    # for LeBron" / "they loking for LeBron"); text before the name this
    # similar to a recent one is the same mention.
    CAPTION_FUZZY_RATIO = 0.6

    def __init__(self, scheduler: NameMuteScheduler, matcher: NameMatcher = None):
        self.scheduler = scheduler
        self.matcher = matcher or NameMatcher()
        self.enabled = True
        self.use_asr = True
        self.use_captions = True
        self._recent_asr = []        # capture-time centers of recent ASR hits
        self._recent_captions = {}   # caption key -> last seen (monotonic)
        self.asr_hits = 0
        self.caption_hits = 0
        self.last_asr_text = ''
        self.last_caption_text = ''

    def on_asr_words(self, transcript: str, words, window_start: float):
        if not (self.enabled and self.use_asr):
            return
        spans = self.matcher.find_word_spans(words) if words else []
        if not spans and self.matcher.find(transcript):
            # No usable word timings: mute the whole window.
            spans = [(0.0, max((e for _, _, e in words), default=2.5))]
        if not spans:
            return
        self.last_asr_text = transcript
        now = time.monotonic()
        self._recent_asr = [c for c in self._recent_asr if now - c < 15.0]
        for s, e in spans:
            if e - s < self.ASR_MIN_WORD_S:
                continue
            if s <= 0.05:
                # Word was already under way when this window began.
                s -= self.TRUNCATED_WORD_EXTRA_S
            if e >= self.window_s - 0.05:
                # Word runs past the end of this window; its true end is
                # unknown. Later windows that see it whole are de-duplicated
                # away, so cover the likely remainder here.
                e += self.TRUNCATED_WORD_EXTRA_S
            c0, c1 = window_start + s, window_start + e
            center = (c0 + c1) / 2
            if any(abs(center - c) < self.ASR_DEDUP_S for c in self._recent_asr):
                continue
            self._recent_asr.append(center)
            self.asr_hits += 1
            self.scheduler.schedule(c0, c1, 'asr', transcript.strip()[:80],
                                    {'word_start': round(s, 3), 'word_end': round(e, 3)},
                                    replace_sources=('caption',))

    def on_caption_results(self, results, capture_time: float, frame_shape=None):
        """OCR results (dicts with 'text' and 'box') for one frame."""
        if not results:
            return
        h, w = (frame_shape[0], frame_shape[1]) if frame_shape is not None else (None, None)
        lines = []
        for r in results:
            meta = {}
            box = r.get('box') if isinstance(r, dict) else None
            if box and h and w:
                xs = [p[0] for p in box]
                ys = [p[1] for p in box]
                meta = {'x': round(sum(xs) / len(xs) / w, 3),
                        'y': round(sum(ys) / len(ys) / h, 3),
                        'h': round((max(ys) - min(ys)) / h, 3)}
            lines.append((r.get('text', '') if isinstance(r, dict) else str(r), meta))
        self._on_caption_lines(lines, capture_time)

    def on_caption_texts(self, texts, capture_time: float):
        self._on_caption_lines([(t, {}) for t in (texts or [])], capture_time)

    def _on_caption_lines(self, lines, capture_time: float):
        if not (self.enabled and self.use_captions) or not lines:
            return
        now = time.monotonic()
        self._recent_captions = {k: v for k, v in self._recent_captions.items()
                                 if now - v[0] < self.CAPTION_MEMORY_S}
        for line, meta in lines:
            norm = _norm(line)
            if not norm:
                continue
            if line.upper() == line and any(c.isalpha() for c in line):
                continue   # all-caps: an on-screen graphic, not a caption
            for m in (_FULL_NAME_RE, _FIRST_NAME_RE, _KING_JAMES_RE):
                for hit in m.finditer(norm):
                    if _is_graphic_name(line):
                        continue
                    # Key: the text before this mention plus its first word.
                    # Stable while the caption grows to the right or scrolls
                    # up, and the same whether the full-name or first-name
                    # pattern matched it.
                    key = (norm[:hit.start()] + hit.group(0).split(' ')[0])[-40:]
                    ctx = norm[:hit.start()].strip()[-20:]
                    if self._seen_caption(key, ctx, now):
                        continue
                    self.caption_hits += 1
                    self.last_caption_text = line
                    pad_after = self.CAPTION_AFTER_S
                    if (hit.end() >= len(norm.rstrip())
                            and 'james' not in hit.group(0)):
                        pad_after += self.CAPTION_LINE_END_EXTRA_S
                    self.scheduler.schedule(capture_time, capture_time,
                                            'caption', line.strip()[:80], meta,
                                            pad_before=self.CAPTION_BEFORE_S,
                                            pad_after=pad_after)

    def _seen_caption(self, key: str, ctx: str, now: float) -> bool:
        """Has this caption mention been seen recently? Records it either way.

        Exact key, or (when there is enough text before the name to compare)
        a fuzzy match on that text, which absorbs OCR misreads of one line.
        """
        seen = key in self._recent_captions
        if not seen and len(ctx) >= 4:
            seen = any(len(c) >= 4 and difflib.SequenceMatcher(None, ctx, c).ratio()
                       >= self.CAPTION_FUZZY_RATIO
                       for _, c in self._recent_captions.values())
        self._recent_captions[key] = (now, ctx)
        return seen

    def get_status(self) -> dict:
        st = self.scheduler.get_status()
        st.update({
            'enabled': self.enabled,
            'use_asr': self.use_asr,
            'use_captions': self.use_captions,
            'include_surname': self.matcher.include_surname,
            'asr_hits': self.asr_hits,
            'caption_hits': self.caption_hits,
            'last_asr_text': self.last_asr_text[:200],
            'last_caption_text': self.last_caption_text[:200],
        })
        return st
