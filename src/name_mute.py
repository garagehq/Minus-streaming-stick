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
_KING_JAMES_RE = re.compile(r"\bking james\b")
_SURNAME_RE = re.compile(r"\bjames(?:'?s)?\b")

# Word-level forms (single token as Moonshine emits it).
_WORD_FIRST_RE = re.compile(r"^(?:le ?bron+|lebrun|la ?bron)(?:'?s)?$")
_WORD_BRON_RE = re.compile(r"^(?:bron|bronx)$")
_WORD_JAMES_RE = re.compile(r"^james(?:'?s)?$")


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
            elif w == 'king' and nxt_is_james:
                spans.append((s, nxt[2]))
                i += 2
                continue
            elif self.include_surname and _WORD_JAMES_RE.match(w):
                spans.append((s, e))
            i += 1
        return spans


class NameMuteScheduler:
    """Mutes the audio over scheduled playback-time windows.

    `audio` must provide set_name_mute(bool). `delay_fn` returns the current
    capture-to-speaker delay in seconds.
    """

    # Moonshine word starts land a median 0.37s (up to ~0.6s) after the
    # caption-track word times on NBA commentary, so lead in generously.
    PAD_BEFORE_S = float(os.environ.get('MINUS_NAME_MUTE_PAD_BEFORE', '0.6'))
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
                 source: str, label: str):
        """Mute the playback of audio captured in [capture_start, capture_end]."""
        delay = max(0.0, float(self._delay_fn() or 0.0))
        start = capture_start + delay - self.PAD_BEFORE_S
        end = capture_end + delay + self.PAD_AFTER_S
        now = time.monotonic()
        on_time = start >= now
        if not on_time:
            self.late_count += 1
            start = now
            end = max(end, now + self.MIN_LATE_MUTE_S)
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
    TRUNCATED_WORD_EXTRA_S = 0.4
    # ASR window length; set from ASRManager.WINDOW_SECONDS by the owner.
    window_s = 2.5
    # A caption mention is remembered this long (lines scroll up and stay
    # visible for several seconds).
    CAPTION_MEMORY_S = 10.0
    # Caption text appears roughly as the word is spoken; mute this much
    # around the frame that first shows it (capture time).
    CAPTION_BEFORE_S = 0.7
    CAPTION_AFTER_S = 0.6

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
            self.scheduler.schedule(c0, c1, 'asr', transcript.strip()[:80])

    def on_caption_texts(self, texts, capture_time: float):
        if not (self.enabled and self.use_captions) or not texts:
            return
        now = time.monotonic()
        self._recent_captions = {k: t for k, t in self._recent_captions.items()
                                 if now - t < self.CAPTION_MEMORY_S}
        for line in texts:
            norm = _norm(line)
            if not norm:
                continue
            for m in (_FULL_NAME_RE, _FIRST_NAME_RE, _KING_JAMES_RE):
                for hit in m.finditer(norm):
                    # Key: the text before this mention plus its first word.
                    # Stable while the caption grows to the right or scrolls
                    # up, and the same whether the full-name or first-name
                    # pattern matched it.
                    key = (norm[:hit.start()] + hit.group(0).split(' ')[0])[-40:]
                    if key in self._recent_captions:
                        self._recent_captions[key] = now
                        continue
                    self._recent_captions[key] = now
                    self.caption_hits += 1
                    self.last_caption_text = line
                    self.scheduler.schedule(capture_time - self.CAPTION_BEFORE_S,
                                            capture_time + self.CAPTION_AFTER_S,
                                            'caption', line.strip()[:80])

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
