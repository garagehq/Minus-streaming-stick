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
import json
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
# The start of "LeBron" cut off by the end of an ASR window: "le", "leb",
# "lebr", "lebra…", "leron". Measured on 40 min of SenseVoice windows: 15
# such fragments were the name and 3 were another word ("le…" of "left",
# "let's"), which the next window corrects (see NameMuteController).
_WORD_PARTIAL_RE = re.compile(r"^(?:le|leb|lebr|lebr[aou]\w*|lebo\w*|ler|lero|leron|lbr\w*)$")


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

    # Default pads (used for ASR word spans). SenseVoice name starts land
    # after the caption word start by a median 0.16-0.19s (p90 0.2-0.6s) on
    # the two held-out videos, and its word ends run ~0.15s past the
    # caption's. Wider pads fully cover a few more names against the
    # caption track but mute more of the speech around them; these keep a
    # mention's mute to roughly the name plus ~0.6s.
    PAD_BEFORE_S = float(os.environ.get('MINUS_NAME_MUTE_PAD_BEFORE', '0.35'))
    PAD_AFTER_S = float(os.environ.get('MINUS_NAME_MUTE_PAD_AFTER', '0.25'))
    # A detection that arrives too late to cover the word still mutes this
    # long, so a missed head start still cuts the tail of the name.
    MIN_LATE_MUTE_S = float(os.environ.get('MINUS_NAME_MUTE_MIN_LATE', '0.8'))
    TICK_S = 0.01

    def __init__(self, audio, delay_fn):
        self._audio = audio
        self._delay_fn = delay_fn
        self._lock = threading.Lock()
        self._windows = []          # [(play_start, play_end, source, label, log entry)]
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
        # Optional callback(entry) for every detection (training-clip
        # collector, src/asr_clips.py). Runs on the detector's thread.
        self.on_detection = None
        # Optional callback(muted) whenever the mute switches (the BLOCKED
        # badge, src/loading_overlay.py). Runs on the scheduler's thread.
        self.on_mute_change = None

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
                 replace_sources=(), keep_head_sources=()):
        """Mute the playback of audio captured in [capture_start, capture_end].

        replace_sources: windows from these sources that overlap this one
        are narrowed first, so a precise detection replaces a coarse one for
        the same mention instead of widening it: dropped if not yet
        playing, cut to end with this window if they are.
        keep_head_sources: a dropped window from these sources that starts
        earlier still sets this one's start (its tail is what goes).
        """
        delay = max(0.0, float(self._delay_fn() or 0.0))
        pb = self.PAD_BEFORE_S if pad_before is None else pad_before
        pa = self.PAD_AFTER_S if pad_after is None else pad_after
        start = capture_start + delay - pb
        end = capture_end + delay + pa
        now = time.monotonic()
        if replace_sources:
            with self._lock:
                kept = []
                for w in self._windows:
                    if w[2] in replace_sources and w[0] < end + 0.8 and w[1] > start - 0.8:
                        if w[0] > now:
                            self._narrowed(w, w[0])      # not started: drop
                            if w[2] in keep_head_sources and w[0] < start:
                                start = w[0]
                            continue
                        w = self._narrowed(w, min(w[1], max(now, end)))
                    kept.append(w)
                self._windows = kept
        with self._lock:
            covered = any(w[0] <= start and end <= w[1] for w in self._windows)
        entry = {'source': source, 'label': label, 'detected': now,
                 'capture_start': capture_start, 'capture_end': capture_end,
                 'delay': delay, 'play_start': None, 'play_end': None,
                 'duplicate': covered, 'meta': meta or {}}
        self.detection_log.append(entry)
        if self.on_detection is not None:
            try:
                self.on_detection(entry)
            except Exception as e:
                logger.debug(f"[NameMute] on_detection hook failed: {e}")
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
            self._windows.append((start, end, source, label, entry))
            self.mute_count += 1
            self.events.append((time.time(), source, label, on_time))
            del self.events[:-50]
        logger.info(f"[NameMute] {source}: '{label}' -> mute in {start - now:+.2f}s "
                    f"for {end - start:.2f}s (delay {delay:.2f}s, "
                    f"{'on time' if on_time else 'LATE'})")

    def cancel(self, source: str, capture_start: float, capture_end: float) -> int:
        """End mutes from `source` whose audio lies inside the capture-time
        stretch [capture_start, capture_end]: drop them if not yet playing,
        cut them short if they are. Returns how many were changed."""
        delay = max(0.0, float(self._delay_fn() or 0.0))
        p0, p1 = capture_start + delay, capture_end + delay
        now = time.monotonic()
        changed = 0
        with self._lock:
            kept = []
            for w in self._windows:
                if w[2] == source and w[0] >= p0 - self.PAD_BEFORE_S - 0.05 and w[0] <= p1:
                    changed += 1
                    if w[0] > now:
                        self._narrowed(w, w[0])  # not started: drop
                        continue
                    w = self._narrowed(w, min(w[1], now))
                kept.append(w)
            self._windows = kept
        if changed:
            logger.info(f"[NameMute] {source}: provisional mute cancelled ({changed})")
        return changed

    @staticmethod
    def _narrowed(w, end):
        """Window `w` cut to end at `end` (== its start: dropped). Its
        detection-log entry follows, so offline analysis scores the mute
        that actually played."""
        entry = w[4] if len(w) > 4 else None
        if entry is not None:
            entry['play_end'] = end
            entry['narrowed'] = True
        return (w[0], end) + tuple(w[2:])

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
            return
        if self.on_mute_change is not None:
            try:
                self.on_mute_change(muted)
            except Exception as e:
                logger.debug(f"[NameMute] on_mute_change hook failed: {e}")

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


class CaptionLag:
    """How late the on-screen captions are relative to the speech.

    Each caption hit, once old enough that ASR has had its say (SETTLE_S), is
    paired with the nearest ASR name hit; the lag is caption frame time minus
    the ASR word start. YouTube's auto captions show within ~0.3 s of the word
    (live, 17 of 18 held-out videos); a broadcast-style track ran ~4.7 s
    behind, past the A/V delay, so every caption mute landed on the wrong
    words. Captions count as late while the median of the recent lags is
    over LATE_S.
    """
    SETTLE_S = 2.5     # past ASR's detection latency (p90 ~1.8 s)
    PAIR_S = 6.0       # a caption up to this long after an ASR hit pairs with it
    LATE_S = 1.5
    MIN_SAMPLES = 3
    KEEP = 7
    HORIZON_S = 180.0

    def __init__(self):
        self._asr = []                       # ASR name word starts (capture time)
        self._pending = []                   # caption hit times not yet paired
        self._caps = []                      # all recent caption hit times
        self._lags = collections.deque(maxlen=self.KEEP)   # (time, lag)

    def asr(self, c0: float):
        self._asr.append(c0)

    def caption(self, t: float, now: float) -> bool:
        """Record a caption hit at capture time t; True if captions run late."""
        self._settle(now)
        self._pending.append(t)
        self._caps.append(t)
        return self.late(now)

    def late(self, now: float) -> bool:
        lags = sorted(l for t, l in self._lags if now - t < self.HORIZON_S)
        return len(lags) >= self.MIN_SAMPLES and lags[len(lags) // 2] > self.LATE_S

    def _settle(self, now: float):
        keep = []
        for t in self._pending:
            if now - t < self.SETTLE_S:
                keep.append(t)
                continue
            near = [a for a in self._asr if -1.0 < t - a < self.PAIR_S]
            if not near:
                continue
            a = min(near, key=lambda a: abs(t - a))
            # One to one: a caption whose own mention ASR missed must not
            # pair with an earlier mention that has a closer caption.
            if all(abs(c - a) >= abs(t - a) for c in self._caps if c != t):
                self._lags.append((t, t - a))
        self._pending = keep
        oldest = min(keep, default=now) - self.PAIR_S - 3.0
        self._asr = [a for a in self._asr if a >= oldest]
        self._caps = [c for c in self._caps if c >= oldest - self.PAIR_S]


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
    # The same comparison on the text after the name (absorbs misreads of
    # the start of a line, "ig LeBron's GOAT status" / "ing LeBron's GOAT
    # status"), over at least this many characters.
    CAPTION_AFTER_FUZZY_RATIO = 0.75
    CAPTION_AFTER_MIN_CHARS = 5
    # A same-text mention on a line this many line-heights lower than where
    # it was last seen is a new caption line (roll-up captions move up); a
    # merged two-row OCR box moves its center by only ~half a line.
    CAPTION_NEW_LINE_DROP = 0.7
    # Text after the name less similar than this (over CAPTION_AFTER_MIN_CHARS
    # or more) is a different line even when the text before it matches.
    CAPTION_NEW_AFTER_RATIO = 0.5
    # Rolling captions add words at the end of the line, so a mention is
    # first seen at (or one or two words from) the end. A line that shows up
    # with the name further in, followed by MID_LINE_WORDS or more words, is
    # usually an on-screen banner ("... questioning LeBron's GOAT status")
    # that nobody is saying: it mutes provisionally, and an ASR window that
    # hears the moment the name would have been spoken whole, without the
    # name, ends that mute.
    MID_LINE_WORDS = 3
    # ...unless an OCR frame this recent showed the start of the same line
    # without the name: the line grew, so it is a rolling caption, whose
    # name is muted like any other caption hit (live, captions on,
    # 2026-10-29: 8 of 9 mid-line hits were spoken mentions, and the cancel
    # cost one -- "influence you LeBron I want you", ASR missed the name).
    CAPTION_GREW_S = 2.0
    # Text much smaller than the captions is a graphic (news headline,
    # ticker, scoreboard), which reappears every few tens of seconds and
    # mutes each time. Live, captions on (2026-10-29, 27 runs): hits under
    # 0.65x the median caption height were 13 of 71 off-mention hits and 2
    # of 647 real ones. Relative, since caption size is a viewer setting;
    # off until CAPTION_H_MIN_SAMPLES heights have been seen.
    CAPTION_MIN_REL_H = 0.65
    CAPTION_H_MIN_SAMPLES = 10

    # Partial names: a "le…"/"lebr…" fragment that is the last word of an
    # ASR window and starts within PARTIAL_EDGE_S of its end mutes
    # provisionally for PARTIAL_MUTE_S from the fragment start, one window
    # before the full name is heard. A later window that hears that stretch
    # whole without a name ends the mute early. Simulated on 40 min of two
    # held-out videos (tests/partial_name_sim.py): SenseVoice fully muted
    # 30/44 names at 1.2s delay instead of 19/44, and 20/44 instead of 7/44
    # at 1.0s. Parakeet-110M gains nothing (its 0.25s windows already see
    # the whole word a quarter-second later) and only gets false "le" hits,
    # so it is on by default for SenseVoice only.
    PARTIAL_EDGE_S = 0.5
    PARTIAL_MUTE_S = 0.8
    # Edge "king": a bare "king" touching a window edge is usually a cut-off
    # word -- the tail of "making"/"talking" when the window starts mid-word,
    # or "Kings" with the "s" past the window end. Such a hit is ignored; a
    # real "King" lies inside the next or previous window (hop 0.25-0.5 s)
    # and is muted from there. On by default (MINUS_NAME_MUTE_EDGE_KING=0
    # turns it off); live it cut fine-tuned SenseVoice's false mutes on
    # non-LeBron video to ~0.6/h with no lost mentions.
    EDGE_KING_S = 0.25

    def __init__(self, scheduler: NameMuteScheduler, matcher: NameMatcher = None,
                 partial_names: bool = False):
        self.scheduler = scheduler
        self.matcher = matcher or NameMatcher()
        self.enabled = True
        self.use_asr = True
        env = os.environ.get('MINUS_NAME_MUTE_PARTIAL')
        self.partial_names = partial_names if env is None else env != '0'
        self._recent_partial = []    # capture starts of provisional mutes
        self._recent_mid = []        # capture times of mid-line caption hits
        self._recent_lines = []      # (capture time, normalized line), all lines
        self.caption_mid_cancels = 0
        self.partial_hits = 0
        self.partial_cancels = 0
        # MINUS_NAME_MUTE_CAPTIONS=0 mutes from ASR only (OCR still runs).
        self.use_captions = os.environ.get('MINUS_NAME_MUTE_CAPTIONS', '1') != '0'
        self._recent_asr = []        # capture-time centers of recent ASR hits
        self._recent_captions = {}   # caption key -> last seen (monotonic)
        self.asr_hits = 0
        self.caption_hits = 0
        self.caption_late_skips = 0
        self.caption_small_skips = 0
        self._caption_h = collections.deque(maxlen=50)   # heights of name lines
        self.caption_lag = CaptionLag()
        self.last_asr_text = ''
        self.last_caption_text = ''
        self.edge_king = os.environ.get('MINUS_NAME_MUTE_EDGE_KING', '1') == '1'
        self.edge_king_skips = 0
        # MINUS_ASR_WORDS_LOG=<path>: append every ASR window's timed words
        # (JSON lines), so a live run can be replayed offline.
        self._words_log = None
        path = os.environ.get('MINUS_ASR_WORDS_LOG')
        if path:
            try:
                self._words_log = open(path, 'a', buffering=1)
            except OSError as e:
                logger.warning(f"ASR words log disabled: {e}")

    def on_asr_words(self, transcript: str, words, window_start: float):
        if not (self.enabled and self.use_asr):
            return
        if self._words_log is not None:
            try:
                self._words_log.write(json.dumps({
                    'ws': round(window_start, 3), 'len': self.window_s, 'text': transcript,
                    'words': [(w, round(s, 3), round(e, 3)) for w, s, e in (words or [])]}) + '\n')
            except (OSError, ValueError, TypeError):
                pass
        spans = self.matcher.find_word_spans(words) if words else []
        if not spans and self.matcher.find(transcript):
            # No usable word timings: mute the whole window.
            spans = [(0.0, max((e for _, _, e in words), default=2.5))]
        elif self.edge_king and spans:
            spans = self._drop_edge_king(words, spans)
        if words:
            self._cancel_unheard_captions(words, window_start, bool(spans))
        if self.partial_names and words:
            self._check_partial(words, window_start, bool(spans))
        if not spans:
            return
        self.last_asr_text = transcript
        now = time.monotonic()
        self._recent_asr = [c for c in self._recent_asr if now - c[0] < 15.0]
        for s, e in spans:
            if e - s < self.ASR_MIN_WORD_S:
                continue
            truncated = s <= 0.05 or e >= self.window_s - 0.05
            if s <= 0.05:
                # Word was already under way when this window began.
                s -= self.TRUNCATED_WORD_EXTRA_S
            if e >= self.window_s - 0.05:
                # Word runs past the end of this window; its true end is
                # unknown, so cover the likely remainder until a later
                # window sees it whole and narrows the mute.
                e += self.TRUNCATED_WORD_EXTRA_S
            c0, c1 = window_start + s, window_start + e
            center = (c0 + c1) / 2
            near = [c for c in self._recent_asr if abs(center - c[0]) < self.ASR_DEDUP_S]
            # Repeats of a mention are dropped, except the first whole
            # sighting after only cut-off ones: it has the real word edges.
            if near and (truncated or not all(c[1] for c in near)):
                continue
            self._recent_asr.append((center, truncated))
            if not near:
                self.asr_hits += 1
                self.caption_lag.asr(c0)
            # The ASR word span is the most precise timing there is: it
            # narrows any coarser mute (caption line, partial name, cut-off
            # word) for the same mention to itself -- except that an
            # earlier caption start is kept: SenseVoice's start can land
            # past the name's first syllable (live, captions on, 2026-10-29:
            # costs ~0.2 s more mute per name).
            self.scheduler.schedule(c0, c1, 'asr_edge' if truncated else 'asr',
                                    transcript.strip()[:80],
                                    {'word_start': round(s, 3), 'word_end': round(e, 3)},
                                    replace_sources=('caption', 'caption_mid',
                                                     'asr_partial', 'asr_edge'),
                                    keep_head_sources=('caption',))

    def _drop_edge_king(self, words, spans):
        """Remove bare-"king" spans that touch a window edge (see EDGE_KING_S)."""
        kings = {(s, e) for w, s, e in words if _WORD_KING_RE.match(_norm(w))}
        keep = []
        for s, e in spans:
            if (s, e) in kings and (s <= self.EDGE_KING_S or e >= self.window_s - self.EDGE_KING_S):
                self.edge_king_skips += 1
                continue
            keep.append((s, e))
        return keep

    def _cancel_unheard_captions(self, words, window_start: float, has_name: bool):
        """End mid-line caption mutes whose moment this window heard whole
        without the name (the name was on screen, not spoken)."""
        now = time.monotonic()
        self._recent_mid = [c for c in self._recent_mid if now - c < 15.0]
        if has_name or len(words) < 2 or not self._recent_mid:
            return
        inner = [(s, e) for _, s, e in words if s > 0.1 and e < self.window_s - 0.2]
        if not inner:
            return
        c0 = window_start + min(s for s, _ in inner)
        c1 = window_start + max(e for _, e in inner)
        # Caption text shows ~0.66 s after the word starts: the name would
        # have been spoken in [t - 1.0, t - 0.2] of the frame time t.
        hit = [c for c in self._recent_mid if c0 <= c - 1.0 and c - 0.2 <= c1]
        if hit:
            n = self.scheduler.cancel('caption_mid', min(hit) - self.CAPTION_BEFORE_S - 0.1,
                                      max(hit) + 0.1)
            self.caption_mid_cancels += n
            self._recent_mid = [c for c in self._recent_mid if c not in hit]

    def _check_partial(self, words, window_start: float, has_name: bool):
        now = time.monotonic()
        self._recent_partial = [c for c in self._recent_partial if now - c < 15.0]
        if not has_name and len(words) >= 2:
            # Words heard whole (clear of both window edges) with no name:
            # a provisional mute over that stretch was another word.
            inner = [(s, e) for _, s, e in words if s > 0.1 and e < self.window_s - 0.2]
            if inner:
                c0 = window_start + min(s for s, _ in inner)
                c1 = window_start + max(e for _, e in inner)
                if any(c0 <= c + 0.05 and c + 0.3 <= c1 for c in self._recent_partial):
                    n = self.scheduler.cancel('asr_partial', c0, c1)
                    if n:
                        self.partial_cancels += n
                        self._recent_partial = [c for c in self._recent_partial
                                                if not (c0 <= c + 0.05 and c + 0.3 <= c1)]
        if has_name:
            return
        w, s, _ = words[-1]
        if s < self.window_s - self.PARTIAL_EDGE_S or not _WORD_PARTIAL_RE.match(_norm(w)):
            return
        c0 = window_start + s
        if any(abs(c0 - c) < self.ASR_DEDUP_S for c in self._recent_partial):
            return
        self._recent_partial.append(c0)
        self.partial_hits += 1
        self.scheduler.schedule(c0, c0 + self.PARTIAL_MUTE_S, 'asr_partial', w,
                                {'word_start': round(s, 3)})

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
        earlier = [n for t, n in self._recent_lines if capture_time - self.CAPTION_GREW_S <= t < capture_time]
        self._recent_lines = [(t, n) for t, n in self._recent_lines
                              if capture_time - t < self.CAPTION_GREW_S]
        self._recent_lines += [(capture_time, _norm(l)) for l, _ in lines if _norm(l)]
        for line, meta in lines:
            norm = _norm(line)
            if not norm:
                continue
            if line.upper() == line and any(c.isalpha() for c in line):
                continue   # all-caps: an on-screen graphic, not a caption
            h = meta.get('h')
            if h and any(m.search(norm) for m in (_FULL_NAME_RE, _FIRST_NAME_RE, _KING_JAMES_RE)):
                small = (len(self._caption_h) >= self.CAPTION_H_MIN_SAMPLES and h <
                         self.CAPTION_MIN_REL_H * sorted(self._caption_h)[len(self._caption_h) // 2])
                self._caption_h.append(h)
                if small:
                    self.caption_small_skips += 1
                    continue
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
                    rest = norm[hit.end():].strip()
                    # Text after the name, the same whichever pattern matched.
                    tail = re.sub(r'^\S+(?: james\b)?\s*', '', norm[hit.start():])
                    if self._seen_caption(key, ctx, rest[:20], now,
                                          meta.get('y'), meta.get('h'), tail[:20]):
                        continue
                    self.caption_hits += 1
                    self.last_caption_text = line
                    if self.caption_lag.caption(capture_time, now) and self.use_asr:
                        # Captions behind the speech by more than they can
                        # cover: their mute would land on the wrong words.
                        self.caption_late_skips += 1
                        continue
                    pad_after = self.CAPTION_AFTER_S
                    if (hit.end() >= len(norm.rstrip())
                            and 'james' not in hit.group(0)):
                        pad_after += self.CAPTION_LINE_END_EXTRA_S
                    before = norm[:hit.start()].strip()
                    mid = (len(rest.split()) >= self.MID_LINE_WORDS
                           and not any(
                               4 <= len(n) <= len(before) and difflib.SequenceMatcher(
                                   None, n, before[:len(n)]).ratio() >= 0.85
                               for n in earlier))
                    if mid:
                        self._recent_mid.append(capture_time)
                    self.scheduler.schedule(capture_time, capture_time,
                                            'caption_mid' if mid else 'caption',
                                            line.strip()[:80], meta,
                                            pad_before=self.CAPTION_BEFORE_S,
                                            pad_after=pad_after)

    def _seen_caption(self, key: str, ctx: str, after: str, now: float,
                      y: float = None, h: float = None, tail: str = None) -> bool:
        """Has this caption mention been seen recently? Records it either way.

        Exact key, or a fuzzy match on the text before the name (when there
        is enough of it) or after it, which absorbs OCR misreads of one
        line. Matching entries are refreshed, so text that stays on screen
        (a banner) stays seen for as long as it is there.

        Not the same mention, whatever the text: a line lower on screen
        than where the entry was last seen (roll-up captions only move up,
        so it is a new line -- "LeBron comes down..." then "LeBron." 11 s
        later, "basket for LeBron so..." then "basket for LeBron that's..."),
        or one whose text after the name clearly differs.
        """
        def similar(a, b, ratio, n):
            k = min(len(a), len(b))
            return k >= n and difflib.SequenceMatcher(None, a[:k], b[:k]).ratio() >= ratio

        tail = after if tail is None else tail

        def same_line(t, ey):
            if y is not None and h and ey is not None and y - ey > self.CAPTION_NEW_LINE_DROP * h:
                return False
            k = min(len(t), len(tail))
            return not (k >= self.CAPTION_AFTER_MIN_CHARS and difflib.SequenceMatcher(
                None, t[:k], tail[:k]).ratio() < self.CAPTION_NEW_AFTER_RATIO)

        matches = [k for k, (_, c, a, ey, t) in self._recent_captions.items()
                   if (k == key
                       or (len(ctx) >= 4 and len(c) >= 4 and difflib.SequenceMatcher(
                           None, ctx, c).ratio() >= self.CAPTION_FUZZY_RATIO)
                       or similar(after, a, self.CAPTION_AFTER_FUZZY_RATIO,
                                  self.CAPTION_AFTER_MIN_CHARS))
                   and same_line(t, ey)]
        for k in matches:
            _, c, a, ey, t = self._recent_captions[k]
            self._recent_captions[k] = (now, c, a, y if y is not None else ey, t)
        self._recent_captions[key] = (now, ctx, after, y, tail)
        return bool(matches)

    def get_status(self) -> dict:
        st = self.scheduler.get_status()
        st.update({
            'enabled': self.enabled,
            'use_asr': self.use_asr,
            'use_captions': self.use_captions,
            'include_surname': self.matcher.include_surname,
            'asr_hits': self.asr_hits,
            'partial_names': self.partial_names,
            'partial_hits': self.partial_hits,
            'partial_cancels': self.partial_cancels,
            'edge_king': self.edge_king,
            'edge_king_skips': self.edge_king_skips,
            'caption_hits': self.caption_hits,
            'caption_mid_cancels': self.caption_mid_cancels,
            'caption_late_skips': self.caption_late_skips,
            'caption_small_skips': self.caption_small_skips,
            'captions_late': self.caption_lag.late(time.monotonic()),
            'last_asr_text': self.last_asr_text[:200],
            'last_caption_text': self.last_caption_text[:200],
        })
        return st
