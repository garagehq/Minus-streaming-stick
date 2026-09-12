"""Ad countdown timestamps as a duration signal.

Ads tell us how long they are going to last. "Ad 0:15", "Skip in 5",
"Ad 10" -- OCR already reads these every frame and the decision engine
throws the number away, keeping only the boolean "an ad keyword matched".

That number is the single most informative thing on the screen:

  * It is a LOWER BOUND on how long the ad continues. While the clock says
    12 seconds remain, an OCR frame that fails to re-read the keyword is
    obviously a misread, not the end of the ad. This is exactly the
    evidence the anti-flap machinery was missing -- it had been inferring
    "is the ad still there?" purely from consecutive-miss counts.

  * It predicts WHEN the ad ends, so the block can be released promptly
    instead of waiting out a fixed no-ad threshold after the fact.

It is a HINT, never a command. The countdown can be misread, can refer to
a skip button rather than the ad, can be frozen by a pause, and pods chain
one ad into the next. So everything here is advisory: it can ask the
engine to hold a little longer or to release a little sooner, and every
existing safeguard (max duration, frozen-stream detection, VLM dissent)
still runs on top.

Pause is the important failure mode. If the viewer pauses mid-ad the
countdown stops decrementing, and a naive deadline would hold the block
for the rest of the pause. A frozen countdown is therefore treated as
evidence of a pause, and the caller corroborates it with audio.
"""

import re
import time
from typing import List, Optional

# OCR misreads digits in ad timers constantly. Documented in CLAUDE.md:
# 0 -> o/O, 1 -> l/L/I/i, and the separator : -> ; or .
# 5 -> s/S shows up in the wild too ("0:1s" for "0:15").
_DIGIT_FIX = str.maketrans({
    'o': '0', 'O': '0',
    'l': '1', 'L': '1', 'I': '1', 'i': '1',
    's': '5', 'S': '5',
    ';': ':', '.': ':',
})

# M:SS or MM:SS, allowing the misread characters above in either position.
# Bounded by digits AND by the separator, not by letters: the timer is very
# often glued to the keyword ('Ad0:30', 'Ado:15'), so a letter before the first
# digit is the normal case rather than a reason to reject. The digit bounds
# stop us slicing a longer number apart, and the separator bounds stop us
# slicing an H:MM:SS runtime apart.
#
# The separator half was caught live: a 70-minute music mix shows its runtime
# as "1:10:55", whose leading "1:10" is a perfectly well-formed M:SS and parsed
# as a 70-second ad countdown on 286 non-ad frames in one night. MAX_PLAUSIBLE_
# SECONDS cannot catch that -- 70 is an entirely plausible ad length. Refusing
# to match when a colon sits on either side leaves real ad timers untouched
# (nothing renders "Ad 0:30" inside a longer timestamp) and drops the whole
# class.
_TS_RE = re.compile(r'(?<![0-9:;.])([0-9oOlIisS]{1,2})[:;.]([0-9oOlIisS]{2})(?![0-9:;.])')

# "Ad 15" / "Ad15" -- Netflix-style bare seconds countdown.
_BARE_RE = re.compile(r'\bad' + r'[\s|·,]{0,3}' + r'([0-9oOlIisS]{1,2})\b')

# "Skip in 5" / "Skip ad in 12s". This is time until the SKIP BUTTON
# appears, not until the ad ends, so it is only a lower bound.
# OCR frequently returns the label and the digit as SEPARATE text elements,
# which arrive here joined -- observed live as "Sponsored | Skip in | 5". A
# short run of separator characters is therefore allowed between the label and
# the number. Kept short so a digit elsewhere on screen cannot be captured.
_SEP = r'[\s|:·,.\-]{0,4}'
_SKIP_RE = re.compile(r'skip' + _SEP + r'(?:ad' + _SEP + r')?(?:in' + _SEP + r')?'
                      r'([0-9oOlIisS]{1,2})\s*s?\b')

# An ad timer is short. Anything longer is a wall clock, a programme runtime
# or a misread that wandered into the ad text.
#
# Caught live: "CIil | 12:49 | a" parsed as 769 seconds. 12:49 is a clock, and
# at the old 15-minute ceiling it sailed through -- a fabricated 12-minute
# deadline is exactly the kind of thing that could pin a block. Real ad breaks
# top out around 2-3 minutes, so 180s rejects the nonsense while leaving every
# plausible ad timer intact.
MAX_PLAUSIBLE_SECONDS = 180


def _to_int(raw: str) -> Optional[int]:
    """Digits, after undoing OCR letter misreads.

    Requires at least one ACTUAL digit. The misread table maps s->5 and
    o->0, so without this an all-letter run reads as a number: the plural
    in "Free with ads PG" parses as ad + s = 5 seconds, which is how this
    was caught live. A real timer always contains a real digit.
    """
    if not any(c.isdigit() for c in raw):
        return None
    return _to_int_lenient(raw)


def _to_int_lenient(raw: str) -> Optional[int]:
    """Same conversion without the real-digit requirement.

    Used for the halves of a M:SS timer, where the digit rule applies to the
    match as a WHOLE -- "Ado:3o" is 0:30 and its only real digit lives in the
    seconds half, so checking each half separately would reject it.
    """
    fixed = raw.translate(_DIGIT_FIX)
    return int(fixed) if fixed.isdigit() else None


def parse_ad_remaining(texts) -> Optional[int]:
    """Seconds left in the ad, or None.

    Prefers an explicit M:SS timer over a bare seconds count, because the
    bare form collides with "Ad 1 of 2" style pod counters.
    """
    if not texts:
        return None
    joined = ' '.join(str(t) for t in texts if t)
    low = joined.lower()

    # "Ad 1 of 2" is a pod position, not a duration. Strip it before the
    # bare-number pass so "Ad 1" doesn't read as 1 second.
    low_nopod = re.sub(r'ad\s*\d+\s*of\s*\d+', ' ', low)

    best = None
    for m in _TS_RE.finditer(low_nopod):
        # At least one real digit across the whole timer.
        #
        # This deliberately gives up on a FULLY misread timer such as
        # "Adl:lo" (1:10, every character a misread). Without the rule the
        # same pattern also accepts ordinary words -- "hello:so" parses as
        # 10:50 -- and a fabricated 650-second deadline would pin a block.
        # Skipping an unreadable frame costs nothing: the countdown is
        # advisory and the existing counters still decide. Guessing wrong
        # costs a held overlay.
        if not any(c.isdigit() for c in m.group(1) + m.group(2)):
            continue
        mins, secs = _to_int_lenient(m.group(1)), _to_int_lenient(m.group(2))
        if mins is None or secs is None or secs >= 60:
            continue
        total = mins * 60 + secs
        if 0 < total <= MAX_PLAUSIBLE_SECONDS and (best is None or total < best):
            best = total
    if best is not None:
        return best

    m = _BARE_RE.search(low_nopod)
    if m:
        v = _to_int(m.group(1))
        if v is not None and 0 < v <= MAX_SKIP_GATE_SECONDS:
            return v
    return None


# "Skip in" with the digit missing. OCR drops the countdown number constantly
# -- measured live on a Disney+ pre-roll, 0 of 36 in-block frames yielded a
# number while "Skip in" / "Skip I" was plainly on screen. The label alone
# still carries a fact: the skip button is NOT yet available, so the ad is in
# its unskippable opening phase and is definitely still running. "Skip intro"
# is a show control and must never match.
_SKIP_LABEL_RE = re.compile(r'skip' + r'[\s|:·,.\-]{0,4}' + r'(?:ad[\s|:·,.\-]{0,4})?i(?:n\b|\b)')
_SKIP_INTRO_RE = re.compile(r's[k]?[i1lI]p[\s|:·,.\-]{0,4}[i1lI]ntro')

# How much longer to assume the unskippable phase runs when the digit is
# unreadable. This is NOT the length of the gate -- it is how long to keep
# holding after the label was last read, since the deadline refreshes on every
# frame the label appears.
#
# The original 5 was picked from "streaming skip gates are ~5s", which the
# soak disproved for this content: gates measured p50 21.5s and up to 53s.
# Worse, 5s was short enough to end the block mid-ad and then re-block, and it
# did -- 60% of every block that night ended at 5-6s, the constant's own
# length, and half of all blocks re-blocked within 5s.
#
# 10 is the expected remaining gate time given an unreadable digit: with a
# ~21s gate and no information about where in it we are, the midpoint is the
# estimate. It is still only a lower bound (skip-bound), so it can hold a
# block but never end one, and the audio pause check still releases it.
SKIP_LABEL_ASSUMED_S = 10


def has_skip_countdown_label(texts) -> bool:
    """True when "Skip in" is on screen but its digit was not readable."""
    if not texts:
        return False
    low = ' '.join(str(t) for t in texts if t).lower()
    if _SKIP_INTRO_RE.search(low):
        return False
    if parse_skip_in(texts) is not None:
        return False        # the digit WAS readable; use the real number
    return bool(_SKIP_LABEL_RE.search(low))


# A standalone number sitting in its own OCR element, e.g. the "20" in
# ['Sponsored', 'Skip in', 'Uber', 'Send to phone', '20', 'uber.com'].
_LONE_NUM_RE = re.compile(r'^[0-9oOlIisS]{1,2}$')

# A skip gate is ~5-30s in practice; nothing makes a viewer wait longer before
# offering the button. A loose cap just lets unrelated on-screen numbers in --
# the Uber creative that counts 20..14 also renders a "75" and a "73", and at
# the old 120s ceiling the "75" was accepted as a countdown because it happened
# to sit right after the label. Applies to both the adjacent and cross-element
# forms.
MAX_SKIP_GATE_SECONDS = 60


def _skip_in_cross_element_candidates(texts) -> List[int]:
    """Recover a skip countdown whose digit landed in a separate OCR element.

    PaddleOCR returns each detected text box as its own element, and the
    countdown digit on a YouTube skip button is its own box. The element
    ORDER follows detection, not layout, so the digit routinely arrives
    several elements away from the label -- measured overnight, the digit was
    present on 46 of 107 'Skip in' frames but immediately after the label on
    only 2. The adjacency-based regex therefore saw almost none of them and
    the tracker fell back to a flat assumption for the rest.

    Only ever a lower bound, same as the adjacent form. A candidate still has
    to survive the tracker's corroboration before it can hold a block, so a
    stray number that happens to be in range cannot pin anything on its own --
    it has to decrement in real time across frames to be believed.
    """
    if not texts:
        return []
    elements = [str(t).strip() for t in texts if t and str(t).strip()]
    label_at = None
    for i, el in enumerate(elements):
        if _SKIP_LABEL_RE.search(el.lower()):
            label_at = i
            break
    if label_at is None:
        return []
    found = []
    for i, el in enumerate(elements):
        if i == label_at or not _LONE_NUM_RE.match(el):
            continue
        v = _to_int(el)
        if v is None or not 0 < v <= MAX_SKIP_GATE_SECONDS:
            continue
        # Nearest to the label first; the smaller value breaks a tie, since a
        # countdown is the number on screen that is heading for zero.
        found.append((abs(i - label_at), v))
    return [v for _, v in sorted(found)]


def parse_skip_in_candidates(texts) -> List[int]:
    """Every plausible skip countdown in the frame, best guess first.

    A frame often contains more than one number that could be the countdown:
    the real one, plus whatever the creative happens to render. Position alone
    cannot separate them -- a Disney+ ad puts a static "33" right where the
    countdown sits. Resolving it needs the running clock, which lives in
    AdCountdownTracker, so the candidates are handed there rather than being
    collapsed to one guess here. See AdCountdownTracker.observe_candidates.
    """
    if not texts:
        return []
    low = ' '.join(str(t) for t in texts if t).lower()
    out = []
    m = _SKIP_RE.search(low)
    if m:
        v = _to_int(m.group(1))
        if v is not None and 0 < v <= MAX_SKIP_GATE_SECONDS:
            out.append(v)
    for v in _skip_in_cross_element_candidates(texts):
        if v not in out:
            out.append(v)
    return out


def parse_skip_in(texts) -> Optional[int]:
    """Seconds until the skip button appears. Lower bound on ad length."""
    if not texts:
        return None
    low = ' '.join(str(t) for t in texts if t).lower()
    m = _SKIP_RE.search(low)
    if m:
        v = _to_int(m.group(1))
        if v is not None and 0 < v <= MAX_SKIP_GATE_SECONDS:
            return v
    cands = _skip_in_cross_element_candidates(texts)
    return cands[0] if cands else None


class AdCountdownTracker:
    """Tracks the ad clock across OCR frames.

    Readings are noisy, so a single one is never trusted on its own. Two
    readings that decrement roughly in real time are, because a misread is
    very unlikely to land on a plausible successor of the previous value.
    """

    # A reading must be corroborated before it is allowed to hold a block.
    MIN_READINGS = 2
    # How far a reading may disagree with the projected value and still be
    # treated as the same countdown rather than a new one.
    TOLERANCE_S = 4.0
    # Repeating the same value for longer than this means the clock is not
    # advancing: a pause, or a frozen stream.
    FROZEN_AFTER_S = 4.0
    # A reading must be at least this recent for "frozen" to mean anything.
    # Deliberately NOT FROZEN_AFTER_S: sharing one constant makes the two
    # conditions collide at the boundary and misreport a dropout as frozen
    # for exactly one cycle. This is about one OCR cadence.
    READING_FRESH_S = 2.5
    # Never hold on a countdown older than this without a fresh reading.
    STALE_AFTER_S = 30.0

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self._deadline = 0.0        # when the ad is expected to end
        self._readings = 0          # corroborating readings for this countdown
        self._last_value = None     # last parsed seconds-remaining
        self._last_seen = 0.0       # when we last parsed any value
        self._value_since = 0.0     # when _last_value was first seen
        self._is_skip_bound = False  # from "skip in", a lower bound only
        # Quarantine for readings that contradict an established countdown.
        self._pending_deadline = 0.0
        self._pending_count = 0

    def observe_candidates(self, candidates, now: Optional[float] = None,
                           is_skip_bound: bool = False) -> Optional[int]:
        """Observe the candidate that best fits the clock already running.

        OCR hands back several numbers per frame and only one of them is the
        countdown. Position is a weak tie-breaker and gets it wrong on real
        creatives, but a countdown that is already anchored makes the choice
        almost free: the right number is the one near where this clock says it
        should be. Measured overnight, the noise values (a static 33, a
        recurring 2) sit nowhere near the projection while the true reading
        lands within a second of it.

        With no clock running yet there is nothing to match against, so the
        first candidate (nearest the label) is taken and MIN_READINGS still
        stops it holding anything until a second reading agrees.

        Returns the value actually observed, for logging.
        """
        if not candidates:
            return None
        now = now if now is not None else time.time()
        chosen = candidates[0]
        if self._deadline > 0 and self._readings > 0:
            projected = self._deadline - now
            fits = [c for c in candidates
                    if abs(projected - c) <= self.TOLERANCE_S]
            if fits:
                chosen = min(fits, key=lambda c: abs(projected - c))
        self.observe(chosen, now, is_skip_bound=is_skip_bound)
        return chosen

    def observe(self, seconds: Optional[int], now: Optional[float] = None,
                is_skip_bound: bool = False, synthetic: bool = False) -> None:
        """Record a countdown reading. None means the frame had no clock.

        Three cases, and the distinction between them is the whole point:

        AGREES -- the reading is close to what the running clock projects.
        Re-anchor on it. This is how 0:15 then 0:12 becomes a 12-second
        deadline: each agreeing reading retargets, absorbing OCR cadence
        jitter and any drift.

        CONTRADICTS, with no clock running -- bootstrap. One reading is
        never enough to hold a block (MIN_READINGS), so this is safe.

        CONTRADICTS an established clock -- quarantine it. A jump from 0:15
        to 0:99 is far more likely a misread than a real ad, so it must NOT
        reassign the deadline on the spot; doing that both fabricates a
        99-second hold and throws away the corroboration already earned,
        collapsing a legitimate hold mid-ad. The value is held aside instead
        and only adopted once a LATER reading agrees with IT -- 0:99 followed
        by 0:97. That is also exactly how a genuine resync looks when the
        next ad in a pod starts, so real transitions are still picked up,
        just one reading later.
        """
        if seconds is None:
            return
        now = now if now is not None else time.time()

        # Freshness and the frozen-clock check follow every observation,
        # adopted or not: "the same number keeps arriving" is what a pause
        # looks like, regardless of whether we act on the number.
        #
        # Synthetic readings are exempt. They come from a countdown LABEL whose
        # digit was unreadable, so the value is a constant we chose -- it
        # repeats by construction and would trip the freeze detector on every
        # ad, killing the hold it exists to provide. Pause protection for this
        # path comes from audio instead (_playback_looks_paused), which is the
        # authoritative playback signal and does not depend on the number.
        if synthetic or seconds != self._last_value or not self._value_since:
            self._value_since = now
        self._last_value = seconds
        self._last_seen = now

        established = self._deadline > 0 and self._readings > 0
        projected = (self._deadline - now) if established else None

        if projected is not None and abs(projected - seconds) <= self.TOLERANCE_S:
            self._readings += 1
            self._anchor(seconds, now, is_skip_bound)
            return

        if not established:
            self._readings = 1
            self._anchor(seconds, now, is_skip_bound)
            return

        # Contradicts a running clock: quarantine rather than reassign.
        pending_projected = ((self._pending_deadline - now)
                             if self._pending_count else None)
        if (pending_projected is not None
                and abs(pending_projected - seconds) <= self.TOLERANCE_S):
            self._pending_count += 1
            self._pending_deadline = now + seconds
            if self._pending_count >= self.MIN_READINGS:
                # Corroborated twice: this really is a different countdown.
                self._readings = self._pending_count
                self._anchor(seconds, now, is_skip_bound)
        else:
            self._pending_deadline = now + seconds
            self._pending_count = 1

    def _anchor(self, seconds: int, now: float, is_skip_bound: bool) -> None:
        """Adopt a reading as the live countdown."""
        self._deadline = now + seconds
        self._is_skip_bound = is_skip_bound
        self._pending_deadline = 0.0
        self._pending_count = 0

    def remaining(self, now: Optional[float] = None) -> float:
        """Seconds the ad is still expected to run. 0 once it has elapsed."""
        now = now if now is not None else time.time()
        if not self._deadline:
            return 0.0
        return max(0.0, self._deadline - now)

    def is_frozen(self, now: Optional[float] = None) -> bool:
        """The same value keeps coming back: paused, or a stuck stream.

        Requires readings to still be ARRIVING. Not reading anything for a
        few seconds is an OCR dropout -- the case this whole signal exists
        to ride out -- and must not be mistaken for a stopped clock, which
        would release the block at exactly the wrong moment.
        """
        now = now if now is not None else time.time()
        if self._last_value is None or not self._value_since:
            return False
        if (now - self._last_seen) > self.READING_FRESH_S:
            return False        # no readings at all: a dropout, not a freeze
        return (now - self._value_since) >= self.FROZEN_AFTER_S

    def is_stale(self, now: Optional[float] = None) -> bool:
        """No reading for a long time; stop leaning on the old one."""
        now = now if now is not None else time.time()
        if not self._last_seen:
            return True
        return (now - self._last_seen) >= self.STALE_AFTER_S

    def is_confident(self, now: Optional[float] = None) -> bool:
        """Enough corroboration to let this influence blocking."""
        return (self._readings >= self.MIN_READINGS
                and not self.is_stale(now))

    def should_hold(self, now: Optional[float] = None) -> bool:
        """Does the clock say the ad is definitely still running?

        Used to veto a stop. Deliberately conservative: needs corroboration,
        a live reading, and time actually left. A frozen clock never holds,
        which is what keeps a pause from pinning the overlay up.
        """
        now = now if now is not None else time.time()
        if not self.is_confident(now) or self.is_frozen(now):
            return False
        return self.remaining(now) > 0

    def expired(self, now: Optional[float] = None) -> bool:
        """The clock ran out, so the ad should be over.

        Lets the engine release promptly instead of waiting out a fixed
        no-ad threshold after the ad has visibly ended.
        """
        now = now if now is not None else time.time()
        if not self.is_confident(now) or self._is_skip_bound:
            # "Skip in N" only bounds the start of the skip window; the ad
            # keeps running past it, so it can never mean "ad is over".
            return False
        return self.remaining(now) <= 0

    def status(self) -> dict:
        now = time.time()
        return {
            'remaining_s': round(self.remaining(now), 1),
            'last_value': self._last_value,
            'readings': self._readings,
            'confident': self.is_confident(now),
            'frozen': self.is_frozen(now),
            'stale': self.is_stale(now),
            'skip_bound': self._is_skip_bound,
        }
