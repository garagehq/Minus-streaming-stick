"""
ASR ad-marker keyword detection (backend-agnostic: works with whisper.cpp,
faster-whisper, or Moonshine — see docs/ASR.md for the benchmark that
chose faster-whisper as the current backend).

whisper-tiny.en is small and fast (~4× real-time on 3 cores of RK3588's
A76+A55) but it WILL make transcription errors — similar shape to OCR
misreads. The marker set below is designed for that constraint:

  - Phrase-level matching, not substring (avoids "now" matching everywhere).
  - Multiple spelling variants per marker (whisper drops syllables, mangles
    less-frequent words). "Available now" → also "vailable now", "the lable now",
    etc.
  - Regex for structurally-stable shapes (prices, phone numbers, dot-coms).
  - Whitespace-tolerant: whisper randomly inserts/drops spaces around
    contractions and numbers.

This is the *only* module that knows which transcripts look like marketing
copy. The decision-engine (Minus._asr_verdict) just consumes the integer
`count_marker_hits()` result. Keep this list portable — no Minus dependencies.

If whisper-tiny.en quality proves insufficient, bumping to base.en or small.en
won't change anything about this module: more accurate transcripts produce
the same hits and fewer false negatives.
"""

import re

# ---------------------------------------------------------------------------
# Marker phrases. Each entry can be a literal string (case-insensitive
# substring search after whitespace collapse) OR a tuple (regex_pattern,
# flags) for shape-based matches. Pick whichever fits — literal forms are
# easier to scan, regex forms catch the shape-shifting cases (prices, URLs).
#
# Naming convention: CTA = call-to-action phrase, PRICE = monetary, URL =
# web address / phone-number shape, BRAND = brand-name spoken in pitch
# context.
# ---------------------------------------------------------------------------

# CTA phrases — these almost only appear in ad voiceover scripts. We list
# multiple spelling variants per phrase because whisper-tiny drops
# unstressed syllables ("available" → "vailable", "introducing" → "tro-
# ducing").
ASR_CTA_MARKERS = [
    # available now / available at
    'available now', 'vailable now', 'a vailable now',
    'available at', 'vailable at',
    'available in stores', 'available wherever',
    # limited time
    'limited time', 'limit time', 'limited time only',
    'for a limited time',
    # call / order
    'call now', 'order now', 'order yours', 'order today',
    'call today',
    'shop now', 'buy now', 'get yours',
    # visit / go to
    'visit us at', 'visit our', 'go to our',
    'sign up at', 'sign up today',
    # download
    'download the app', 'download our app', 'download today',
    'download now',
    # introducing / new
    'introducing the', 'introducing our',
    'new from', 'all new', 'all-new',
    # offer / deal
    'special offer', 'special deal', 'today only',
    "don't miss", 'dont miss', 'do not miss',
    # exclusivity
    'while supplies last', 'while quantities last',
    'in select', 'at participating',
    # ad self-references (extremely strong signals)
    'this commercial', 'this advertisement', 'this message',
    'paid for by', 'paid sponsorship',
    'this episode is brought to you by', 'brought to you by',
    'sponsored by', 'sponsored content',
    # restrictions / disclaimers (heard in nearly all ads)
    'restrictions apply', 'terms apply',
    'terms and conditions apply', 'see store for details',
    'consult your doctor', 'side effects may include',
    'ask your doctor about',
    # price language
    'save up to', 'save big', 'save now',
    'percent off', 'percentoff',
    'free shipping', 'free trial',
    'no cost to you', 'absolutely free',
    # comparison ad markers
    'compared to', 'unlike other',
    'voted number one', 'voted #1',
    # rating / recommendation
    "doctor recommended", 'dentists recommend',
    'clinically proven',
    # --- expanded marker set (2026-05) — broaden ASR ad-copy coverage ---
    # urgency / CTA
    'act now', 'act fast', 'hurry', 'hurry in', "don't wait", 'dont wait',
    'get started today', 'find out more', 'find out how', 'learn more at',
    'for more information', 'for more info', 'more info', 'click the link',
    'tap the link', 'swipe up', 'link below', 'check out',
    # try / guarantee
    'try it free', 'try it today', 'try risk free', 'risk free', 'risk-free',
    'money back', 'money-back guarantee', 'satisfaction guaranteed',
    'guaranteed', '100% guaranteed', 'no risk',
    # sale / deal / urgency-time
    'on sale now', 'on sale', 'sale ends', 'ends soon', 'ends tonight',
    'this week only', 'this weekend', 'today and tomorrow', 'while it lasts',
    'last chance', 'final days', "deal of the day", 'door buster',
    # availability / location
    'in stores now', 'in stores', 'a store near you', 'near you',
    'a dealer near you', 'see your local', 'visit your local',
    'at your local', 'find a store', 'find a location', 'nationwide',
    # pre-order / reserve / claim
    'pre order', 'pre-order', 'preorder', 'reserve yours', 'reserve now',
    'claim your', 'claim yours', 'get yours today',
    # codes / coupons
    'use promo code', 'promo code', 'use code', 'enter code', 'with code',
    'coupon code', 'discount code', 'use the code',
    # bundle / quantity
    'buy one get one', 'bogo', 'two for one', 'half price', 'half off',
    'free gift', 'free sample', 'free quote', 'free consultation',
    'free estimate', 'free shipping and returns',
    # subscription / commitment
    'cancel anytime', 'no contract', 'no commitment', 'no obligation',
    'first month free', 'months free', 'no hidden fees', 'no fees',
    # price superlatives
    'lowest price', 'lowest prices', 'best price', 'best deal',
    'great deal', 'unbeatable', 'lowest rate', 'low monthly',
    # switch / save (insurance / telco ad staples)
    'switch to', 'switch and save', 'switch today', 'bundle and save',
    'save on your', 'lower your', 'cut your bill', 'get a quote',
    'get a free quote', 'free quote today',
    # pharma / health disclaimers
    'ask your pharmacist', 'talk to your doctor', 'ask your doctor',
    'do not take if', 'individual results may vary', 'results may vary',
    'results not typical', 'consult a physician', 'use as directed',
    # pharma side-effect disclosures: a long stretch of them read like show
    # dialog with no markers and the 8 s window vetoed a real Mounjaro ad
    # (minus, 2026-10-19). Phrases ad copy is legally required to say and
    # TV dialog almost never does.
    'side effects', 'serious side effects', 'common side effects',
    'allergic reaction', 'serious allergic', 'thyroid cancer', 'medullary thyroid',
    'endocrine neoplasia', 'stop taking', 'tell your doctor', 'call your doctor',
    'available by prescription', 'prescription medicine', 'prescription medication',
    'once weekly', 'once daily', 'may cause', 'may increase the risk',
    'not for people with', 'pancreatitis', 'gallbladder problems',
    'low blood sugar', 'kidney problems', 'fda approved', 'fda-approved',
    # sports-betting disclaimers: these ads open with game footage and
    # commentary, so the window heard "show dialog" and vetoed a Fanatics
    # Sportsbook ad for ~4 s until OCR caught it (minus, 2026-10-19). The
    # legal lines are said in every one and never in a broadcast.
    'gambling problem', 'if you or someone you know has a gambling',
    '1 800 gambler', '1-800-gambler', 'eight hundred gambler', 'call 1 800',
    'must be 21', 'must be twenty one', '21 and over', '21 plus',
    'sportsbook', 'sports book', 'bonus bets', 'bet credits',
    'deposit match', 'first bet', 'no sweat bet', 'odds boost',
    'void where prohibited', 'terms apply',
    # sponsor self-reference
    'official sponsor', 'proud sponsor', 'proud partner',
    'in partnership with', 'presented by',
    # phone / screen CTAs
    'call us', 'call the number', 'number on your screen',
    'call the number on your screen', 'text us', 'visit the website',
    # shop online
    'shop online', 'order online', 'online or in stores', 'shop now at',
    # entertainment promos
    'in theaters', 'in theaters now', 'now streaming', 'coming soon',
    'this fall', 'this summer', 'this holiday season', 'rated pg',
    # product hype
    'limited edition', 'brand new', 'all-new', 'now available',
    'introducing', 'proven to', 'designed to', 'engineered to',
]

# Shape-based regex markers. Compiled lazily once.
#
# Tightening note (2026-05-27): the bare price regex `\$\s?\d+\b` was
# removed after the control corpus showed a false positive — TTS-rendered
# show dialog "She paid fifty dollars for that dress" was transcribed by
# whisper-tiny.en as "She paid $15 for net rest", and bare "$15" hit the
# old regex. Show characters mention round-dollar amounts all the time;
# whisper rarely transcribes those with cents. So we now only score
# *contextual* prices:
#   - $X.XX with decimal (ad-typical pricing — "$9.99")
#   - "only $X", "only X dollars" (CTA prefix)
#   - "save $X", "starting at $X"
#   - "X dollars and Y cents" (full ad-style spoken price)
# This loses no real ads (ads always use one of these patterns), and
# keeps show-dialog false positives at zero.
_REGEX_MARKERS_SOURCE = [
    # Prices WITH cents — strict $X.XX pattern. Ads frequently say
    # "$9.99", "$19.95"; show characters rarely speak prices with cents.
    (r'\$\s?\d{1,4}[\.,]\d{2}\b', 'price-cents'),
    # Contextual prices (require an ad-style preceding word).
    (r'\bonly\s+\$?\s?\d{1,4}(?:[\.,]\d{2})?\b', 'price-only'),
    (r'\bonly\s+\d{1,4}\s+dollars?\b', 'price-only-spoken'),
    (r'\bsave\s+\$?\s?\d{1,4}(?:[\.,]\d{2})?\b', 'price-save'),
    (r'\bstarting\s+(?:at|from)\s+\$?\s?\d{1,4}\b', 'price-starting'),
    # Spoken full price — "X dollars and Y cents"
    (r'\b\d{1,4}\s*dollars?\s+(?:and\s+)?\d{1,2}\s*cents?\b', 'price-spoken'),
    # Spoken X-99 pattern — "nine ninety nine" whisper often renders as
    # "9 99" or "9.99" or "999". Keep this tight: the "99" suffix is
    # the universal ad-pricing tail.
    (r'\b\d{1,3}\s+(?:ninety[\s-]?nine|99)\b', 'price-x99-spoken'),
    # Percent off — "50% off", "save 50 percent"
    (r'\b\d{1,3}\s*%\s*off\b', 'pct-off'),
    (r'\bsave\s+\d{1,3}\s*percent\b', 'pct-off-spoken'),
    # URLs spoken out — "visit example dot com", "go to brand dot net"
    (r'\b\w{2,}\s+dot\s+(?:com|org|net|co|io|us)\b', 'url-spoken'),
    # URLs in WRITTEN form — whisper transcribes "Hotels.com" / "brand.net"
    # literally with the dot (this is exactly the Hotels.com ad we missed).
    (r'\b[a-z][\w-]{1,}\.(?:com|net|org|io|co|tv|app|shop|store|gov)\b', 'url-written'),
    # 1-800 / 1-888 numbers — usual ad phone-number shapes
    (r'\b1[\s-]?(?:800|888|877|866|855|844)[\s-]?[\d-]{3,}\b', 'phone-tollfree'),
    # Spoken toll-free — "one eight hundred", "one eight oh oh"
    (r'\bone\s+eight\s+(?:hundred|oh\s+oh)\b', 'phone-spoken'),
    # "Up to N%" / "up to N percent"
    (r'\bup\s+to\s+\d{1,3}\s*(?:%|percent)\b', 'up-to-pct'),
    # "N years/months free"
    (r'\b\d{1,2}\s+(?:months?|years?)\s+free\b', 'period-free'),
]

# Build compiled forms once.
_COMPILED_REGEX = [
    (re.compile(pat, re.IGNORECASE), label) for pat, label in _REGEX_MARKERS_SOURCE
]

# Phrases that are NOT ad markers even though they share keywords.
# Whisper hallucinates pleasantries; show dialog mentions money and shopping
# without being an ad. False positives we've enumerated:
ASR_EXCLUSIONS = [
    # show metadata read aloud (Netflix, YouTube descriptions)
    'available on netflix', 'available on prime', 'available on hulu',
    'available on disney', 'available on apple', 'available on max',
    'streaming now on',
    # show recap voiceovers
    'previously on',
    # Common YouTube-creator phrasings (NOT marketing copy)
    'subscribe to my channel', 'like and subscribe', 'hit that subscribe',
    'check out my', 'link in the description',
    # incidental show dialog around money
    'how much does it cost', 'how much is that',
]


def _normalize_for_match(text: str) -> str:
    """Lowercase + collapse runs of whitespace + strip punctuation that
    whisper sprinkles around contractions ("dont't", "I 'm")."""
    if not text:
        return ''
    s = text.lower()
    # Collapse whitespace
    s = re.sub(r'\s+', ' ', s)
    # Whisper sometimes outputs "9 9" instead of "99" — collapse single-
    # digit runs back together where they look like a number.
    s = re.sub(r'(\d)\s+(?=\d\b)', r'\1', s)
    return s.strip()


def _has_exclusion(normalized: str) -> bool:
    return any(excl in normalized for excl in ASR_EXCLUSIONS)


def count_marker_hits(transcript: str) -> int:
    """Count the number of distinct ad-marker matches in `transcript`.

    Distinct = each marker phrase counts at most once per transcript so a
    transcript with one repeated phrase doesn't inflate the score. The
    intent is "how many independent ad-flavored signals did we hear,"
    not "how loud was the ad copy."

    Returns 0 for transcripts that match an exclusion phrase OR have
    fewer than ~3 alpha words (silence / whisper hallucination on
    music-only audio).
    """
    if not transcript:
        return 0
    normalized = _normalize_for_match(transcript)
    if not normalized:
        return 0

    # Require enough actual transcribed content to score. Whisper sometimes
    # hallucinates short phrases like "you" or "thank you" on
    # silence/music — those shouldn't accrue marker hits even by accident.
    alpha_words = re.findall(r'[a-z]{2,}', normalized)
    if len(alpha_words) < 3:
        return 0

    if _has_exclusion(normalized):
        return 0

    hits = 0
    for marker in ASR_CTA_MARKERS:
        if marker in normalized:
            hits += 1
    for compiled, _label in _COMPILED_REGEX:
        if compiled.search(normalized):
            hits += 1
    return hits


def explain_hits(transcript: str) -> list:
    """Same as count_marker_hits but returns the list of matched markers
    (for debugging / log output / future training-data review)."""
    if not transcript:
        return []
    normalized = _normalize_for_match(transcript)
    if not normalized:
        return []
    alpha_words = re.findall(r'[a-z]{2,}', normalized)
    if len(alpha_words) < 3 or _has_exclusion(normalized):
        return []
    hits = []
    for marker in ASR_CTA_MARKERS:
        if marker in normalized:
            hits.append(marker)
    for compiled, label in _COMPILED_REGEX:
        m = compiled.search(normalized)
        if m:
            hits.append(f"{label}:{m.group(0)}")
    return hits
