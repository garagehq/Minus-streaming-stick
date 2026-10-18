"""Which Spanish word to show during a block, and which of its examples.

Picking with random.choice() repeats words far sooner than the list size
suggests: with ~800 words a repeat is more likely than not within ~33 blocks,
so a household watching a few hours a day saw the same words within a day or
two. A shuffle bag shows every word once before any word comes back, and the
bag is saved so a restart doesn't reset it. Each showing picks two of the
word's example sentences at random, so a word that does come back reads
differently.
"""

import json
import os
import random
import threading
import unicodedata

DEFAULT_BAG_PATH = os.environ.get('MINUS_VOCAB_BAG', '/var/lib/minus/vocab_bag.json')
EXAMPLES_SHOWN = 2


def pick_examples(examples, k=EXAMPLES_SHOWN, rng=random):
    examples = [e for e in examples if e]
    if len(examples) <= k:
        return list(examples)
    return rng.sample(examples, k)


class VocabBag:
    """Every entry once, in random order, before any repeats. Persisted."""

    def __init__(self, entries, path=DEFAULT_BAG_PATH, rng=None):
        self._rng = rng or random.Random()
        self._by_key = {}
        for e in entries:
            self._by_key.setdefault(e[0], e)    # first wins on a duplicate headword
        self._path = path
        self._lock = threading.Lock()
        self._remaining, self._seen, self._last = [], set(), None
        self._load()

    def _load(self):
        state = {}
        if self._path:
            try:
                with open(self._path, encoding='utf-8') as f:
                    state = json.load(f)
            except (OSError, ValueError):
                state = {}
        keys = set(self._by_key)
        self._remaining = [k for k in state.get('remaining', []) if k in keys]
        self._seen = {k for k in state.get('seen', []) if k in keys} - set(self._remaining)
        self._last = state.get('last')
        # Words added since the bag was saved join the unseen pile.
        new = [k for k in self._by_key if k not in self._seen and k not in set(self._remaining)]
        for k in new:
            self._remaining.insert(self._rng.randrange(len(self._remaining) + 1), k)

    def _save(self):
        if not self._path:
            return
        try:
            os.makedirs(os.path.dirname(self._path), exist_ok=True)
            tmp = self._path + '.tmp'
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump({'remaining': self._remaining, 'seen': sorted(self._seen),
                           'last': self._last}, f, ensure_ascii=False)
            os.replace(tmp, self._path)
        except OSError:
            pass   # read-only or missing dir: the bag still works in memory

    def next(self):
        with self._lock:
            if not self._remaining:
                order = list(self._by_key)
                self._rng.shuffle(order)
                # pop() takes from the end: don't open a new round with the
                # word that closed the last one.
                if len(order) > 1 and order[-1] == self._last:
                    order[0], order[-1] = order[-1], order[0]
                self._remaining, self._seen = order, set()
            key = self._remaining.pop()
            self._seen.add(key)
            self._last = key
            self._save()
            return self._by_key[key]

    def __len__(self):
        return len(self._by_key)

    @property
    def remaining(self):
        return len(self._remaining)


# --- overlay text encoding -------------------------------------------------
#
# ustreamer draws the overlay one BYTE at a time (blocking.c:
# FT_Get_Char_Index(face, (FT_ULong)c)), and char is unsigned on aarch64, so
# each byte is taken as a code point. UTF-8 "ñ" (C3 B1) therefore drew as
# "Ã±"; Latin-1 bytes are exactly the code points U+0000-U+00FF, so sending
# Latin-1 makes á é í ó ú ñ ü ¿ ¡ draw correctly with no ustreamer change.

_REPLACE = {'—': '-', '–': '-', '‘': "'", '’': "'",
            '“': '"', '”': '"', '…': '...', ' ': ' '}


def latin1_safe(text: str) -> str:
    """`text` reduced to characters ustreamer can draw (Latin-1)."""
    if not isinstance(text, str):
        return text
    try:
        text.encode('latin-1')
        return text
    except UnicodeEncodeError:
        pass
    out = []
    for ch in text:
        ch = _REPLACE.get(ch, ch)
        try:
            ch.encode('latin-1')
            out.append(ch)
            continue
        except UnicodeEncodeError:
            pass
        base = unicodedata.normalize('NFKD', ch).encode('latin-1', 'ignore').decode('latin-1')
        out.append(base if base else '?')
    return ''.join(out)
