#!/usr/bin/env python3
"""Check the ASR dataset's labels against an independent ASR and fix them.

The labels are YouTube captions (tools/label_clips.py), kept when enough of
their words also appear in what SenseVoice heard. This pass re-transcribes
every clip with Parakeet TDT 0.6B (a different model from the one that
filtered them), aligns the label to that transcript and finds:

  * audible words at the clip edges the label leaves out (the labeller's
    lead/tail pads catch the neighbouring words): the audio is trimmed to
    the words the label covers;
  * caption misspellings of the name ("jane" for "james", "brawny" for
    "bronny"): fixed to what Parakeet heard;
  * labels that don't match the audio (garbled captions, wrong timing):
    removed. A clip where only the name is missed by Parakeet is kept,
    since those are the hard examples training needs.

usage: python3 tools/audit_dataset.py transcribe [--workers 2 --threads 4] [--limit N]
       python3 tools/audit_dataset.py review       # decisions + report, changes nothing
       python3 tools/audit_dataset.py verify       # re-transcribe the proposed trims
       python3 tools/audit_dataset.py review       # again, now with checked trims
       python3 tools/audit_dataset.py apply [--keep-untrimmed]   # manifest backup, trims, quarantine

Transcripts go to DATASET_DIR/audit/parakeet.jsonl (resumable). Removed
clips move to DATASET_DIR/audit/quarantine/ and the manifest is backed up.
Trims are only applied once Parakeet confirms the trimmed audio matches the
label better; each trimmed row keeps its trim span and original duration
under 'audit'. The original audio of trimmed clips (about half the dataset)
is copied to DATASET_DIR/audit/untrimmed/ only with --keep-untrimmed.
"""
import argparse
import collections
import json
import multiprocessing as mp
import os
import random
import re
import shutil
import sys
import time
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import label_clips as lc

PARAKEET_DIR = os.environ.get(
    'AUDIT_PARAKEET_DIR',
    '/home/radxa/asr_models/parakeet/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8')
AUDIT_DIR = lc.DATASET_DIR / 'audit'
SR = 16000

# A label is removed when, leaving out edge words, more than this fraction
# of its characters disagree with Parakeet ...
DROP_CER = 0.5
# ... or more than this and SenseVoice also had doubts when labelling.
DROP_CER_LOW_OVERLAP, LOW_OVERLAP = 0.3, 0.8
# Caption words this far past a relabelled clip's edges are still offered to
# the alignment (spread line timings are approximate).
RELABEL_SLACK_S = 0.5
MIN_WORDS, MIN_SECONDS = 3, 1.0
# The closest a cut may come to a word Parakeet heard.
CUT_GUARD_S = 0.02
# Label words corrected to what Parakeet heard when they are close in
# spelling (caption mishearings of the names).
NAME_FIXES = {'lebron', 'james', 'bronny'}
# Already the name as spoken: Parakeet often drops a possessive "'s", and
# "bron" is a nickname announcers really say.
NAME_FORMS = {"lebron's", 'lebrons', "james'", "james's", "bronny's", 'bron', "bron's"}
NAME_FIX_MAX_CHARS = 3


# ---------------------------------------------------------------- transcribe

_rec = None


def _init(threads):
    global _rec
    import sherpa_onnx as so
    d = PARAKEET_DIR
    _rec = so.OfflineRecognizer.from_transducer(
        encoder=f'{d}/encoder.int8.onnx', decoder=f'{d}/decoder.int8.onnx',
        joiner=f'{d}/joiner.int8.onnx', tokens=f'{d}/tokens.txt',
        num_threads=threads, model_type='nemo_transducer')


def read_wav(path):
    with wave.open(str(path)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), np.int16)


def words_from_tokens(tokens, starts, durs):
    """[(word, start_s, end_s)] in normalize()d form from Parakeet tokens."""
    out, cur = [], None
    for tok, t, d in zip(tokens, starts, durs):
        if tok.startswith(' ') or cur is None:
            if cur:
                out.append(cur)
            cur = [tok.strip(), t, t + d]
        else:
            cur[0] += tok
            cur[2] = t + d
    if cur:
        out.append(cur)
    words = []
    for w, a, b in out:
        for part in lc.normalize(w).split():     # "two-year" -> two, year
            words.append((part, round(a, 3), round(b, 3)))
    return words


def _transcribe(path):
    a = read_wav(path).astype(np.float32) / 32768
    s = _rec.create_stream()
    s.accept_waveform(SR, a)
    _rec.decode_stream(s)
    r = s.result
    return {'name': Path(path).stem, 'text': r.text,
            'words': words_from_tokens(r.tokens, r.timestamps, r.durations)}


def transcribe(workers, threads, limit=None):
    AUDIT_DIR.mkdir(exist_ok=True)
    out = AUDIT_DIR / 'parakeet.jsonl'
    done = set()
    if out.exists():
        for line in open(out):
            try:
                done.add(json.loads(line)['name'])
            except ValueError:
                pass                              # torn last line from a kill
    todo = [r['audio_filepath'] for r in manifest()
            if Path(r['audio_filepath']).stem not in done]
    random.Random(0).shuffle(todo)          # so --limit gives a random sample
    if limit:
        todo = todo[:limit]
    print(f'{len(done)} done, {len(todo)} to transcribe', flush=True)
    t0, n = time.time(), 0
    with mp.Pool(workers, _init, (threads,)) as pool, open(out, 'a') as fh:
        for res in pool.imap_unordered(_transcribe, todo, chunksize=8):
            fh.write(json.dumps(res) + '\n')
            n += 1
            if n % 500 == 0:
                fh.flush()
                rate = n / (time.time() - t0)
                print(f'{n}/{len(todo)} clips, {rate:.1f}/s, '
                      f'~{(len(todo) - n) / rate / 3600:.1f} h left', flush=True)
    print(f'transcribed {n} clips in {(time.time() - t0) / 3600:.2f} h')


def _transcribe_span(job):
    path, a, b = job
    x = read_wav(path)[int(a * SR):int(b * SR)].astype(np.float32) / 32768
    s = _rec.create_stream()
    s.accept_waveform(SR, x)
    _rec.decode_stream(s)
    return {'name': Path(path).stem, 'trim': [a, b], 'text': s.result.text}


def verify(workers, threads):
    """Re-transcribe each proposed trim (from the last review), so review
    keeps only trims that bring Parakeet closer to the label."""
    decisions = json.load(open(AUDIT_DIR / 'decisions.json'))
    out = AUDIT_DIR / 'parakeet_trimmed.jsonl'
    done = set()
    if out.exists():
        for t in _read_jsonl(out):
            done.add((t['name'], tuple(t['trim'])))
    paths = {Path(r['audio_filepath']).stem: r['audio_filepath'] for r in manifest()}
    jobs = [(paths[n], *d['trim']) for n, d in decisions.items()
            if d.get('trim') and n in paths and (n, tuple(d['trim'])) not in done]
    print(f'{len(jobs)} trims to verify', flush=True)
    t0 = time.time()
    with mp.Pool(workers, _init, (threads,)) as pool, open(out, 'a') as fh:
        for n, res in enumerate(pool.imap_unordered(_transcribe_span, jobs, chunksize=8), 1):
            fh.write(json.dumps(res) + '\n')
            if n % 500 == 0:
                fh.flush()
                rate = n / (time.time() - t0)
                print(f'{n}/{len(jobs)} trims, {rate:.1f}/s, '
                      f'~{(len(jobs) - n) / rate / 3600:.1f} h left', flush=True)


def _read_jsonl(path):
    with open(path) as fh:
        for line in fh:
            try:
                yield json.loads(line)
            except ValueError:
                pass                              # torn last line from a kill


# ---------------------------------------------------------------- review

def align(ref, hyp):
    """Levenshtein alignment: [(op, ref_idx, hyp_idx)], op in = s d i."""
    n, m = len(ref), len(hyp)
    d = np.zeros((n + 1, m + 1), np.int32)
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1,
                          d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    ops, i, j = [], n, m
    while i or j:
        if i and j and d[i, j] == d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            ops.append(('=' if ref[i - 1] == hyp[j - 1] else 's', i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i and d[i, j] == d[i - 1, j] + 1:
            ops.append(('d', i - 1, None))
            i -= 1
        else:
            ops.append(('i', None, j - 1))
            j -= 1
    return ops[::-1]


def lev(a, b):
    """Levenshtein distance between two sequences (numpy rows, O(len(a)) steps)."""
    if not a or not b:
        return max(len(a), len(b))
    b = np.frombuffer(b.encode(), np.uint8) if isinstance(b, str) else np.asarray(b)
    ar = np.arange(len(b) + 1)
    row = ar.copy()
    for k, x in enumerate(a.encode() if isinstance(a, str) else a, 1):
        new = np.empty_like(row)
        new[0] = k
        new[1:] = np.minimum(row[1:] + 1, row[:-1] + (b != x))
        row = np.minimum.accumulate(new - ar) + ar       # insertions
    return int(row[-1])


def decide(row, tr, relabelled=False):
    """What to do with one manifest row given its Parakeet transcript.

    relabelled: row['text'] was rebuilt from corrected caption times, so
    label words at its edges that Parakeet didn't hear are dropped (their
    audio may lie outside the clip)."""
    label = row['text'].split()
    hw = tr['words']
    hyp = [w[0] for w in hw]
    if not label:
        return {'action': 'drop', 'why': 'empty label'}
    if not hyp:
        # Parakeet sometimes returns nothing for clear speech (the same clip
        # transcribes with a little padding), so SenseVoice decides alone.
        if not relabelled and row.get('asr_overlap', 0) >= LOW_OVERLAP:
            return {'action': 'keep', 'why': 'parakeet heard nothing; sensevoice agrees'}
        return {'action': 'drop', 'why': 'parakeet heard nothing'}
    ops = align(label, hyp)
    if relabelled:
        while ops and ops[0][0] == 'd':
            ops.pop(0)
        while ops and ops[-1][0] == 'd':
            ops.pop()
        keep = sorted(i for _, i, _ in ops if i is not None)
        if len(keep) < MIN_WORDS:
            return {'action': 'drop', 'why': 'too few words after relabelling'}
        # Re-index the label to the kept words.
        lo = keep[0]
        label = label[lo:keep[-1] + 1]
        ops = [(op, None if i is None else i - lo, j) for op, i, j in ops]
    # Edge words: hyp-only words before the first / after the last label word.
    lead = 0
    while lead < len(ops) and ops[lead][0] == 'i':
        lead += 1
    trail = 0
    while trail < len(ops) - lead and ops[len(ops) - 1 - trail][0] == 'i':
        trail += 1
    core = ops[lead:len(ops) - trail]
    # Disagreement over the words both cover, in characters (so a near miss
    # on a name, "willard" for "lillard", counts as close), not counting
    # Parakeet missing the name itself: those clips are what fine-tuning is
    # for.
    both = [(op, i, j) for op, i, j in core
            if not (op in 'sd' and label[i] == 'lebron')]
    ls = ' '.join(label[i] for _, i, _ in both if i is not None)
    hs = ' '.join(hyp[j] for _, _, j in both if j is not None)
    cer = lev(ls, hs) / max(len(ls), 1)
    out = {'cer': round(cer, 3), 'lead': lead, 'trail': trail}
    ov = row.get('asr_overlap', 1.0)
    if cer > DROP_CER or (cer > DROP_CER_LOW_OVERLAP and ov < LOW_OVERLAP):
        return {**out, 'action': 'drop', 'why': 'label does not match audio'}

    label_name = any(lc.NAME_RE.search(w) for w in label)
    hyp_name = any(lc.NAME_RE.search(w) for w in hyp)
    # Name spelling fixes.
    fixed = list(label)
    fixes = []
    for k, (op, i, j) in enumerate(core):
        # Only where the words either side agree, so the alignment is sure
        # which word was misheard ("jab step" is not "james jabstep").
        anchored = all(core[x][0] == '=' for x in (k - 1, k + 1) if 0 <= x < len(core))
        if (op == 's' and anchored and hyp[j] in NAME_FIXES and label[i] not in NAME_FIXES | NAME_FORMS
                and lev(label[i], hyp[j]) <= NAME_FIX_MAX_CHARS):
            fixes.append([label[i], hyp[j]])
            fixed[i] = hyp[j]

    # Trim audible edge words the label doesn't cover.
    dur = row['duration']
    a, b = 0.0, dur
    if lead:
        first = hw[core[0][2]] if core and core[0][2] is not None else None
        extra_end = hw[lead - 1][2]
        if first:
            # As close to the extra word as possible: Parakeet's word times
            # are approximate, and a late cut clips the label's first word.
            a = min(extra_end + CUT_GUARD_S, first[1] - CUT_GUARD_S)
    if trail:
        last_hyp = next((j for op, i, j in reversed(core) if j is not None), None)
        extra_start = hw[len(hw) - trail][1]
        if last_hyp is not None:
            last_end = hw[last_hyp][2]
            b = max(extra_start - CUT_GUARD_S, last_end + CUT_GUARD_S)
    a, b = max(0.0, round(a, 3)), min(dur, round(b, 3))
    trimmed = a > 0.05 or b < dur - 0.05
    if b - a < MIN_SECONDS or len(fixed) < MIN_WORDS:
        return {**out, 'action': 'drop', 'why': 'too short after trimming'}
    text = ' '.join(fixed)
    changed = text != row['text']
    if not trimmed and not changed:
        return {**out, 'action': 'keep',
                'name_only_label': label_name and not hyp_name}
    return {**out, 'action': 'fix', 'trim': [a, b] if trimmed else None,
            'text': text if changed else None, 'name_fixes': fixes,
            'relabelled': relabelled, 'name_only_label': label_name and not hyp_name}


def line_caption_videos():
    """Videos with line-level json3 captions: their clips were labelled
    with compressed word times before 3be70e0."""
    out = set()
    for j in (lc.FARM_DIR / 'captions').glob('*.json3'):
        try:
            events = json.load(open(j)).get('events', [])
        except (ValueError, OSError):
            continue
        segs = [(sg.get('utf8') or '').split() for ev in events for sg in ev.get('segs') or []]
        segs = [s for s in segs if s]
        # Auto-captions are one word per segment, bar the odd ">> word".
        if segs and sum(len(s) > 1 for s in segs) > len(segs) / 2:
            out.add(j.stem)
    return out


def relabel(row):
    """The label for a clip's audio span from the (fixed) caption times."""
    v0 = row['video_start']
    v1 = v0 + row['duration']
    ws = [w for w in lc.caption_words(lc.FARM_DIR, row['video_id'])
          if w[1] >= v0 - RELABEL_SLACK_S and w[2] <= v1 + RELABEL_SLACK_S]
    return lc.normalize(' '.join(w[0] for w in ws))


def manifest():
    with open(lc.DATASET_DIR / 'manifest.jsonl') as fh:
        return [json.loads(line) for line in fh]


def load_transcripts():
    return {t['name']: t for t in _read_jsonl(AUDIT_DIR / 'parakeet.jsonl')}


def load_trim_checks():
    p = AUDIT_DIR / 'parakeet_trimmed.jsonl'
    return {(t['name'], tuple(t['trim'])): t['text'] for t in _read_jsonl(p)} if p.exists() else {}


def check_trim(d, label, before, checks, name):
    """Keep a proposed trim only if Parakeet on the trimmed audio is closer
    to the label than on the whole clip; unverified trims are not applied."""
    after = checks.get((name, tuple(d['trim'])))
    if after is None:
        return 'unverified'
    c0 = lev(label, lc.normalize(before)) / max(len(label), 1)
    c1 = lev(label, lc.normalize(after)) / max(len(label), 1)
    return 'kept' if c1 < c0 else 'rejected'


def review(verbose=True):
    rows, trs, checks = manifest(), load_transcripts(), load_trim_checks()
    decisions, counts = {}, collections.Counter()
    seconds = collections.Counter()
    by_video = collections.defaultdict(collections.Counter)
    line_videos = line_caption_videos()
    for r in rows:
        name = Path(r['audio_filepath']).stem
        if name not in trs:
            counts['not transcribed'] += 1
            continue
        if '__' in r.get('text_raw', ''):
            d = {'action': 'drop', 'why': 'censored word in captions'}
            decisions[name] = d
            counts[f"drop: {d['why']}"] += 1
            seconds['dropped'] += r['duration']
            continue
        new = relabel(r) if r['video_id'] in line_videos else r['text']
        if '[' in r.get('text_raw', '') and new == r['text']:
            # Sound tags ("[clears throat]") that reached the label as words.
            new = lc.normalize(re.sub(r'\[[^\]]*\]', ' ', r['text_raw']))
            if new != r['text']:
                counts['fix: sound tag removed'] += 1
                d = decide({**r, 'text': new}, trs[name])
                if d['action'] != 'drop':
                    d = {**d, 'action': 'fix', 'text': d.get('text') or new}
                decisions[name] = d
                by_video[r['video_id']][d['action']] += 1
                if d['action'] == 'drop':
                    counts[f"drop: {d['why']}"] += 1
                    seconds['dropped'] += r['duration']
                continue
        if new != r['text']:
            d = decide({**r, 'text': new}, trs[name], relabelled=True)
            if d['action'] != 'drop':
                text = d.get('text') or new
                d = {**d, 'action': 'fix', 'text': text, 'relabelled': True}
                counts['fix: relabelled from caption times'] += 1
        else:
            d = decide(r, trs[name])
        if d.get('trim'):
            d['trim_check'] = check_trim(d, d.get('text') or r['text'],
                                         trs[name]['text'], checks, name)
            counts[f"trim {d['trim_check']}"] += 1
        decisions[name] = d
        key = d['action'] if d['action'] != 'drop' else f"drop: {d['why']}"
        counts[key] += 1
        by_video[r['video_id']][d['action']] += 1
        if d['action'] == 'drop':
            seconds['dropped'] += r['duration']
        elif d.get('trim_check') == 'kept':
            seconds['trimmed off'] += r['duration'] - (d['trim'][1] - d['trim'][0])
        if d.get('name_fixes'):
            counts['fix: name spelling'] += 1
        if d.get('name_only_label'):
            counts['kept: name heard only by captions'] += 1
    with open(AUDIT_DIR / 'decisions.json', 'w') as fh:
        json.dump(decisions, fh)
    worst = sorted(((v['drop'] / sum(v.values()), k, sum(v.values()))
                    for k, v in by_video.items() if sum(v.values()) >= 20), reverse=True)[:10]
    report = {'clips': len(rows), 'counts': dict(counts),
              'hours': {k: round(v / 3600, 2) for k, v in seconds.items()},
              'videos_most_dropped': [(k, round(f, 2), n) for f, k, n in worst]}
    with open(AUDIT_DIR / 'report.json', 'w') as fh:
        json.dump(report, fh, indent=1)
    if verbose:
        print(json.dumps(report, indent=1))
    return decisions


# ---------------------------------------------------------------- apply

def apply(keep_untrimmed=False):
    decisions = review(verbose=False)
    ds = lc.DATASET_DIR
    stamp = time.strftime('%Y%m%d_%H%M%S')
    shutil.copy2(ds / 'manifest.jsonl', AUDIT_DIR / f'manifest.before_audit_{stamp}.jsonl')
    (AUDIT_DIR / 'quarantine').mkdir(exist_ok=True)
    if keep_untrimmed:
        (AUDIT_DIR / 'untrimmed').mkdir(exist_ok=True)
    kept, n = [], collections.Counter()
    dropped = open(AUDIT_DIR / 'quarantine' / 'manifest.jsonl', 'a')
    for r in manifest():
        p = Path(r['audio_filepath'])
        d = decisions.get(p.stem)
        trim = d.get('trim') if d and d.get('trim_check') == 'kept' else None
        if d is None or d['action'] == 'keep' or (d['action'] == 'fix' and not trim
                                                   and not d.get('text')):
            kept.append(r)
            continue
        if d['action'] == 'drop':
            if p.exists():
                shutil.move(str(p), str(AUDIT_DIR / 'quarantine' / p.name))
            dropped.write(json.dumps({**r, 'audit': d}) + '\n')
            n['dropped'] += 1
            continue
        if trim:
            a, b = trim
            x = read_wav(p)
            if keep_untrimmed:
                shutil.copy2(p, AUDIT_DIR / 'untrimmed' / p.name)
            untrimmed_s = r['duration']
            with wave.open(str(p), 'wb') as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(SR)
                w.writeframes(x[int(a * SR):int(b * SR)].tobytes())
            r['duration'] = round(b - a, 3)
            r['video_start'] = round(r['video_start'] + a, 3)
            n['trimmed'] += 1
        if d.get('text'):
            r['text'] = d['text']
            r['has_name'] = bool(lc.NAME_RE.search(d['text']))
            n['renamed'] += 1
        r['audit'] = {'parakeet_cer': d['cer'], 'trim': trim,
                      'name_fixes': d.get('name_fixes')}
        if trim:
            r['audit']['untrimmed_duration'] = untrimmed_s
        kept.append(r)
    dropped.close()
    tmp = ds / 'manifest.jsonl.tmp'
    with open(tmp, 'w') as fh:
        for r in kept:
            fh.write(json.dumps(r) + '\n')
    os.replace(tmp, ds / 'manifest.jsonl')
    print(dict(n), 'kept', len(kept))
    print(lc.summary(ds))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['transcribe', 'review', 'verify', 'apply'])
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--limit', type=int)
    ap.add_argument('--keep-untrimmed', action='store_true',
                    help='copy the original audio of trimmed clips to audit/untrimmed/')
    args = ap.parse_args()
    if args.cmd == 'transcribe':
        transcribe(args.workers, args.threads, args.limit)
    elif args.cmd == 'review':
        review()
    elif args.cmd == 'verify':
        verify(args.workers, args.threads)
    else:
        apply(args.keep_untrimmed)


if __name__ == '__main__':
    main()
