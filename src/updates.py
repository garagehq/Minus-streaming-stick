"""
Release / update checks for Minus: the code, ustreamer, and the VLM model.

Settings -> Updates in the web UI shows, for each component, what is installed
and what is available, and offers one "Update now" button.

Code (the Minus checkout itself) and ustreamer (a separate checkout built from
source) are git repos. For each:
  - On a published release (HEAD exactly at a `v*` tag): compare with the
    latest GitHub release of that repo.
  - Running from source on a branch that tracks an upstream (e.g. `main` ->
    `origin/main`): `git fetch`, then count commits behind/ahead.
  - On a branch with no upstream (a local deploy/dev branch): compare with
    `origin/<default branch>` for information only; "Update now" never touches
    it (it can't know what that branch should become).
  Updating is fast-forward only and refuses when tracked files have local
  changes, so it never discards anyone's work.

The VLM model is described by models/vlm_model.json in the repo (Hugging Face
repo + subfolder + local directory). The installed copy is compared with the
Hugging Face files by checksum, not by commit, so README edits on the model repo
are not "updates". Moving to a new model (e.g. Minus v0.2) is a manifest change
in the code: after a code update the check sees the new model isn't installed
and offers to download it. Downloads go to a staging directory, are verified by
sha256, and replace the old directory in one rename (the old one is kept as
`<dir>.prev`).

Everything here is stdlib + git + HTTPS (no new dependencies on the box).
"""

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_DIR = Path(__file__).resolve().parent.parent
MODEL_MANIFEST = REPO_DIR / 'models' / 'vlm_model.json'
USTREAMER_DIR = Path(os.environ.get('MINUS_USTREAMER_SRC', '/home/radxa/ustreamer-garagehq'))
STATUS_FILE = Path(os.environ.get('MINUS_UPDATE_STATUS_FILE', '/var/lib/minus/update_status.json'))
MARKER_NAME = '.minus_model.json'      # written into a model dir once verified
HF_ENDPOINT = os.environ.get('MINUS_HF_ENDPOINT', 'https://huggingface.co')
GITHUB_API = os.environ.get('MINUS_GITHUB_API', 'https://api.github.com')
CHECK_CACHE_S = float(os.environ.get('MINUS_UPDATE_CHECK_INTERVAL_S', '3600'))
HTTP_TIMEOUT = 20
RELEASE_TAG_RE = re.compile(r'^v\d')


# ----------------------------------------------------------------------- git
def _as_owner(path):
    """Prefix that runs a command as the owner of `path` when we are root and
    they aren't. Minus runs as root but the checkouts belong to the user who
    cloned them: as root git refuses them ("dubious ownership"), root has no
    GitHub SSH key for `git@github.com:` remotes, and files git writes would end
    up root-owned. Running git as the owner avoids all three."""
    try:
        if os.geteuid() != 0:
            return []
        import pwd
        uid = Path(path).stat().st_uid
        if uid == 0:
            return []
        return ['runuser', '-u', pwd.getpwuid(uid).pw_name, '--']
    except Exception:
        return []


def _chown_like_parent(path):
    """Give a tree we created as root to the owner of its parent directory."""
    try:
        if os.geteuid() != 0:
            return
        st = Path(path).parent.stat()
        for root, dirs, files in os.walk(path):
            os.chown(root, st.st_uid, st.st_gid)
            for f in files:
                os.chown(os.path.join(root, f), st.st_uid, st.st_gid)
    except Exception as e:
        logger.debug(f'chown {path}: {e}')


def _git(repo, *args, timeout=60, check=True):
    r = subprocess.run([*_as_owner(repo), 'git', '-C', str(repo), *args],
                       capture_output=True, text=True, timeout=timeout)
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {r.stderr.strip() or r.stdout.strip()}")
    return r.stdout.strip()


def _github_slug(repo):
    """'owner/name' from the origin URL (ssh or https), or None."""
    try:
        url = _git(repo, 'remote', 'get-url', 'origin')
    except Exception:
        return None
    m = re.search(r'github\.com[:/]([^/]+/[^/]+?)(?:\.git)?/?$', url)
    return m.group(1) if m else None


def _http_json(url, timeout=HTTP_TIMEOUT):
    req = urllib.request.Request(url, headers={'User-Agent': 'Minus-updater',
                                               'Accept': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode('utf-8'))


def latest_github_release(slug):
    """{'tag', 'name', 'url', 'published_at'} of the latest published release, or None."""
    if not slug:
        return None
    try:
        d = _http_json(f'{GITHUB_API}/repos/{slug}/releases/latest')
        return {'tag': d.get('tag_name'), 'name': d.get('name') or d.get('tag_name'),
                'url': d.get('html_url'), 'published_at': d.get('published_at')}
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None     # no releases published yet
        raise


def check_git_repo(repo, name, default_branch='main', fetch=True):
    """Status of one git checkout. Never raises; problems land in 'error'."""
    repo = Path(repo)
    st = {'name': name, 'path': str(repo), 'kind': 'git', 'up_to_date': None,
          'updatable': False, 'reason': None, 'error': None}
    try:
        if not (repo / '.git').exists():
            st['error'] = 'not a git checkout'
            return st
        head = _git(repo, 'rev-parse', 'HEAD')
        st['commit'] = head[:7]
        st['commit_date'] = _git(repo, 'log', '-1', '--format=%cI')
        st['subject'] = _git(repo, 'log', '-1', '--format=%s')[:120]
        branch = _git(repo, 'rev-parse', '--abbrev-ref', 'HEAD')
        st['branch'] = None if branch == 'HEAD' else branch
        tag = _git(repo, 'describe', '--tags', '--exact-match', 'HEAD', check=False)
        st['tag'] = tag if RELEASE_TAG_RE.match(tag or '') else None
        dirty = _git(repo, 'status', '--porcelain', '--untracked-files=no')
        st['local_changes'] = bool(dirty)

        if fetch:
            _git(repo, 'fetch', '--quiet', '--tags', 'origin', timeout=120)

        slug = _github_slug(repo)
        st['github'] = slug
        try:
            release = latest_github_release(slug) if slug else None
        except Exception as e:
            if st['tag']:
                raise                 # a release install needs it
            release = None            # from source it's only informational
            st['release_check_error'] = str(e)[:200]
        st['latest_release'] = release

        if st['tag']:
            # A published release: is there a newer one?
            st['mode'] = 'release'
            if release and release['tag']:
                st['latest'] = release['tag']
                st['up_to_date'] = (release['tag'] == st['tag'])
                st['updatable'] = not st['up_to_date'] and not st['local_changes']
                if st['local_changes'] and not st['up_to_date']:
                    st['reason'] = 'local changes to tracked files'
            else:
                st['latest'] = st['tag']
                st['up_to_date'] = True
            return st

        st['mode'] = 'source'
        upstream = _git(repo, 'rev-parse', '--abbrev-ref', '--symbolic-full-name', '@{u}',
                        check=False) if st['branch'] else ''
        if upstream and not upstream.startswith('fatal'):
            st['upstream'] = upstream
            target = upstream
        else:
            st['upstream'] = None
            target = f'origin/{default_branch}'
        st['target'] = target
        behind, ahead = (int(x) for x in _git(repo, 'rev-list', '--left-right', '--count',
                                              f'{target}...HEAD').split())
        st['behind'], st['ahead'] = behind, ahead
        st['latest'] = _git(repo, 'rev-parse', '--short=7', target)
        st['latest_subject'] = _git(repo, 'log', '-1', '--format=%s', target)[:120]
        st['up_to_date'] = behind == 0
        if not st['upstream']:
            st['reason'] = (f"local branch '{st['branch'] or 'detached'}' has no upstream — "
                            f"compared with {target} for information; update it by hand")
        elif behind and ahead:
            st['reason'] = f'diverged from {target} ({ahead} local commits) — update by hand'
        elif behind and st['local_changes']:
            st['reason'] = 'local changes to tracked files — update by hand'
        st['updatable'] = bool(behind) and st['reason'] is None
    except Exception as e:
        st['error'] = str(e)[:300]
    return st


def update_git_repo(repo, status):
    """Bring one checkout up to date per its check status. Returns (old, new) commit."""
    repo = Path(repo)
    old = _git(repo, 'rev-parse', 'HEAD')
    if status.get('mode') == 'release':
        tag = status['latest']
        _git(repo, 'fetch', '--quiet', '--tags', 'origin', timeout=120)
        _git(repo, 'checkout', '--quiet', tag)
    else:
        _git(repo, 'merge', '--ff-only', '--quiet', status['target'])
    return old, _git(repo, 'rev-parse', 'HEAD')


# --------------------------------------------------------------------- model
def load_manifest(path=MODEL_MANIFEST):
    with open(path) as f:
        m = json.load(f)
    m.setdefault('revision', 'main')
    m.setdefault('subfolder', '')
    env_dir = os.environ.get('MINUS_VLM_MODEL_DIR')
    m['local_dir'] = env_dir or m['local_dir']
    return m


def hf_model_files(manifest, timeout=HTTP_TIMEOUT):
    """{relative path: {'sha256', 'size'}} of the model's files on Hugging Face,
    plus the commit sha. Paths are relative to the manifest subfolder."""
    url = (f"{HF_ENDPOINT}/api/models/{manifest['hf_repo']}/revision/"
           f"{manifest['revision']}?blobs=true")
    d = _http_json(url, timeout=timeout)
    prefix = (manifest['subfolder'].strip('/') + '/') if manifest['subfolder'] else ''
    files = {}
    for s in d.get('siblings', []):
        path = s.get('rfilename', '')
        if not path.startswith(prefix):
            continue
        rel = path[len(prefix):]
        if not rel or rel.lower() == 'readme.md':
            continue
        lfs = s.get('lfs') or {}
        files[rel] = {'sha256': lfs.get('sha256'), 'size': lfs.get('size', s.get('size')),
                      'blob': s.get('blobId')}
    return d.get('sha'), files


def _sha256(path, bufsize=4 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _git_blob_sha(path):
    data = Path(path).read_bytes()
    return hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()


def _file_matches(path, want):
    if not path.exists():
        return False
    if want.get('size') is not None and path.stat().st_size != want['size']:
        return False
    if want.get('sha256'):
        return _sha256(path) == want['sha256']
    if want.get('blob'):
        return _git_blob_sha(path) == want['blob']
    return True


def check_model(manifest=None, remote=None):
    """Status of the installed VLM model vs Hugging Face. Never raises.

    The first check on a box hashes the installed files once (~10 s for 740 MB)
    and records the result in <local_dir>/.minus_model.json; later checks just
    compare that record with Hugging Face."""
    st = {'name': 'VLM model', 'kind': 'model', 'up_to_date': None, 'updatable': False,
          'reason': None, 'error': None}
    try:
        manifest = manifest or load_manifest()
        st.update({'model': manifest['name'], 'hf_repo': manifest['hf_repo'],
                   'subfolder': manifest['subfolder'], 'path': manifest['local_dir']})
        sha, files = remote or hf_model_files(manifest)
        st['latest'] = (sha or '')[:7]
        if not files:
            st['error'] = 'no model files found on Hugging Face for this manifest'
            return st
        local = Path(manifest['local_dir'])
        marker = local / MARKER_NAME
        rec = None
        if marker.exists():
            try:
                rec = json.loads(marker.read_text())
            except Exception:
                rec = None
        same_model = rec and rec.get('hf_repo') == manifest['hf_repo'] and \
            rec.get('subfolder') == manifest['subfolder']
        if not same_model and local.exists():
            # Never verified (installed by hand) or a different model: check the
            # files once, and adopt them if they already match.
            if all(_file_matches(local / rel, want) for rel, want in files.items()):
                rec = {'hf_repo': manifest['hf_repo'], 'subfolder': manifest['subfolder'],
                       'name': manifest['name'], 'commit': sha, 'files': files,
                       'verified_at': time.time()}
                try:
                    marker.write_text(json.dumps(rec, indent=1))
                    if os.geteuid() == 0:
                        dst = local.stat()
                        os.chown(marker, dst.st_uid, dst.st_gid)
                except OSError:
                    pass
                same_model = True
        if not local.exists():
            st['installed'] = None
            st['changed_files'] = sorted(files)
        elif not same_model:
            st['installed'] = (rec or {}).get('name') or 'unknown / different files'
            st['changed_files'] = sorted(rel for rel, want in files.items()
                                         if not _file_matches(local / rel, want))
        else:
            st['installed'] = rec.get('name') or manifest['name']
            st['installed_commit'] = (rec.get('commit') or '')[:7]
            st['changed_files'] = sorted(rel for rel, want in files.items()
                                         if (rec.get('files') or {}).get(rel) != want)
        st['download_bytes'] = sum((files[r].get('size') or 0) for r in st['changed_files'])
        st['up_to_date'] = not st['changed_files']
        st['updatable'] = not st['up_to_date']
    except Exception as e:
        st['error'] = str(e)[:300]
    return st


def _download(url, dest, want, progress=None, timeout=60):
    tmp = dest.with_name(dest.name + '.part')
    tmp.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={'User-Agent': 'Minus-updater'})
    h = hashlib.sha256()
    with urllib.request.urlopen(req, timeout=timeout) as r, open(tmp, 'wb') as f:
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b)
            h.update(b)
            if progress:
                progress(len(b))
    if want.get('sha256') and h.hexdigest() != want['sha256']:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f'checksum mismatch for {dest.name}')
    os.replace(tmp, dest)


def update_model(manifest=None, remote=None, log=print):
    """Download the model per the manifest into a staging dir, verify, swap.
    Unchanged files are copied from the installed dir instead of downloaded."""
    manifest = manifest or load_manifest()
    sha, files = remote or hf_model_files(manifest)
    local = Path(manifest['local_dir'])
    stage = local.with_name(local.name + '.staging')
    prev = local.with_name(local.name + '.prev')
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    prefix = (manifest['subfolder'].strip('/') + '/') if manifest['subfolder'] else ''
    total = sum((w.get('size') or 0) for w in files.values())
    done = [0]
    for rel, want in sorted(files.items()):
        dest = stage / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        old = local / rel
        if old.exists() and _file_matches(old, want):
            shutil.copy2(old, dest)
            done[0] += want.get('size') or 0
            continue
        url = f"{HF_ENDPOINT}/{manifest['hf_repo']}/resolve/{sha}/{prefix}{rel}"
        log(f'downloading {rel} ({(want.get("size") or 0) / 1e6:.0f} MB)')
        _download(url, dest, want, progress=lambda n: done.__setitem__(0, done[0] + n))
    for rel, want in files.items():
        if not _file_matches(stage / rel, want):
            raise RuntimeError(f'verification failed for {rel}')
    # keep anything in the old dir the manifest doesn't know (scripts, logs)
    if local.exists():
        for p in local.rglob('*'):
            rel = p.relative_to(local)
            if p.is_file() and not (stage / rel).exists() and p.name != MARKER_NAME:
                (stage / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, stage / rel)
    (stage / MARKER_NAME).write_text(json.dumps(
        {'hf_repo': manifest['hf_repo'], 'subfolder': manifest['subfolder'], 'name': manifest['name'],
         'commit': sha, 'files': files, 'verified_at': time.time()}, indent=1))
    _chown_like_parent(stage)
    if prev.exists():
        shutil.rmtree(prev)
    if local.exists():
        os.replace(local, prev)
    os.replace(stage, local)
    log(f'model {manifest["name"]} installed at {local} (previous copy kept at {prev})')
    return {'commit': sha, 'bytes': total}


# ------------------------------------------------------------------- checker
class UpdateChecker:
    """Cached checks for the web UI + launcher for the detached updater."""

    def __init__(self, repo_dir=None, ustreamer_dir=None, status_file=None):
        self.repo_dir = Path(repo_dir or REPO_DIR)
        self.ustreamer_dir = Path(ustreamer_dir or USTREAMER_DIR)
        self.status_file = Path(status_file or STATUS_FILE)
        self._lock = threading.Lock()
        self._result = None
        self._checked_at = 0.0
        self._checking = False

    def check(self, force=False):
        with self._lock:
            fresh = self._result and time.time() - self._checked_at < CHECK_CACHE_S
            if (fresh and not force) or self._checking:
                return self._result or {'checking': True}
            self._checking = True
        try:
            comps = [check_git_repo(self.repo_dir, 'Minus', default_branch='main')]
            if (self.ustreamer_dir / '.git').exists():
                comps.append(check_git_repo(self.ustreamer_dir, 'ustreamer', default_branch='master'))
            if MODEL_MANIFEST.exists():
                comps.append(check_model())
            result = {
                'checked_at': time.time(),
                'components': comps,
                'up_to_date': all(c.get('up_to_date') is not False for c in comps),
                'updatable': any(c.get('updatable') for c in comps),
                'errors': [f"{c['name']}: {c['error']}" for c in comps if c.get('error')],
            }
            with self._lock:
                self._result, self._checked_at = result, time.time()
            return result
        finally:
            with self._lock:
                self._checking = False

    def check_in_background(self, force=False):
        threading.Thread(target=self.check, kwargs={'force': force}, daemon=True,
                         name='update-check').start()

    def last(self):
        return self._result

    def run_status(self):
        try:
            return json.loads(self.status_file.read_text())
        except Exception:
            return None

    def start_update(self):
        """Launch tools/minus_update.py detached from this process (it restarts
        Minus, which kills this web server). systemd-run when available, so it
        survives the service restart; a new session otherwise."""
        st = self.run_status()
        if st and st.get('state') == 'running' and time.time() - st.get('updated_at', 0) < 1800:
            return {'success': False, 'error': 'an update is already running'}
        script = self.repo_dir / 'tools' / 'minus_update.py'
        cmd = ['/usr/bin/python3', str(script), '--status-file', str(self.status_file)]
        self.status_file.parent.mkdir(parents=True, exist_ok=True)
        self.status_file.write_text(json.dumps({'state': 'starting', 'updated_at': time.time(),
                                                'steps': []}))
        if shutil.which('systemd-run') and os.geteuid() == 0:
            unit = f'minus-update-{int(time.time())}'
            subprocess.run(['systemd-run', '--unit', unit, '--collect', '--quiet', *cmd],
                           check=True, timeout=20)
        else:
            subprocess.Popen(cmd, start_new_session=True, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
        return {'success': True}
