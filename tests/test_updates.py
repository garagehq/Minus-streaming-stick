"""Release / update checks (src/updates.py) and the updater (tools/minus_update.py).

Uses real git repos (a bare "origin" + a clone in a temp dir), fake commits and
tags, and a local fake Hugging Face server — nothing touches GitHub or HF.
"""
import hashlib
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
import updates  # noqa: E402

REAL_SUBPROCESS_RUN = subprocess.run


def git(repo, *args):
    return subprocess.run(['git', '-C', str(repo), *args], check=True, capture_output=True,
                          text=True).stdout.strip()


@pytest.fixture(autouse=True)
def real_clock(monkeypatch):
    """tests/test_vlm_decision_sim.py freezes the GLOBAL time.time at import;
    staleness/age logic here needs a real clock."""
    import time as _time
    from datetime import datetime
    monkeypatch.setattr(_time, 'time', lambda: datetime.now().timestamp())


@pytest.fixture(autouse=True)
def no_github(monkeypatch):
    """No GitHub calls unless a test sets releases."""
    monkeypatch.setattr(updates, 'latest_github_release', lambda slug: None)
    for k in ('GIT_AUTHOR_NAME', 'GIT_COMMITTER_NAME'):
        monkeypatch.setenv(k, 'Test')
    for k in ('GIT_AUTHOR_EMAIL', 'GIT_COMMITTER_EMAIL'):
        monkeypatch.setenv(k, 'test@example.com')


@pytest.fixture
def repos(tmp_path):
    """A bare origin with main + a clone on main tracking it."""
    origin = tmp_path / 'origin.git'
    seed = tmp_path / 'seed'
    subprocess.run(['git', 'init', '-q', '--bare', '-b', 'main', str(origin)], check=True)
    subprocess.run(['git', 'init', '-q', '-b', 'main', str(seed)], check=True)
    (seed / 'a.txt').write_text('one\n')
    git(seed, 'add', '.')
    git(seed, 'commit', '-q', '-m', 'first')
    git(seed, 'remote', 'add', 'origin', str(origin))
    git(seed, 'push', '-q', '-u', 'origin', 'main')
    box = tmp_path / 'box'
    subprocess.run(['git', 'clone', '-q', str(origin), str(box)], check=True)
    return origin, seed, box


def push_commit(seed, msg='fake commit', fname='a.txt'):
    p = seed / fname
    p.write_text(p.read_text() + msg + '\n' if p.exists() else msg + '\n')
    git(seed, 'add', '.')
    git(seed, 'commit', '-q', '-m', msg)
    git(seed, 'push', '-q', 'origin', 'HEAD')
    return git(seed, 'rev-parse', '--short=7', 'HEAD')


# ------------------------------------------------------------------ git: source
def test_up_to_date_on_main(repos):
    _, _, box = repos
    st = updates.check_git_repo(box, 'Minus')
    assert st['error'] is None
    assert st['mode'] == 'source' and st['branch'] == 'main' and st['upstream'] == 'origin/main'
    assert st['up_to_date'] is True and st['behind'] == 0 and st['updatable'] is False


def test_behind_main_is_updatable_and_update_fast_forwards(repos):
    _, seed, box = repos
    push_commit(seed, 'fake 1')
    new = push_commit(seed, 'fake 2')
    st = updates.check_git_repo(box, 'Minus')
    assert st['behind'] == 2 and st['up_to_date'] is False and st['updatable'] is True
    assert st['latest'] == new and st['latest_subject'] == 'fake 2'
    old, after = updates.update_git_repo(box, st)
    assert after.startswith(new) and old != after
    assert updates.check_git_repo(box, 'Minus')['up_to_date'] is True


def test_local_changes_block_the_update(repos):
    _, seed, box = repos
    push_commit(seed)
    (box / 'a.txt').write_text('edited on the box\n')
    st = updates.check_git_repo(box, 'Minus')
    assert st['behind'] == 1 and st['updatable'] is False
    assert 'local changes' in st['reason']


def test_untracked_files_do_not_block(repos):
    _, seed, box = repos
    push_commit(seed)
    (box / 'screenshot.png').write_text('x')
    assert updates.check_git_repo(box, 'Minus')['updatable'] is True


def test_diverged_branch_is_not_updatable(repos):
    _, seed, box = repos
    push_commit(seed, 'upstream')
    (box / 'b.txt').write_text('local\n')
    git(box, 'add', '.')
    git(box, 'commit', '-q', '-m', 'local work')
    st = updates.check_git_repo(box, 'Minus')
    assert st['behind'] == 1 and st['ahead'] == 1 and st['updatable'] is False
    assert 'diverged' in st['reason']


def test_branch_without_upstream_is_info_only(repos):
    _, seed, box = repos
    git(box, 'checkout', '-q', '-b', 'deploy/local')
    push_commit(seed)
    st = updates.check_git_repo(box, 'Minus')
    assert st['upstream'] is None and st['target'] == 'origin/main'
    assert st['behind'] == 1 and st['updatable'] is False
    assert 'no upstream' in st['reason']


def test_tracks_its_own_upstream_branch(repos):
    """ustreamer runs a feature branch tracking origin/<that branch>."""
    _, seed, box = repos
    git(seed, 'checkout', '-q', '-b', 'fix/thing')
    push_commit(seed, 'branch work')
    git(seed, 'push', '-q', '-u', 'origin', 'fix/thing')
    git(box, 'fetch', '-q')
    git(box, 'checkout', '-q', '-b', 'fix/thing', '--track', 'origin/fix/thing')
    push_commit(seed, 'more branch work')
    st = updates.check_git_repo(box, 'ustreamer', default_branch='master')
    assert st['upstream'] == 'origin/fix/thing' and st['behind'] == 1 and st['updatable'] is True


# ------------------------------------------------------------------ git: release
def test_release_mode_newer_release(repos, monkeypatch):
    _, seed, box = repos
    git(seed, 'tag', 'v0.1.0')
    git(seed, 'push', '-q', 'origin', 'v0.1.0')
    push_commit(seed, 'work for 0.2')
    git(seed, 'tag', 'v0.2.0')
    git(seed, 'push', '-q', 'origin', 'v0.2.0')
    git(box, 'fetch', '-q', '--tags')
    git(box, 'checkout', '-q', 'v0.1.0')
    monkeypatch.setattr(updates, '_github_slug', lambda repo: 'garagehq/Minus')
    monkeypatch.setattr(updates, 'latest_github_release',
                        lambda slug: {'tag': 'v0.2.0', 'name': 'Minus 0.2', 'url': 'u', 'published_at': 'x'})
    st = updates.check_git_repo(box, 'Minus')
    assert st['mode'] == 'release' and st['tag'] == 'v0.1.0' and st['latest'] == 'v0.2.0'
    assert st['up_to_date'] is False and st['updatable'] is True
    updates.update_git_repo(box, st)
    assert git(box, 'describe', '--tags', '--exact-match') == 'v0.2.0'
    assert updates.check_git_repo(box, 'Minus')['up_to_date'] is True


def test_release_mode_current(repos, monkeypatch):
    _, seed, box = repos
    git(seed, 'tag', 'v0.1.0')
    git(seed, 'push', '-q', 'origin', 'v0.1.0')
    git(box, 'fetch', '-q', '--tags')
    git(box, 'checkout', '-q', 'v0.1.0')
    monkeypatch.setattr(updates, '_github_slug', lambda repo: 'garagehq/Minus')
    monkeypatch.setattr(updates, 'latest_github_release',
                        lambda slug: {'tag': 'v0.1.0', 'name': 'x', 'url': 'u', 'published_at': 'x'})
    st = updates.check_git_repo(box, 'Minus')
    assert st['mode'] == 'release' and st['up_to_date'] is True and st['updatable'] is False


def test_github_outage_does_not_break_source_checks(repos, monkeypatch):
    _, seed, box = repos
    push_commit(seed)
    monkeypatch.setattr(updates, '_github_slug', lambda repo: 'garagehq/Minus')

    def boom(slug):
        raise OSError('rate limited')
    monkeypatch.setattr(updates, 'latest_github_release', boom)
    st = updates.check_git_repo(box, 'Minus')
    assert st['error'] is None and st['updatable'] is True and 'rate limited' in st['release_check_error']


def test_github_slug_parsing(repos):
    _, _, box = repos
    for url, want in (('git@github.com:garagehq/Minus-streaming-stick.git', 'garagehq/Minus-streaming-stick'),
                      ('https://github.com/garagehq/ustreamer.git', 'garagehq/ustreamer'),
                      ('https://github.com/garagehq/ustreamer', 'garagehq/ustreamer')):
        git(box, 'remote', 'set-url', 'origin', url)
        assert updates._github_slug(box) == want


def test_not_a_checkout(tmp_path):
    st = updates.check_git_repo(tmp_path, 'Minus')
    assert st['error'] == 'not a git checkout' and st['updatable'] is False


# ----------------------------------------------------------------------- model
class FakeHF:
    """/api/models/<repo>/revision/<rev>?blobs=true and /<repo>/resolve/<sha>/<path>."""

    def __init__(self, repo, files, sha='abc1234def'):
        self.repo, self.files, self.sha = repo, dict(files), sha
        self.downloads = []
        owner = self

        class H(BaseHTTPRequestHandler):
            def do_GET(self):
                path = self.path.split('?')[0]
                if path.startswith(f'/api/models/{owner.repo}/revision/'):
                    sib = [{'rfilename': 'README.md', 'size': 10, 'blobId': 'x'}]
                    for rel, data in owner.files.items():
                        sib.append({'rfilename': rel, 'size': len(data), 'blobId': 'b',
                                    'lfs': {'sha256': hashlib.sha256(data).hexdigest(), 'size': len(data)}})
                    body = json.dumps({'sha': owner.sha, 'siblings': sib}).encode()
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(body)
                    return
                prefix = f'/{owner.repo}/resolve/{owner.sha}/'
                if path.startswith(prefix) and path[len(prefix):] in owner.files:
                    owner.downloads.append(path[len(prefix):])
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(owner.files[path[len(prefix):]])
                    return
                self.send_response(404)
                self.end_headers()

            def log_message(self, *a):
                pass

        self.httpd = HTTPServer(('127.0.0.1', 0), H)
        self.url = f'http://127.0.0.1:{self.httpd.server_port}'
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()


@pytest.fixture
def hf(monkeypatch):
    files = {'axera/model.axmodel': b'weights-v1' * 100, 'axera/sub/layer.axmodel': b'layer' * 50,
             'axera/tokenizer.json': b'{"t": 1}', 'other/ignored.bin': b'nope'}
    srv = FakeHF('Org/Minus-v0.1', files)
    monkeypatch.setattr(updates, 'HF_ENDPOINT', srv.url)
    yield srv
    srv.close()


def manifest(tmp_path, name='minus-v0.1', repo='Org/Minus-v0.1', dirname='minus-v0.1'):
    return {'name': name, 'hf_repo': repo, 'subfolder': 'axera', 'revision': 'main',
            'local_dir': str(tmp_path / 'models' / dirname)}


def test_model_not_installed_then_installed(tmp_path, hf):
    m = manifest(tmp_path)
    st = updates.check_model(m)
    assert st['error'] is None and st['installed'] is None and st['updatable'] is True
    assert sorted(st['changed_files']) == ['model.axmodel', 'sub/layer.axmodel', 'tokenizer.json']
    updates.update_model(m, log=lambda *a: None)
    local = Path(m['local_dir'])
    assert (local / 'model.axmodel').read_bytes() == b'weights-v1' * 100
    assert not (local / 'README.md').exists() and not (local / 'ignored.bin').exists()
    st = updates.check_model(m)
    assert st['up_to_date'] is True and st['installed'] == 'minus-v0.1' and st['installed_commit'] == 'abc1234'


def test_hand_installed_copy_is_adopted_once(tmp_path, hf):
    m = manifest(tmp_path)
    local = Path(m['local_dir'])
    (local / 'sub').mkdir(parents=True)
    for rel, data in hf.files.items():
        if rel.startswith('axera/'):
            (local / rel[len('axera/'):]).write_bytes(data)
    (local / 'eval.log').write_text('local extra')
    st = updates.check_model(m)
    assert st['up_to_date'] is True and (local / updates.MARKER_NAME).exists()
    assert hf.downloads == []


def test_only_changed_files_download_and_extras_survive(tmp_path, hf):
    m = manifest(tmp_path)
    updates.update_model(m, log=lambda *a: None)
    local = Path(m['local_dir'])
    (local / 'eval.log').write_text('keep me')
    hf.files['axera/model.axmodel'] = b'weights-v2' * 100
    hf.sha = 'def5678abc'
    hf.downloads.clear()
    st = updates.check_model(m)
    assert st['up_to_date'] is False and st['changed_files'] == ['model.axmodel']
    updates.update_model(m, log=lambda *a: None)
    assert hf.downloads == ['axera/model.axmodel']
    assert (local / 'model.axmodel').read_bytes() == b'weights-v2' * 100
    assert (local / 'eval.log').read_text() == 'keep me'
    assert (local.with_name(local.name + '.prev') / 'model.axmodel').read_bytes() == b'weights-v1' * 100
    assert updates.check_model(m)['up_to_date'] is True


def test_readme_change_on_hf_is_not_an_update(tmp_path, hf):
    m = manifest(tmp_path)
    updates.update_model(m, log=lambda *a: None)
    hf.sha = 'newreadme0'
    assert updates.check_model(m)['up_to_date'] is True


def test_corrupt_download_is_rejected_and_old_model_kept(tmp_path, hf, monkeypatch):
    m = manifest(tmp_path)
    updates.update_model(m, log=lambda *a: None)
    hf.files['axera/model.axmodel'] = b'weights-v2' * 100
    remote = updates.hf_model_files(m)
    remote[1]['model.axmodel']['sha256'] = '0' * 64          # what HF promises != what arrives
    with pytest.raises(RuntimeError, match='checksum'):
        updates.update_model(m, remote=remote, log=lambda *a: None)
    assert (Path(m['local_dir']) / 'model.axmodel').read_bytes() == b'weights-v1' * 100


def test_switching_to_a_new_model_is_a_manifest_change(tmp_path, hf):
    """Minus v0.2 = models/vlm_model.json pointing elsewhere: the check sees the
    new model isn't installed, the update downloads it to its own dir."""
    v1 = manifest(tmp_path)
    updates.update_model(v1, log=lambda *a: None)
    v2 = manifest(tmp_path, name='minus-v0.2', dirname='minus-v0.2')
    st = updates.check_model(v2)
    assert st['installed'] is None and st['updatable'] is True
    updates.update_model(v2, log=lambda *a: None)
    assert updates.check_model(v2)['installed'] == 'minus-v0.2'
    assert Path(v1['local_dir']).exists()            # the old model stays put


def test_manifest_and_config_agree():
    m = json.loads((ROOT / 'models' / 'vlm_model.json').read_text())
    assert m['hf_repo'] == 'TheGarageDev/Minus-v0.2' and m['subfolder'] == 'axera'
    assert m['name'] == 'minus-v0.2' and m['local_dir'].endswith('/minus-v0.2')
    import config
    if not os.environ.get('MINUS_VLM_MODEL_DIR'):
        assert config.VLM_MODEL_DIR == m['local_dir']


def test_env_overrides_the_manifest_dir(monkeypatch, tmp_path):
    monkeypatch.setenv('MINUS_VLM_MODEL_DIR', str(tmp_path / 'x'))
    assert updates.load_manifest()['local_dir'] == str(tmp_path / 'x')


# ------------------------------------------------------------- the updater run
def test_updater_dry_run_and_noop(repos, tmp_path, monkeypatch, hf):
    """End to end through tools/minus_update.py phase 1, in-process."""
    _, seed, box = repos
    sys.path.insert(0, str(ROOT / 'tools'))
    import importlib
    mu = importlib.import_module('minus_update')
    monkeypatch.setattr(updates, 'REPO_DIR', box)
    monkeypatch.setattr(updates, 'USTREAMER_DIR', tmp_path / 'no-ustreamer')
    monkeypatch.setattr(updates, 'MODEL_MANIFEST', tmp_path / 'none.json')
    status = tmp_path / 'status.json'
    run = mu.Run(status)
    mu.phase1(run, dry=False)
    assert json.loads(status.read_text())['state'] == 'done'           # nothing to do
    push_commit(seed, 'fake release')
    run = mu.Run(status)
    mu.phase1(run, dry=True)
    st = json.loads(status.read_text())
    assert st['state'] == 'done' and any('Minus:' in s['msg'] for s in st['steps'])
    assert updates.check_git_repo(box, 'Minus')['behind'] == 1          # dry run changed nothing


def test_web_routes_exist():
    src = (ROOT / 'src' / 'webui.py').read_text()
    for route in ("'/api/updates'", "'/api/updates/check'", "'/api/updates/apply'"):
        assert route in src
    html = (ROOT / 'src' / 'templates' / 'index.html').read_text()
    assert 'id="updates-apply-btn"' in html and 'function applyUpdates' in html


# --------------------------------------------------------- rollback behaviour
def _updater(monkeypatch, tmp_path):
    sys.path.insert(0, str(ROOT / 'tools'))
    import importlib
    mu = importlib.import_module('minus_update')
    calls = []

    def fake_run(cmd, *a, **k):
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, '', '')
    monkeypatch.setattr(mu.subprocess, 'run', fake_run)
    monkeypatch.setattr(mu.time, 'sleep', lambda s: None)
    return mu, calls


def test_service_start_clears_the_start_limit_first(monkeypatch, tmp_path):
    """A crash-looping build leaves the unit 'failed'; restart is refused until
    reset-failed (the live rollback test on minus left it down, 2026-10-17)."""
    mu, calls = _updater(monkeypatch, tmp_path)
    mu.service('restart')
    assert calls[0] == ['systemctl', 'reset-failed', mu.SERVICE]
    assert calls[1] == ['systemctl', 'restart', mu.SERVICE]


def test_crash_loop_ends_the_wait_early(monkeypatch, tmp_path):
    mu, calls = _updater(monkeypatch, tmp_path)
    monkeypatch.setattr(mu, 'web_ok', lambda: False)
    monkeypatch.setattr(mu, 'service_failed', lambda: True)
    t = [0.0]
    monkeypatch.setattr(mu.time, 'time', lambda: t.__setitem__(0, t[0] + 1) or t[0])
    assert mu.wait_healthy(240) is False
    assert t[0] < 10                     # gave up on the first failed probe, not after 240 s


def test_rollback_that_does_not_come_up_is_reported_failed(monkeypatch, tmp_path):
    mu, calls = _updater(monkeypatch, tmp_path)
    monkeypatch.setattr(mu, 'wait_healthy', lambda s: False)
    monkeypatch.setattr(updates, '_git', lambda *a, **k: '')
    run = mu.Run(tmp_path / 's.json')
    run.state['rollback'] = {'minus_commit': 'abc'}
    mu.undo(run, 'test')
    assert json.loads((tmp_path / 's.json').read_text())['state'] == 'failed'
    monkeypatch.setattr(mu, 'wait_healthy', lambda s: True)
    run = mu.Run(tmp_path / 's.json')
    run.state['rollback'] = {'minus_commit': 'abc'}
    mu.undo(run, 'test')
    assert json.loads((tmp_path / 's.json').read_text())['state'] == 'rolled_back'


def test_never_downgrades_to_an_older_release(repos, monkeypatch):
    """A newer commit always beats an older release: on v0.2.0 with GitHub's
    'latest' pointing at the older v0.1.0 (re-published/back-ported), stay."""
    _, seed, box = repos
    git(seed, 'tag', 'v0.1.0')
    push_commit(seed, 'newer work')
    git(seed, 'tag', 'v0.2.0')
    git(seed, 'push', '-q', 'origin', '--tags')
    git(box, 'fetch', '-q', '--tags')
    git(box, 'checkout', '-q', 'v0.2.0')
    monkeypatch.setattr(updates, '_github_slug', lambda repo: 'garagehq/Minus')
    monkeypatch.setattr(updates, 'latest_github_release',
                        lambda slug: {'tag': 'v0.1.0', 'name': 'old', 'url': 'u', 'published_at': 'x'})
    st = updates.check_git_repo(box, 'Minus')
    assert st['mode'] == 'release' and st['up_to_date'] is True and st['updatable'] is False
    assert st['latest'] == 'v0.2.0' and 'not newer' in st['reason']


def test_source_box_ahead_of_the_latest_release_stays_on_its_branch(repos, monkeypatch):
    """Running from source past the latest release: compared with the branch,
    never offered the (older) release."""
    _, seed, box = repos
    git(seed, 'tag', 'v0.1.0')
    git(seed, 'push', '-q', 'origin', 'v0.1.0')
    push_commit(seed, 'after the release')
    git(box, 'pull', '-q')
    monkeypatch.setattr(updates, '_github_slug', lambda repo: 'garagehq/Minus')
    monkeypatch.setattr(updates, 'latest_github_release',
                        lambda slug: {'tag': 'v0.1.0', 'name': 'x', 'url': 'u', 'published_at': 'x'})
    st = updates.check_git_repo(box, 'Minus')
    assert st['mode'] == 'source' and st['up_to_date'] is True and st['updatable'] is False
    assert st['latest_release']['tag'] == 'v0.1.0'


# ----------------------------------------------------- version + ownership
def test_installed_version_strings(repos):
    """The footer used to say a hard-coded 'MINUS v1.0' while the first real
    release is v0.1.0; it now shows git describe."""
    _, seed, box = repos
    sha = git(box, 'rev-parse', '--short=7', 'HEAD')
    assert updates.installed_version(box) == sha                     # no tags yet
    git(seed, 'tag', 'v0.1.0')
    git(seed, 'push', '-q', 'origin', 'v0.1.0')
    git(box, 'fetch', '-q', '--tags')
    assert updates.installed_version(box) == 'v0.1.0'                # on a release
    push_commit(seed, 'after')
    git(box, 'pull', '-q')
    v = updates.installed_version(box)
    assert v.startswith('v0.1.0-1-g') and not v.endswith('+')        # from source
    (box / 'a.txt').write_text('local edit\n')
    assert updates.installed_version(box).endswith('+')              # local changes
    assert updates.installed_version(box / 'nope') == 'unknown'


def test_owner_kwargs_switch_uid_without_pam(tmp_path, monkeypatch):
    """runuser opened a PAM session (2 journal lines) per git call; the root
    service now switches uid/gid in subprocess instead."""
    assert updates._owner_kwargs(tmp_path) == {} or os.geteuid() == 0   # not root: nothing to do
    monkeypatch.setattr(updates.os, 'geteuid', lambda: 0)
    import pwd
    me = pwd.getpwuid(os.getuid())
    kw = updates._owner_kwargs(tmp_path)
    if os.getuid() == 0:
        assert kw == {}                                                 # root-owned: run as is
    else:
        assert kw['user'] == me.pw_uid and kw['group'] == me.pw_gid
        assert kw['env']['HOME'] == me.pw_dir and me.pw_gid in kw['extra_groups']
    src = (ROOT / 'src' / 'updates.py').read_text()
    assert "'runuser'" not in src


def test_version_route_and_footer():
    assert "'/api/version'" in (ROOT / 'src' / 'webui.py').read_text()
    html = (ROOT / 'src' / 'templates' / 'index.html').read_text()
    assert 'MINUS v1.0' not in html and 'function loadVersion' in html


# ------------------------------------------- review fixes (2026-10-17)
def test_box_on_main_at_a_release_commit_stays_a_source_box(repos, monkeypatch):
    """release.sh tags origin/main's HEAD, so a box on main sits exactly on the
    new tag; it must keep following main, not become a release box."""
    _, seed, box = repos
    git(seed, 'tag', 'v0.1.0')
    git(seed, 'push', '-q', 'origin', 'v0.1.0')
    git(box, 'fetch', '-q', '--tags')
    monkeypatch.setattr(updates, '_github_slug', lambda repo: 'garagehq/Minus')
    monkeypatch.setattr(updates, 'latest_github_release',
                        lambda slug: {'tag': 'v0.1.0', 'name': 'x', 'url': 'u', 'published_at': 'x'})
    st = updates.check_git_repo(box, 'Minus')
    assert st['mode'] == 'source' and st['branch'] == 'main' and st['on_release_tag'] == 'v0.1.0'
    push_commit(seed, 'after the release')
    st = updates.check_git_repo(box, 'Minus')
    assert st['mode'] == 'source' and st['behind'] == 1 and st['updatable'] is True
    updates.update_git_repo(box, st)
    assert git(box, 'rev-parse', '--abbrev-ref', 'HEAD') == 'main'          # still on its branch


def test_a_moved_tag_does_not_break_the_check(repos):
    """A re-published (moved) tag made `git fetch --tags` fail forever."""
    _, seed, box = repos
    git(seed, 'tag', 'v0.1.0')
    git(seed, 'push', '-q', 'origin', 'v0.1.0')
    git(box, 'fetch', '-q', '--tags')
    push_commit(seed, 'fix')
    git(seed, 'tag', '-f', 'v0.1.0')
    git(seed, 'push', '-q', '-f', 'origin', 'v0.1.0')
    st = updates.check_git_repo(box, 'Minus')
    assert st['error'] is None
    assert git(box, 'rev-parse', 'v0.1.0') == git(seed, 'rev-parse', 'v0.1.0')


def test_summary_is_not_up_to_date_when_a_check_failed(tmp_path, monkeypatch):
    monkeypatch.setattr(updates, 'MODEL_MANIFEST', tmp_path / 'none.json')
    uc = updates.UpdateChecker(repo_dir=tmp_path / 'not-a-repo', ustreamer_dir=tmp_path / 'x',
                               status_file=tmp_path / 's.json')
    res = uc.check(force=True)
    assert res['errors'] and res['up_to_date'] is None and res['updatable'] is False


def _checker(tmp_path):
    return updates.UpdateChecker(repo_dir=tmp_path, ustreamer_dir=tmp_path / 'x',
                                 status_file=tmp_path / 'run' / 'status.json')


def test_lock_makes_the_updater_exclusive(tmp_path, monkeypatch):
    import fcntl
    uc = _checker(tmp_path)
    uc.status_file.parent.mkdir(parents=True)
    assert uc.updater_running() is False
    fd = os.open(uc.lock_file, os.O_RDWR | os.O_CREAT)
    fcntl.flock(fd, fcntl.LOCK_EX)
    try:
        assert uc.updater_running() is True and uc.busy() is True
        res = uc.start_update()
        assert res['success'] is False and 'already running' in res['error']
    finally:
        os.close(fd)
    assert uc.updater_running() is False


def test_starting_counts_as_busy_until_the_grace_runs_out(tmp_path):
    uc = _checker(tmp_path)
    uc.status_file.parent.mkdir(parents=True)
    uc.status_file.write_text(json.dumps({'state': 'starting', 'updated_at': __import__('time').time(),
                                          'steps': []}))
    assert uc.busy() is True and uc.run_status()['state'] == 'starting'
    uc.status_file.write_text(json.dumps({'state': 'starting', 'updated_at': 0, 'steps': []}))
    assert uc.run_status()['state'] == 'interrupted' and uc.busy() is False


def test_a_dead_running_update_is_reported_interrupted(tmp_path):
    """Power cut / crash mid-update left 'running' forever and the UI disabled
    Update now for good."""
    uc = _checker(tmp_path)
    uc.status_file.parent.mkdir(parents=True)
    uc.status_file.write_text(json.dumps({'state': 'running', 'updated_at': 0, 'steps': []}))
    st = uc.run_status()
    assert st['state'] == 'interrupted' and 'stopped before finishing' in st['steps'][-1]['msg']


def test_a_failed_launch_is_recorded(tmp_path, monkeypatch):
    uc = _checker(tmp_path)
    monkeypatch.setattr(updates.shutil, 'which', lambda n: None)

    def boom(*a, **k):
        raise OSError('no such file')
    monkeypatch.setattr(updates.subprocess, 'Popen', boom)
    res = uc.start_update()
    assert res['success'] is False
    assert json.loads(uc.status_file.read_text())['state'] == 'failed'


def test_mkdirs_owned_creates_the_tree(tmp_path):
    target = tmp_path / 'a' / 'b' / 'c.staging'
    updates._mkdirs_owned(target)
    assert target.is_dir()
    updates._mkdirs_owned(target)          # idempotent


def test_wait_healthy_needs_a_stable_process_and_the_vlm(monkeypatch, tmp_path):
    mu, calls = _updater(monkeypatch, tmp_path)
    clock = [0.0]
    monkeypatch.setattr(mu.time, 'time', lambda: clock[0])
    monkeypatch.setattr(mu.time, 'sleep', lambda s: clock.__setitem__(0, clock[0] + s))
    monkeypatch.setattr(mu, 'service_failed', lambda: False)
    monkeypatch.setattr(mu, 'STABLE_S', 30.0)
    pids = iter(['100', '100', '200'] + ['200'] * 100)         # one restart early on
    monkeypatch.setattr(mu, 'main_pid', lambda: next(pids))
    monkeypatch.setattr(mu, 'status_json', lambda: {'vlm_ready': clock[0] > 80})
    assert mu.wait_healthy(240, need_vlm=True) is True
    assert clock[0] > 80                                          # waited for the VLM, not just the UI
    clock[0] = 0.0
    monkeypatch.setattr(mu, 'status_json', lambda: {'vlm_ready': False})
    assert mu.wait_healthy(240, need_vlm=True) is False          # model never loads -> roll back
    clock[0] = 0.0
    monkeypatch.setattr(mu, 'main_pid', lambda: str(int(clock[0])))   # a new pid every probe: crash loop
    assert mu.wait_healthy(240, need_vlm=False) is False


def test_phase2_that_dies_is_rolled_back_by_phase1(repos, tmp_path, monkeypatch, hf):
    """phase 2 used to be an execv: if the new code couldn't import, the box
    was left fast-forwarded with no rollback and 'running' forever."""
    _, seed, box = repos
    mu, _ = _updater(monkeypatch, tmp_path)
    real_run = REAL_SUBPROCESS_RUN

    def run_but_phase2_dies(cmd, *a, **k):
        if '--phase2' in cmd:
            return subprocess.CompletedProcess(cmd, 1)     # the new code failed to start
        return real_run(cmd, *a, **k)
    monkeypatch.setattr(mu.subprocess, 'run', run_but_phase2_dies)
    monkeypatch.setattr(updates, 'REPO_DIR', box)
    monkeypatch.setattr(mu, 'REPO_DIR', box)
    monkeypatch.setattr(updates, 'USTREAMER_DIR', tmp_path / 'no-ustreamer')
    monkeypatch.setattr(updates, 'MODEL_MANIFEST', tmp_path / 'none.json')
    monkeypatch.setattr(mu, 'vlm_must_come_back', lambda: False)
    monkeypatch.setattr(mu, 'wait_healthy', lambda *a, **k: True)
    before = git(box, 'rev-parse', 'HEAD')
    push_commit(seed, 'update with a broken updater')
    real_git = updates._git
    undone = []
    monkeypatch.setattr(mu, 'undo', lambda run, why, restart=True: (
        undone.append(why), real_git(box, 'reset', '--hard', '--quiet', run.state['rollback']['minus_commit']),
        run.finish('rolled_back', why)))
    run = mu.Run(tmp_path / 'status.json')
    mu.phase1(run, dry=False)
    assert undone and 'new updater stopped' in undone[0]
    assert git(box, 'rev-parse', 'HEAD') == before
    assert json.loads((tmp_path / 'status.json').read_text())['state'] == 'rolled_back'


# ------------------------------------------------------------ shutdown
_SHUTDOWN_CHILD = r'''
import os, signal, sys, time, threading, traceback
sys.path.insert(0, {root!r}); sys.path.insert(0, {src!r})
import minus as M
class Fake:
    def __init__(self): self.stops = 0; self.running = True; self._stop_requested = False
    def stop(self):
        self.stops += 1
        in_handler = any(f.name == 'signal_handler' for f in traceback.extract_stack())
        print('stop', self.stops, 'in_handler' if in_handler else 'main_thread', flush=True)
        if {hang}:
            time.sleep(3600)           # a stop that never finishes
fake = Fake()
M._install_shutdown_handlers(fake, timeout_s=1.0)
os.kill(os.getpid(), signal.SIGTERM)
os.kill(os.getpid(), signal.SIGTERM)  # systemctl stop: ExecStop kill + systemd's own
deadline = time.time() + 30
while fake.running and time.time() < deadline:   # run()'s loop
    time.sleep(0.05)
print('loop exited', fake._stop_requested, flush=True)
fake.stop()
sys.exit(0)
'''


def _run_shutdown_child(hang):
    code = _SHUTDOWN_CHILD.format(root=str(ROOT), src=str(ROOT / 'src'), hang=hang)
    t0 = __import__('time').monotonic()
    r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=60)
    return r, __import__('time').monotonic() - t0


@pytest.fixture
def minus_importable():
    try:
        import minus  # noqa: F401
    except Exception as e:
        pytest.skip(f'minus.py not importable here: {e}')


def test_shutdown_ignores_a_second_sigterm_and_is_bounded(minus_importable):
    """A stop during an update hung until systemd's 90 s SIGKILL after a second
    SIGTERM re-entered stop() (minus, 2026-10-17)."""
    r, took = _run_shutdown_child(hang=True)
    assert 'loop exited True' in r.stdout        # the handler only set the flags
    assert r.stdout.count('stop ') == 1          # the repeated signal didn't re-enter stop()
    assert took < 15                             # bounded by the timeout, not by a SIGKILL


def test_normal_shutdown_still_exits_cleanly(minus_importable):
    r, took = _run_shutdown_child(hang=False)
    assert r.returncode == 0 and r.stdout.count('stop ') == 1


def test_stop_runs_on_the_main_thread_not_in_the_handler(minus_importable):
    """Stopping inside the signal handler re-entered the log writer
    ('reentrant call inside BufferedWriter')."""
    r, _ = _run_shutdown_child(hang=False)
    assert 'stop 1 main_thread' in r.stdout and 'in_handler' not in r.stdout


def test_service_stops_minus_py_only():
    unit = (ROOT / 'minus.service').read_text()
    assert 'KillMode=mixed' in unit and 'TimeoutStopSec=30' in unit
