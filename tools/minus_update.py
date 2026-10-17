#!/usr/bin/env python3
"""Apply available updates to a Minus box: code, ustreamer, VLM model. Then restart.

Started by Settings -> Updates -> "Update now" (src/updates.py UpdateChecker.
start_update), as a separate systemd unit so restarting Minus doesn't kill it.
Progress goes to a JSON status file the web UI polls (it survives the restart).

Phase 1 (the code that is installed now):
  1. ustreamer: fast-forward its checkout and build it (nothing is installed yet);
     a failed build stops the whole update with nothing changed.
  2. Minus: fast-forward (or move to the newer release tag); reinstall Python
     requirements if requirements.txt changed.
  then re-executes itself, so phase 2 runs the NEW code — a code update that
  moves to a new model (models/vlm_model.json) is acted on by the code that
  knows about it.
Phase 2 (the updated code):
  3. VLM model: download changed files to a staging dir, verify, swap (old copy
     kept as <dir>.prev).
  4. Restart Minus (installing the new ustreamer binary while it is stopped) and
     wait for the web UI to answer. If it doesn't within MINUS_UPDATE_HEALTH_S,
     roll everything back (commits, binary, model) and restart again.

Usage: minus_update.py [--status-file PATH] [--dry-run]
       minus_update.py --install-model     (install.sh: fetch the model in
                                            models/vlm_model.json if it isn't
                                            installed; no restart)
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR / 'src'))
import updates  # noqa: E402

USTREAMER_BIN = Path(os.environ.get('MINUS_USTREAMER_PATH', '/home/radxa/ustreamer-patched'))
WEB_PORT = int(os.environ.get('MINUS_WEB_PORT', '80'))
HEALTH_S = float(os.environ.get('MINUS_UPDATE_HEALTH_S', '240'))
SERVICE = os.environ.get('MINUS_SERVICE_NAME', 'minus')


class Run:
    def __init__(self, path, state=None):
        self.path = Path(path)
        self.state = state or {'state': 'running', 'started_at': time.time(), 'steps': []}
        self.state['state'] = 'running'

    def log(self, msg):
        print(msg, flush=True)
        self.state['steps'].append({'t': time.time(), 'msg': msg})
        self.save()

    def save(self):
        self.state['updated_at'] = time.time()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix('.tmp')
        tmp.write_text(json.dumps(self.state, indent=1))
        os.replace(tmp, self.path)

    def finish(self, state, msg):
        self.state['state'] = state
        self.log(msg)


def sh(*cmd, cwd=None, timeout=1800):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed: {(r.stderr or r.stdout).strip()[-400:]}")
    return r.stdout


def service(action):
    # A crash-looping build can hit systemd's start limit, after which the unit
    # is "failed" and start/restart are refused ("Start request repeated too
    # quickly") — seen in the rollback test on minus, 2026-10-17. Clear it first.
    if action in ('start', 'restart'):
        subprocess.run(['systemctl', 'reset-failed', SERVICE], timeout=60,
                       capture_output=True)
    sh('systemctl', action, SERVICE, timeout=180)


def service_failed():
    return subprocess.run(['systemctl', 'is-failed', '--quiet', SERVICE],
                          timeout=30).returncode == 0


def wait_healthy(timeout_s):
    """True once the web UI answers; False on timeout or if systemd gives up
    on the unit (a crash loop — no point waiting the full timeout)."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        time.sleep(5)
        if web_ok():
            return True
        if service_failed():
            return False
    return False


def web_ok():
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{WEB_PORT}/api/status', timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def comp(status, name):
    return next((c for c in status['components'] if c['name'] == name), None)


# ----------------------------------------------------------------- phase 1
def phase1(run, dry):
    checker = updates.UpdateChecker()
    status = checker.check(force=True)
    run.state['before'] = status
    todo = [c['name'] for c in status['components'] if c.get('updatable')]
    if not todo:
        run.finish('done', 'Everything is already up to date.')
        return
    run.log(f"Updating: {', '.join(todo)}")
    rollback = run.state.setdefault('rollback', {})

    us = comp(status, 'ustreamer')
    if us and us.get('updatable'):
        run.log(f"ustreamer: {us['commit']} -> {us['latest']}")
        if not dry:
            old, new = updates.update_git_repo(updates.USTREAMER_DIR, us)
            rollback['ustreamer_commit'] = old
            try:
                run.log('ustreamer: building (make WITH_MPP=1)...')
                sh(*updates._as_owner(updates.USTREAMER_DIR), 'make', 'WITH_MPP=1', f'-j{os.cpu_count() or 4}',
                   cwd=updates.USTREAMER_DIR, timeout=3600)
                run.state['ustreamer_built'] = True
            except Exception as e:
                updates._git(updates.USTREAMER_DIR, 'reset', '--hard', '--quiet', old)
                run.finish('failed', f'ustreamer build failed, nothing was changed: {e}')
                return

    mc = comp(status, 'Minus')
    if mc and mc.get('updatable'):
        run.log(f"Minus: {mc['commit']} -> {mc['latest']}")
        if not dry:
            req_before = (REPO_DIR / 'requirements.txt').read_bytes() if (REPO_DIR / 'requirements.txt').exists() else b''
            old, new = updates.update_git_repo(REPO_DIR, mc)
            rollback['minus_commit'] = old
            req_after = (REPO_DIR / 'requirements.txt').read_bytes() if (REPO_DIR / 'requirements.txt').exists() else b''
            if req_after != req_before:
                run.log('requirements.txt changed: installing Python packages...')
                try:
                    sh(sys.executable, '-m', 'pip', 'install', '--break-system-packages', '-q',
                       '-r', str(REPO_DIR / 'requirements.txt'), timeout=1800)
                except Exception as e:
                    run.log(f'pip install failed (continuing; the health check decides): {e}')
    run.save()
    if dry:
        run.finish('done', 'Dry run: nothing changed.')
        return
    # Phase 2 with the code we just installed.
    os.execv(sys.executable, [sys.executable, str(REPO_DIR / 'tools' / 'minus_update.py'),
                              '--status-file', str(run.path), '--phase2'])


# ----------------------------------------------------------------- phase 2
def phase2(run):
    rollback = run.state.setdefault('rollback', {})
    model = updates.check_model()
    if model.get('updatable'):
        run.log(f"VLM model: {model.get('installed') or 'not installed'} -> {model['model']} "
                f"({len(model['changed_files'])} files, {model['download_bytes'] / 1e6:.0f} MB)")
        try:
            updates.update_model(log=run.log)
            rollback['model_dir'] = model['path']
        except Exception as e:
            undo(run, f'model download failed: {e}', restart=False)
            return
    elif model.get('error'):
        run.log(f"VLM model: check failed ({model['error']}) — leaving it as it is")

    run.log(f'Restarting {SERVICE}...')
    backup = None
    try:
        service('stop')
        if run.state.get('ustreamer_built'):
            built = updates.USTREAMER_DIR / 'ustreamer'
            if USTREAMER_BIN.exists():
                backup = USTREAMER_BIN.with_name(USTREAMER_BIN.name + '.prev')
                shutil.copy2(USTREAMER_BIN, backup)
                rollback['ustreamer_bin'] = str(backup)
            shutil.copy2(built, USTREAMER_BIN)
            run.log('ustreamer: new binary installed')
        service('start')
    except Exception as e:
        undo(run, f'restart failed: {e}')
        return
    run.save()
    if wait_healthy(HEALTH_S):
        after = updates.UpdateChecker().check(force=True)
        run.state['after'] = after
        run.finish('done', f'Update complete — {SERVICE} is running the new version.')
        return
    why = ('it kept crashing' if service_failed()
           else f'it did not answer within {HEALTH_S:.0f} s')
    undo(run, f'{SERVICE} failed to come up after the update ({why})')


def undo(run, why, restart=True):
    rb = run.state.get('rollback', {})
    run.log(f'Rolling back: {why}')
    try:
        if restart:
            subprocess.run(['systemctl', 'stop', SERVICE], timeout=180, capture_output=True)
        if rb.get('minus_commit'):
            updates._git(REPO_DIR, 'reset', '--hard', '--quiet', rb['minus_commit'])
        if rb.get('ustreamer_commit'):
            updates._git(updates.USTREAMER_DIR, 'reset', '--hard', '--quiet', rb['ustreamer_commit'])
        if rb.get('ustreamer_bin'):
            shutil.copy2(rb['ustreamer_bin'], USTREAMER_BIN)
        if rb.get('model_dir'):
            local = Path(rb['model_dir'])
            prev = local.with_name(local.name + '.prev')
            if prev.exists():
                failed = local.with_name(local.name + '.failed')
                if failed.exists():
                    shutil.rmtree(failed)
                os.replace(local, failed)
                os.replace(prev, local)
        if restart:
            service('restart')
            if not wait_healthy(HEALTH_S):
                run.finish('failed', f'Rolled back ({why}), but {SERVICE} is not answering on the '
                                     f'previous version either — check the box.')
                return
        run.finish('rolled_back', f'Rolled back to the previous version ({why}).')
    except Exception as e:
        run.finish('failed', f'Rollback incomplete ({why}; {e}) — check the box by hand.')


def install_model():
    """Download/verify the VLM model named in models/vlm_model.json. 0 = ready."""
    st = updates.check_model()
    if st.get('error'):
        print(f"VLM model: check failed: {st['error']}")
        return 1
    if st.get('up_to_date'):
        print(f"VLM model {st['model']} already installed at {st['path']}")
        return 0
    print(f"VLM model {st['model']}: downloading {len(st['changed_files'])} files "
          f"({st['download_bytes'] / 1e6:.0f} MB) from huggingface.co/{st['hf_repo']} ...")
    updates.update_model(log=print)
    return 0 if updates.check_model().get('up_to_date') else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--status-file', default=str(updates.STATUS_FILE))
    ap.add_argument('--phase2', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--install-model', action='store_true')
    a = ap.parse_args()
    if a.install_model:
        sys.exit(install_model())
    prev = None
    if a.phase2:
        try:
            prev = json.loads(Path(a.status_file).read_text())
        except Exception:
            prev = None
    run = Run(a.status_file, prev)
    try:
        if a.phase2:
            phase2(run)
        else:
            run.state = {'state': 'running', 'started_at': time.time(), 'steps': []}
            phase1(run, a.dry_run)
    except Exception as e:
        if run.state.get('rollback'):
            undo(run, f'unexpected error: {e}')
        else:
            run.finish('failed', f'Update failed, nothing was changed: {e}')


if __name__ == '__main__':
    main()
