"""Unattended security upgrades — install / enable / disable / status.

Minus boxes run headless for weeks at a time; Debian security updates should
apply themselves. This module owns the two apt config files that control it:

  /etc/apt/apt.conf.d/20auto-upgrades              the on/off switch
      APT::Periodic::Update-Package-Lists "1";
      APT::Periodic::Unattended-Upgrade "1";
  /etc/apt/apt.conf.d/52minus-unattended-upgrades  the Minus policy overlay
      * never auto-reboot — a reboot mid-show is worse than a delayed patch
      * blacklist every package this box's video path depends on (kernel,
        u-boot, Rockchip MPP/RGA, GStreamer, Mesa, RKNN, Tailscale) so a
        routine security run can never swap out a driver the pipeline was
        validated against. Those get updated deliberately, by a human.

Debian's stock 50unattended-upgrades already restricts origins to
bookworm-security, so ordinary package churn from the Radxa / backports /
vendor repos is never auto-applied — only security fixes.

The minus service runs as root, so the web-UI toggle can write these files
directly. Outside the service (dev runs as a normal user) set_enabled()
reports an error carrying the manual command instead of failing silently.
"""

import logging
import os
import re
import subprocess
from pathlib import Path

logger = logging.getLogger('Minus.Upgrades')

AUTO_UPGRADES_FILE = Path('/etc/apt/apt.conf.d/20auto-upgrades')
POLICY_FILE = Path('/etc/apt/apt.conf.d/52minus-unattended-upgrades')
LOG_FILE = Path('/var/log/unattended-upgrades/unattended-upgrades.log')
REBOOT_REQUIRED_FILE = Path('/var/run/reboot-required')
PACKAGE = 'unattended-upgrades'
INSTALL_COMMAND = 'sudo apt-get install -y unattended-upgrades'

# Packages a security run must never touch on a Minus box. These are the
# regex forms Unattended-Upgrade::Package-Blacklist expects.
BLACKLIST = [
    'linux-image-.*', 'linux-headers-.*', 'u-boot-.*',
    'librockchip-.*', 'librga.*', 'gstreamer1.0-rockchip.*',
    'libgstreamer.*', 'gstreamer1.0-.*',
    'libmali.*', 'mesa-.*', 'libegl.*', 'libgl1.*', 'libgbm.*',
    'rknpu.*', 'python3-gi', 'tailscale',
]

POLICY_HEADER = "// Written by Minus (src/unattended_upgrades.py). Edits are overwritten.\n"


def _policy_text() -> str:
    bl = "\n".join(f'    "{p}";' for p in BLACKLIST)
    return (
        POLICY_HEADER +
        "// Never reboot on its own: a reboot mid-show is worse than a delayed patch.\n"
        'Unattended-Upgrade::Automatic-Reboot "false";\n'
        "// Apply each upgrade in the smallest possible step so a SIGTERM can't\n"
        "// leave dpkg half-configured.\n"
        'Unattended-Upgrade::MinimalSteps "true";\n'
        'Unattended-Upgrade::Remove-Unused-Dependencies "true";\n'
        "// The video path was validated against exact versions of these; a\n"
        "// routine security run must not swap them. Update deliberately.\n"
        "Unattended-Upgrade::Package-Blacklist {\n" + bl + "\n};\n"
    )


def _auto_text(enabled: bool) -> str:
    v = "1" if enabled else "0"
    return (f'APT::Periodic::Update-Package-Lists "{v}";\n'
            f'APT::Periodic::Unattended-Upgrade "{v}";\n')


def is_root() -> bool:
    try:
        return os.geteuid() == 0
    except AttributeError:
        return False


def is_installed() -> bool:
    try:
        out = subprocess.run(
            ['dpkg-query', '-W', '-f=${Status}', PACKAGE],
            capture_output=True, text=True, timeout=5)
        return out.returncode == 0 and 'install ok installed' in out.stdout
    except Exception:
        return False


def is_enabled() -> bool:
    """True if 20auto-upgrades turns the periodic unattended run on."""
    try:
        text = AUTO_UPGRADES_FILE.read_text()
    except Exception:
        return False
    m = re.search(r'APT::Periodic::Unattended-Upgrade\s+"(\d+)"', text)
    return bool(m and m.group(1) != '0')


def policy_present() -> bool:
    try:
        return POLICY_FILE.read_text().startswith(POLICY_HEADER)
    except Exception:
        return False


def write_policy() -> bool:
    try:
        POLICY_FILE.write_text(_policy_text())
        return True
    except Exception as e:
        logger.warning(f"[Upgrades] could not write policy {POLICY_FILE}: {e}")
        return False


def set_enabled(enabled: bool) -> dict:
    """Flip the periodic switch (and lay down the policy when enabling)."""
    if not is_root():
        v = '1' if enabled else '0'
        return {'success': False,
                'error': 'must run as root to change apt configuration',
                'manual_command': (f"sudo sed -i 's/APT::Periodic::Unattended-Upgrade \"[01]\"/"
                                   f"APT::Periodic::Unattended-Upgrade \"{v}\"/' {AUTO_UPGRADES_FILE}")}
    if enabled and not is_installed():
        return {'success': False,
                'error': f'{PACKAGE} is not installed',
                'manual_command': INSTALL_COMMAND}
    try:
        AUTO_UPGRADES_FILE.write_text(_auto_text(enabled))
    except Exception as e:
        return {'success': False, 'error': f'could not write {AUTO_UPGRADES_FILE}: {e}'}
    if enabled:
        write_policy()
    logger.info(f"[Upgrades] unattended security upgrades {'ENABLED' if enabled else 'disabled'}")
    return {'success': True, 'enabled': bool(enabled)}


_LOG_START = re.compile(r'^(\S+ \S+) INFO Starting unattended upgrades script')
_LOG_RESULT = re.compile(
    r'^(\S+ \S+) (?:INFO|WARNING|ERROR) '
    r'(No packages found that can be upgraded unattended.*|'
    r'Packages that will be upgraded: .*|All upgrades installed|'
    r'Installing the packages failed.*|Package .* has a higher version.*|'
    r'Packages that are upgraded: .*)')


def last_run(log_path: Path = None):
    """(timestamp, one-line result) of the most recent run, or (None, None)."""
    path = log_path or LOG_FILE
    try:
        lines = path.read_text(errors='replace').splitlines()[-400:]
    except Exception:
        return None, None
    ts, result = None, None
    for line in lines:
        m = _LOG_START.match(line)
        if m:
            ts, result = m.group(1), None
            continue
        m = _LOG_RESULT.match(line)
        if m and ts:
            result = m.group(2)[:120]
    return ts, result


def next_run() -> str:
    try:
        out = subprocess.run(
            ['systemctl', 'show', '-p', 'NextElapseUSecRealtime', '--value',
             'apt-daily-upgrade.timer'], capture_output=True, text=True, timeout=5)
        v = out.stdout.strip()
        return v if v and v != 'n/a' else None
    except Exception:
        return None


def reboot_required() -> bool:
    return REBOOT_REQUIRED_FILE.exists()


def status() -> dict:
    installed = is_installed()
    ts, result = last_run() if installed else (None, None)
    return {
        'installed': installed,
        'enabled': is_enabled() if installed else False,
        'policy_present': policy_present(),
        'last_run_time': ts,
        'last_run_result': result,
        'next_run': next_run() if installed else None,
        'reboot_required': reboot_required(),
        'is_root': is_root(),
        'install_command': INSTALL_COMMAND,
        'blacklist': list(BLACKLIST),
    }
