#!/bin/bash
# Prepare this unit's card for imaging: the LAST thing to run before you pull
# the card and clone it. Every clone (and this card, if booted again) then
# comes up with its own identity:
#
#   Tailscale  logged out, node state wiped -> each clone enrolls itself as
#              a new node (tools/tailscale_enroll.sh, needs the auth key)
#   SSH        host keys deleted -> regenerated before sshd on first boot
#   machine-id regenerated on first boot (minus-first-boot.service)
#   journal    cleared
#
#   sudo ~/Minus/tools/prepare_for_clone.sh --yes            # then it shuts down
#   sudo ~/Minus/tools/prepare_for_clone.sh --yes --no-shutdown
#
# This unit drops off Tailscale immediately; reach it on its LAN IP until it
# shuts down.

set -u
if [ "$EUID" -ne 0 ]; then echo "Run as root: sudo $0 --yes"; exit 1; fi
if [ "${1:-}" != "--yes" ]; then
    sed -n '2,17p' "$0"; echo; echo "Re-run with --yes to proceed."; exit 1
fi
SHUTDOWN=1; [ "${2:-}" = "--no-shutdown" ] && SHUTDOWN=0

echo "== checks"
ok=1
[ -s /etc/minus/tailscale-authkey ] && echo "  auth key: present" ||
    { echo "  WARNING: no /etc/minus/tailscale-authkey: clones will NOT join Tailscale"; ok=0; }
for u in minus-tailscale-enroll.timer minus-first-boot.service; do
    [ "$(systemctl is-enabled $u 2>/dev/null)" = enabled ] && echo "  $u: enabled" ||
        { echo "  WARNING: $u not enabled (run install.sh)"; ok=0; }
done
[ "$ok" = 1 ] || { read -r -p "Continue anyway? [y/N] " a; [ "$a" = y ] || exit 1; }

echo "== Tailscale: log out and wipe node state"
systemctl stop minus-tailscale-enroll.timer 2>/dev/null
timeout -k 5 20 tailscale logout 2>/dev/null
systemctl stop tailscaled
rm -rf /var/lib/tailscale/*

echo "== SSH host keys"
rm -f /etc/ssh/ssh_host_*

echo "== mark as fresh clone (new machine-id on first boot)"
install -d /etc/minus
touch /etc/minus/.clone-fresh

echo "== journal and runtime stamps"
journalctl --rotate >/dev/null 2>&1
journalctl --vacuum-time=1s >/dev/null 2>&1
rm -f /run/minus-update.* /run/minus-ts-enroll.*

sync
echo "== done: image this card now"
[ "$SHUTDOWN" = 1 ] && shutdown -h now
exit 0
