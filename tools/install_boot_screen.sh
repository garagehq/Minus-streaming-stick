#!/bin/bash
# Install the KingBlocker9000 boot splash (Plymouth theme plymouth/kingblocker).
#
#   sudo tools/install_boot_screen.sh
#
# Plymouth draws the boot screen from early boot (it runs from the initramfs,
# so the initramfs is rebuilt; the previous one is kept as
# /boot/initrd.img-<kernel>.bak-bootscreen) and stays up until Minus draws
# its own copy of the boot screen, so there is no black gap in between.
# install.sh runs this (step 9). Idempotent: does nothing if already current.
#
# Undo: sudo rm /etc/systemd/system/plymouth-quit{,-wait}.service.d/kingblocker.conf
#       sudo plymouth-set-default-theme spinner && sudo update-initramfs -u

set -euo pipefail
if [ "$EUID" -ne 0 ]; then echo "Run as root: sudo $0"; exit 1; fi
MINUS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THEME=kingblocker
SRC="$MINUS_DIR/plymouth/$THEME"
DST="/usr/share/plymouth/themes/$THEME"
KVER="$(uname -r)"

if ! command -v plymouth-set-default-theme >/dev/null; then
    echo "    boot screen: plymouth not installed; skipped"
    exit 0
fi

# Theme files: the script, its .plymouth and the shared screen images.
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
cp "$SRC/$THEME.plymouth" "$SRC/$THEME.script" "$STAGE/"
for f in boot_background.png bar_track.png bar_fill_left.png bar_fill_mid.png bar_fill_right.png; do
    cp "$MINUS_DIR/assets/screens/$f" "$STAGE/"
done
CHANGED=0
if ! diff -qr "$STAGE" "$DST" >/dev/null 2>&1; then
    install -d "$DST"
    rm -f "$DST"/*
    install -m 644 "$STAGE"/* "$DST/"
    CHANGED=1
fi

# Show it at once (no delay) and make it the default theme.
CONF=/etc/plymouth/plymouthd.conf
WANT="$(printf '# Installed by Minus (tools/install_boot_screen.sh)\n[Daemon]\nTheme=%s\nShowDelay=0\n' "$THEME")"
if [ "$(cat "$CONF" 2>/dev/null)" != "$WANT" ]; then
    [ -f "$CONF.bak-minus" ] || cp "$CONF" "$CONF.bak-minus" 2>/dev/null || true
    echo "$WANT" > "$CONF"
    CHANGED=1
fi

# Keep the splash up until Minus draws. Normally Plymouth quits when boot
# finishes (~9 s) and Minus draws at ~25-30 s: a black TV in between. Both
# quit services run before multi-user.target, which minus.service starts
# after, so they can't simply wait for Minus; they become no-ops instead and
# Minus quits Plymouth itself just before its first screen
# (boot_screen.release_plymouth). If Minus never draws, a timer quits it
# 3 minutes after boot.
dropin() {   # $1 = unit, $2 = ExecStart
    local f="/etc/systemd/system/$1.d/kingblocker.conf" want
    want="$(printf '# Installed by Minus (tools/install_boot_screen.sh)\n[Service]\nExecStart=\nExecStart=%s\n' "$2")"
    if [ "$(cat "$f" 2>/dev/null)" != "$want" ]; then
        install -d "$(dirname "$f")"
        echo "$want" > "$f"
        RELOAD=1
    fi
}
RELOAD=0
dropin plymouth-quit.service \
    "-/usr/bin/systemd-run --no-block --on-active=180 --unit=minus-plymouth-failsafe /usr/bin/plymouth quit --retain-splash"
dropin plymouth-quit-wait.service "-/bin/true"
[ "$RELOAD" = 1 ] && systemctl daemon-reload

# The kernel's only console is the serial port (ttyFIQ0, silent boot), and
# with a serial console Plymouth shows plain text there instead of the
# splash on HDMI ("serial consoles detected, managing them with details
# forced"). Kernel messages and the recovery shell still go to the serial port.
if [ -f /etc/kernel/cmdline ] && ! grep -qw 'plymouth.ignore-serial-consoles' /etc/kernel/cmdline; then
    [ -f /etc/kernel/cmdline.bak-minus ] || cp /etc/kernel/cmdline /etc/kernel/cmdline.bak-minus
    sed -i -E 's/$/ plymouth.ignore-serial-consoles/; s/  +/ /g' /etc/kernel/cmdline
    /usr/sbin/u-boot-update >/dev/null
    echo "    boot screen: kernel cmdline plymouth.ignore-serial-consoles (reboot to apply)"
fi

INITRD="/boot/initrd.img-$KVER"
if [ "$CHANGED" = 1 ] || ! lsinitramfs "$INITRD" 2>/dev/null | grep -q "themes/$THEME/$THEME.script"; then
    [ -f "$INITRD.bak-bootscreen" ] || cp -p "$INITRD" "$INITRD.bak-bootscreen"
    plymouth-set-default-theme "$THEME"
    update-initramfs -u -k "$KVER" >/dev/null
    sync
    if ! lsinitramfs "$INITRD" | grep -q "themes/$THEME/$THEME.script"; then
        echo "    boot screen: WARNING theme not found in the new initramfs"
        exit 1
    fi
    echo "    boot screen: installed (initramfs rebuilt; previous: $INITRD.bak-bootscreen)"
else
    echo "    boot screen: up to date"
fi
