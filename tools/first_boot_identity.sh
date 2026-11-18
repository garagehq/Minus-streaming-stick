#!/bin/sh
# Give a freshly cloned unit its own identity (minus-first-boot.service).
#
# Runs before sshd when tools/prepare_for_clone.sh left the marker
# /etc/minus/.clone-fresh, or whenever the SSH host keys are missing:
#   - new SSH host keys (sshd will not start without them);
#   - new machine-id (DHCP client id, journal directory, D-Bus).
# The machine-id is replaced here rather than left empty in the image:
# an empty /etc/machine-id makes systemd treat the boot as a factory first
# boot and re-apply unit presets, which can re-enable the desktop that
# install.sh turned off.
# Tailscale identity is handled by tools/tailscale_enroll.sh (its state is
# wiped by prepare_for_clone.sh, so it enrolls as a new node).

ssh-keygen -A >/dev/null 2>&1 && echo "first boot: SSH host keys present"

if [ -e /etc/minus/.clone-fresh ]; then
    rm -f /etc/machine-id
    if [ -L /var/lib/dbus/machine-id ]; then :; else rm -f /var/lib/dbus/machine-id; fi
    systemd-machine-id-setup >/dev/null 2>&1
    [ -e /var/lib/dbus/machine-id ] || ln -s /etc/machine-id /var/lib/dbus/machine-id
    systemctl restart systemd-journald 2>/dev/null
    echo "first boot: new machine-id $(cut -c1-8 /etc/machine-id)"
    rm -f /etc/minus/.clone-fresh
fi
exit 0
