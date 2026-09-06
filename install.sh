#!/bin/bash
# Minus Installation Script
# Sets up systemd service for auto-start on boot
# Configures hostname and mDNS for easy network access

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_NAME="minus"
SERVICE_FILE="${SCRIPT_DIR}/minus.service"
HOSTNAME="minus"

echo "=== Minus Installation ==="
echo ""

# Check root
if [ "$EUID" -ne 0 ]; then
    echo "Please run as root: sudo $0"
    exit 1
fi

# Check if service file exists
if [ ! -f "$SERVICE_FILE" ]; then
    echo "ERROR: Service file not found: $SERVICE_FILE"
    exit 1
fi

# Enable hardware device-tree overlays needed by the IR transmitter (PWM3 on
# pin 38) and the WS2812B status LED strip (SPI0_MOSI_M2 on pin 19). Both
# overlays ship in /boot/dtbo with a .disabled suffix; we move them in place
# and re-run u-boot-update so /boot/extlinux/extlinux.conf includes them on
# the next boot.
REBOOT_NEEDED=0

echo "[1/12] Enabling hardware device-tree overlays..."
NEED_UBOOT_UPDATE=0

SPI_OVERLAY="/boot/dtbo/rk3588-spi0-m2-cs0-spidev.dtbo"
if [ -f "${SPI_OVERLAY}.disabled" ] && [ ! -f "${SPI_OVERLAY}" ]; then
    echo "    Enabling rk3588-spi0-m2-cs0-spidev (status LED strip)..."
    mv "${SPI_OVERLAY}.disabled" "${SPI_OVERLAY}"
    NEED_UBOOT_UPDATE=1
elif [ -f "${SPI_OVERLAY}" ]; then
    echo "    rk3588-spi0-m2-cs0-spidev already enabled"
else
    echo "    WARNING: rk3588-spi0-m2-cs0-spidev not found in /boot/dtbo (status LEDs unavailable)"
fi

PWM_OVERLAY="/boot/dtbo/rk3588-pwm3-m1.dtbo"
if [ -f "${PWM_OVERLAY}.disabled" ] && [ ! -f "${PWM_OVERLAY}" ]; then
    echo "    Enabling rk3588-pwm3-m1 (IR transmitter)..."
    mv "${PWM_OVERLAY}.disabled" "${PWM_OVERLAY}"
    NEED_UBOOT_UPDATE=1
elif [ -f "${PWM_OVERLAY}" ]; then
    echo "    rk3588-pwm3-m1 already enabled"
else
    echo "    WARNING: rk3588-pwm3-m1 not found in /boot/dtbo (IR transmitter unavailable)"
fi

if [ "${NEED_UBOOT_UPDATE}" -eq 1 ]; then
    echo "    Regenerating /boot/extlinux/extlinux.conf..."
    /usr/sbin/u-boot-update
    REBOOT_NEEDED=1
fi

# Python bindings for SPI (status LEDs). The status_leds module imports
# `spidev` at runtime; without this the LED toggle in the web UI 503s.
echo "[2/12] Installing python3-spidev..."
if ! dpkg -l 2>/dev/null | grep -q "^ii\s\+python3-spidev"; then
    apt-get install -y -qq python3-spidev || \
        echo "    WARNING: python3-spidev install failed; status LEDs will be unavailable"
else
    echo "    python3-spidev already installed"
fi

# Add radxa to the spi group so interactive scripts (test_status_leds.py)
# can open /dev/spidev0.0 without sudo. Service-mode runs as root anyway.
# The group is created by udev on overlay probe, so on a fresh install
# this can fail silently the first time — it'll be a no-op next run.
echo "[3/12] Adding radxa to spi group..."
if getent group spi >/dev/null; then
    if id -nG radxa 2>/dev/null | tr ' ' '\n' | grep -q '^spi$'; then
        echo "    radxa already in spi group"
    else
        usermod -aG spi radxa && echo "    Added radxa to spi group (log out/in to take effect)"
    fi
else
    echo "    spi group not present yet — will appear after reboot when overlay loads"
fi

# Setup hostname and mDNS
echo "[4/12] Setting up hostname and mDNS..."
hostnamectl set-hostname ${HOSTNAME}
sed -i "s/127.0.1.1.*/127.0.1.1\t${HOSTNAME}/" /etc/hosts 2>/dev/null || echo "127.0.1.1	${HOSTNAME}" >> /etc/hosts

# Install and enable avahi for mDNS (.local resolution)
if ! dpkg -l | grep -q avahi-daemon; then
    echo "    Installing avahi-daemon for mDNS..."
    apt-get update -qq && apt-get install -y -qq avahi-daemon
fi
systemctl enable avahi-daemon 2>/dev/null || true
systemctl restart avahi-daemon 2>/dev/null || true
echo "    Hostname set to: ${HOSTNAME}"
echo "    Access via: http://${HOSTNAME}.local:80"

# Stop existing service if running
echo "[5/12] Stopping existing service..."
systemctl stop ${SERVICE_NAME} 2>/dev/null || true
systemctl disable ${SERVICE_NAME} 2>/dev/null || true

# Stop X11 to free up display
echo "[6/12] Stopping X11 (gdm3)..."
systemctl stop gdm3 2>/dev/null || true
systemctl disable gdm3 2>/dev/null || true

# Copy service file
echo "[7/12] Installing systemd service..."
cp "$SERVICE_FILE" /etc/systemd/system/${SERVICE_NAME}.service
chmod 644 /etc/systemd/system/${SERVICE_NAME}.service

# Reload systemd
echo "[8/12] Reloading systemd..."
systemctl daemon-reload

# Enable and start service
echo "[9/12] Enabling service..."
systemctl enable ${SERVICE_NAME}

# Create screenshot directories
echo "[10/12] Creating screenshot directories..."
mkdir -p "${SCRIPT_DIR}/screenshots/ads"
mkdir -p "${SCRIPT_DIR}/screenshots/non_ads"
mkdir -p "${SCRIPT_DIR}/screenshots/vlm_spastic"
mkdir -p "${SCRIPT_DIR}/screenshots/static"
chown -R radxa:radxa "${SCRIPT_DIR}/screenshots" 2>/dev/null || true

# Ensure login shells (tmux, ssh) source ~/.bashrc so $HOME/.local/bin is on PATH.
# Without this, a fresh tmux window picks up /usr/local/bin/claude (older) instead
# of ~/.local/bin/claude (newer), forcing a manual `source ~/.bashrc` each time.
echo "[11/12] Ensuring login shells source ~/.bashrc for radxa..."
RADXA_HOME="$(getent passwd radxa | cut -d: -f6)"
if [ -n "$RADXA_HOME" ] && [ -d "$RADXA_HOME" ]; then
    BASH_PROFILE="${RADXA_HOME}/.bash_profile"
    if [ ! -f "$BASH_PROFILE" ] || ! grep -q "\.bashrc" "$BASH_PROFILE"; then
        cat >> "$BASH_PROFILE" <<'EOF'
if [ -f ~/.bashrc ]; then
    . ~/.bashrc
fi
EOF
        chown radxa:radxa "$BASH_PROFILE"
        echo "    Wrote ${BASH_PROFILE}"
    else
        echo "    ${BASH_PROFILE} already sources .bashrc, skipping"
    fi
fi

# Unattended security upgrades. Minus boxes run headless for weeks; Debian
# security patches should apply themselves. The policy (never auto-reboot,
# kernel/u-boot/Rockchip/GStreamer/Mesa/RKNN/Tailscale blacklisted) is
# written by the service at startup from src/unattended_upgrades.py and can
# be toggled from the web UI: Settings -> System Updates.
echo "[12/12] Installing unattended-upgrades..."
if ! dpkg -l 2>/dev/null | grep -q "^ii\s\+unattended-upgrades"; then
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq unattended-upgrades || \
        echo "    WARNING: unattended-upgrades install failed; later: sudo apt-get install unattended-upgrades"
else
    echo "    unattended-upgrades already installed"
fi

echo ""
echo "=== Installation Complete ==="
echo ""
echo "Access:"
echo "  Web UI:  http://${HOSTNAME}.local:80"
echo "  Stream:  http://${HOSTNAME}.local:9090/stream"
echo ""
echo "Commands:"
echo "  Start:   sudo systemctl start ${SERVICE_NAME}"
echo "  Stop:    sudo systemctl stop ${SERVICE_NAME}"
echo "  Status:  sudo systemctl status ${SERVICE_NAME}"
echo "  Logs:    sudo journalctl -u ${SERVICE_NAME} -f"
echo "  Disable: sudo systemctl disable ${SERVICE_NAME}"
echo ""
echo "The service will auto-start on boot."
echo "To start now: sudo systemctl start ${SERVICE_NAME}"

if [ "${REBOOT_NEEDED}" -eq 1 ]; then
    echo ""
    echo "*** REBOOT REQUIRED ***"
    echo "Hardware overlays were enabled (SPI / PWM3). They take effect on the"
    echo "next boot. Reboot when convenient, then status LEDs and the IR"
    echo "transmitter will be available. Until reboot, the web-UI toggles for"
    echo "those features will report 'hardware not available'."
fi
