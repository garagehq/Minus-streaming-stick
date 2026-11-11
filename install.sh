#!/bin/bash
# KingBlocker9000 (Minus) installer for a ROCK 5B / 5B+ on Radxa Debian 12
# (kernel 6.1, rk2410 images). Takes a fresh image to a running service:
#
#   git clone -b <branch> https://github.com/garagehq/Minus-streaming-stick.git ~/Minus
#   cd ~/Minus && sudo ./install.sh
#
# Idempotent: re-running only does what is missing. Steps:
#   1  device-tree overlays (status LED SPI, IR PWM; only when asked for)
#   2  apt packages (GStreamer + Rockchip plugins, build deps, fonts, tools)
#   3  Python packages (pip --user for the radxa user; the root service
#      reads them through PYTHONPATH in minus.service)
#   4  ustreamer (garagehq fork, MPP JPEG encoder) -> /home/radxa/ustreamer-patched
#   5  SenseVoice ASR model (stock from Hugging Face if no model is installed)
#   6  HDMI-RX EDID (24-30 Hz only, so sources send 4K30)
#   7  groups (gpio/spi/video/audio) for the radxa user
#   8  hostname + mDNS (only when MINUS_HOSTNAME is set)
#   9  silent boot, auto-login (tty2 + desktop), never sleep/lock, desktop off
#   10 systemd service
#   11 screenshot dirs, login shell, unattended-upgrades
#
# Environment:
#   MINUS_HOSTNAME=name      set the hostname (default: keep the current one)
#   MINUS_USTREAMER_BRANCH   ustreamer branch to build (default asr-first)
#   MINUS_SKIP_APT=1         skip apt (offline re-run)
#   MINUS_START=1            start the service at the end
#   MINUS_KEEP_DESKTOP=1     leave the display manager enabled
#   MINUS_VERBOSE_BOOT=1     keep kernel/boot text and the tty1 login on screen
#   MINUS_STATUS_LEDS=1      enable the WS2812B status strip (SPI0 overlay)
#   MINUS_IR=1               enable the IR transmitter (PWM3 overlay)
# Kingblocker units only have the red mute LED (pins 40/39, plain GPIO, no
# overlay), so the strip and IR are off unless asked for.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_NAME="minus"
SERVICE_FILE="${SCRIPT_DIR}/minus.service"
RUN_USER="radxa"
RUN_HOME="$(getent passwd "${RUN_USER}" | cut -d: -f6)"
USTREAMER_REPO="https://github.com/garagehq/ustreamer.git"
USTREAMER_BRANCH="${MINUS_USTREAMER_BRANCH:-asr-first}"
USTREAMER_SRC="${RUN_HOME}/ustreamer-garagehq"
USTREAMER_BIN="${RUN_HOME}/ustreamer-patched"
ASR_MODELS="${RUN_HOME}/asr_models"
SENSEVOICE_HF="https://huggingface.co/happyme531/SenseVoiceSmall-RKNN2/resolve/main"
TOTAL=11

step() { echo; echo "[$1/${TOTAL}] $2"; }
as_user() { sudo -u "${RUN_USER}" -H "$@"; }

echo "=== KingBlocker9000 installation ==="

if [ "$EUID" -ne 0 ]; then
    echo "Please run as root: sudo $0"
    exit 1
fi
if [ -z "${RUN_HOME}" ] || [ ! -d "${RUN_HOME}" ]; then
    echo "ERROR: user ${RUN_USER} not found"
    exit 1
fi
if [ ! -f "$SERVICE_FILE" ]; then
    echo "ERROR: Service file not found: $SERVICE_FILE"
    exit 1
fi

REBOOT_NEEDED=0

# ---------------------------------------------------------------------------
step 1 "Enabling hardware device-tree overlays..."
# IR transmitter (PWM3 on pin 38) and the WS2812B status strip (SPI0_MOSI_M2
# on pin 19). Both ship in /boot/dtbo with a .disabled suffix. The red mute
# LED on pin 40 is plain GPIO and needs no overlay.
NEED_UBOOT_UPDATE=0
enable_overlay() {
    local f="/boot/dtbo/$1.dtbo"
    if [ -f "${f}.disabled" ] && [ ! -f "${f}" ]; then
        echo "    Enabling $1 ($2)"
        mv "${f}.disabled" "${f}"
        NEED_UBOOT_UPDATE=1
    elif [ -f "${f}" ]; then
        echo "    $1 already enabled"
    else
        echo "    WARNING: $1 not found in /boot/dtbo ($2 unavailable)"
    fi
}
if [ "${MINUS_STATUS_LEDS:-0}" = "1" ]; then
    enable_overlay rk3588-spi0-m2-cs0-spidev "status LED strip"
else
    # The strip controller defaults to on; save it as off (the service runs
    # as root, so its settings live in /root).
    [ -f /root/.minus_status_leds.json ] || echo '{"enabled": false}' > /root/.minus_status_leds.json
    echo "    status LED strip: skipped (MINUS_STATUS_LEDS=1 to enable)"
fi
if [ "${MINUS_IR:-0}" = "1" ]; then
    enable_overlay rk3588-pwm3-m1 "IR transmitter"
else
    echo "    IR transmitter: skipped (MINUS_IR=1 to enable)"
fi
if [ "${NEED_UBOOT_UPDATE}" -eq 1 ]; then
    echo "    Regenerating /boot/extlinux/extlinux.conf..."
    /usr/sbin/u-boot-update
    REBOOT_NEEDED=1
fi

# ---------------------------------------------------------------------------
step 2 "Installing apt packages..."
APT_PACKAGES=(
    # GStreamer video/audio pipelines (+ Rockchip MPP decoder, kmssink)
    gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good
    gstreamer1.0-plugins-bad gstreamer1.0-alsa gstreamer1.0-rockchip1
    gir1.2-gstreamer-1.0 gir1.2-gst-plugins-base-1.0 python3-gi python3-gst-1.0
    # ustreamer build (MPP JPEG encoder, FreeType overlays)
    # librga-dev: RGA does the 4K BGR24->NV12 conversion in hardware; without
    # it ustreamer converts on the CPU (~280% CPU, 20-30 fps instead of 30)
    build-essential git pkg-config librockchip-mpp-dev librga-dev libfreetype-dev
    libjpeg-dev libevent-dev libbsd-dev
    # Python packages taken from Debian
    python3-pip python3-numpy python3-pil python3-yaml python3-psutil
    python3-spidev python3-requests libsndfile1
    # Tools the service shells out to
    v4l-utils ffmpeg curl imagemagick libdrm-tests alsa-utils adb
    network-manager iw psmisc
    # Overlay fonts
    fonts-dejavu-core fonts-ibm-plex
    # mDNS + unattended security updates
    avahi-daemon unattended-upgrades
)
if [ "${MINUS_SKIP_APT:-0}" = "1" ]; then
    echo "    skipped (MINUS_SKIP_APT=1)"
else
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    # One package missing from a mirror shouldn't stop the rest.
    # Radxa images ship android-libcutils from bookworm-backports, and the
    # bookworm adb pins an older one ("held broken packages"): take adb from
    # backports too. Done first, so the bulk install below finds it present.
    if ! dpkg -s adb >/dev/null 2>&1 && ! apt-get install -s adb >/dev/null 2>&1; then
        apt-get install -y -qq -t bookworm-backports adb || echo "    WARNING: could not install adb"
    fi
    if ! apt-get install -y -qq "${APT_PACKAGES[@]}"; then
        echo "    Bulk install failed; retrying one by one..."
        for p in "${APT_PACKAGES[@]}"; do
            apt-get install -y -qq "$p" >/dev/null 2>&1 || echo "    WARNING: could not install $p"
        done
    fi
fi
if [ ! -e /usr/lib/librknnrt.so ]; then
    echo "    WARNING: /usr/lib/librknnrt.so missing (rknpu2 runtime); OCR/ASR on the NPU will not load."
    echo "             Install Radxa's rknpu2 package: sudo apt-get install rknpu2-rk3588"
fi

# ---------------------------------------------------------------------------
step 3 "Installing Python packages for ${RUN_USER}..."
# Pinned to the versions on the reference box (minus-2).
as_user pip3 install --user --break-system-packages --no-warn-script-location -q -r "${SCRIPT_DIR}/requirements.txt"
as_user python3 -c "import cv2, numpy, flask, androidtv, pyclipper, shapely, rknnlite, kaldi_native_fbank, sentencepiece, soundfile, onnxruntime, yaml" \
    && echo "    Python imports OK"

# ---------------------------------------------------------------------------
step 4 "Building ustreamer (${USTREAMER_BRANCH})..."
if [ ! -d "${USTREAMER_SRC}/.git" ]; then
    as_user git clone -q -b "${USTREAMER_BRANCH}" "${USTREAMER_REPO}" "${USTREAMER_SRC}"
else
    as_user git -C "${USTREAMER_SRC}" fetch -q origin "${USTREAMER_BRANCH}"
    as_user git -C "${USTREAMER_SRC}" checkout -q "${USTREAMER_BRANCH}"
    as_user git -C "${USTREAMER_SRC}" merge -q --ff-only "origin/${USTREAMER_BRANCH}" || true
fi
# Build from clean: the Makefile picks up RGA only when librga-dev is
# present, and make would otherwise keep objects built without it.
as_user make -s -C "${USTREAMER_SRC}" clean >/dev/null
as_user make -s -C "${USTREAMER_SRC}" WITH_MPP=1 -j"$(nproc)"
if ldd "${USTREAMER_SRC}/ustreamer" | grep -q librga; then
    echo "    RGA hardware colour conversion: on"
else
    echo "    WARNING: ustreamer built without RGA (is librga-dev installed?)"
fi
install -o "${RUN_USER}" -g "${RUN_USER}" -m 755 "${USTREAMER_SRC}/ustreamer" "${USTREAMER_BIN}"
echo "    ${USTREAMER_BIN} ($(${USTREAMER_BIN} --version 2>/dev/null || echo '?'))"

# ---------------------------------------------------------------------------
step 5 "Checking the SenseVoice ASR model..."
# The fine-tuned model (sensevoice-rknn-minus-ft) is not public; copy it from
# another box into ${ASR_MODELS}. Without it the stock model is used.
if [ -f "${ASR_MODELS}/sensevoice-rknn-minus-ft/sense-voice-encoder.rknn" ]; then
    echo "    fine-tuned model present"
elif [ -f "${ASR_MODELS}/sensevoice-rknn/sense-voice-encoder.rknn" ]; then
    echo "    stock model present (fine-tuned model not installed)"
else
    echo "    Downloading stock SenseVoiceSmall-RKNN2 (~480 MB)..."
    as_user mkdir -p "${ASR_MODELS}/sensevoice-rknn"
    for f in am.mvn chn_jpn_yue_eng_ko_spectok.bpe.model embedding.npy \
             fsmn-am.mvn fsmn-config.yaml fsmnvad-offline.onnx \
             sensevoice_rknn.py sense-voice-encoder.rknn; do
        as_user curl -fsSL --retry 3 -o "${ASR_MODELS}/sensevoice-rknn/${f}" "${SENSEVOICE_HF}/${f}"
    done
    echo "    stock model installed"
fi

# ---------------------------------------------------------------------------
step 6 "Installing the HDMI-RX EDID (max 4K30 / 1080p60)..."
# Sources pick the highest refresh the EDID offers. 4K60 runs the SoC at
# 80-85 C, so the EDID offers 4K 24/25/30 and 1080p 24-60 but no 4K60: a
# source that needs 60 Hz gets 1080p60, the Google TV 1080p60 or 4K30. The
# old 24-30 Hz-only EDID (hdmirx-edid-30hz.bin) left a 60 Hz-only source
# sending nothing at all (TV: No Signal). Applied at boot, before minus.
install -d /etc/minus
install -m 644 "${SCRIPT_DIR}/hdmirx/hdmirx-edid-4k30-1080p60.bin" /etc/minus/
install -m 644 "${SCRIPT_DIR}/hdmirx/hdmirx-edid-30hz.bin" /etc/minus/
install -m 644 "${SCRIPT_DIR}/hdmirx/hdmirx-edid-default.bin" /etc/minus/
install -m 644 "${SCRIPT_DIR}/minus-hdmirx-edid.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable -q minus-hdmirx-edid.service
# With the input capped by the EDID, pass the source's own frame rate
# through (1080p60 stays 60 fps, 4K30 stays 30) instead of ustreamer's
# default 30 fps cap.
install -d /etc/systemd/system/minus.service.d
cat > /etc/systemd/system/minus.service.d/stream-fps.conf <<'CONF'
[Service]
Environment=MINUS_STREAM_FPS=0
CONF

# ---------------------------------------------------------------------------
step 7 "Adding ${RUN_USER} to hardware groups..."
for g in gpio spi spidev video audio render; do
    if getent group "$g" >/dev/null; then
        id -nG "${RUN_USER}" | tr ' ' '\n' | grep -qx "$g" || usermod -aG "$g" "${RUN_USER}"
    fi
done
echo "    $(id -nG "${RUN_USER}")"

# ---------------------------------------------------------------------------
step 8 "Hostname and mDNS..."
if [ -n "${MINUS_HOSTNAME:-}" ]; then
    hostnamectl set-hostname "${MINUS_HOSTNAME}"
    if grep -q "^127.0.1.1" /etc/hosts; then
        sed -i "s/^127.0.1.1.*/127.0.1.1\t${MINUS_HOSTNAME}/" /etc/hosts
    else
        echo -e "127.0.1.1\t${MINUS_HOSTNAME}" >> /etc/hosts
    fi
fi
systemctl enable -q avahi-daemon 2>/dev/null || true
systemctl restart avahi-daemon 2>/dev/null || true
echo "    Hostname: $(hostname)  ->  http://$(hostname).local"

# ---------------------------------------------------------------------------
step 9 "Auto-login, no sleep / screen lock, desktop off..."
# The box runs unattended: it must never suspend, lock or sit at a login
# prompt. Everything here is applied whether or not the desktop is kept.

# Never suspend/hibernate; logind never idles out or kills the session.
systemctl mask -q sleep.target suspend.target hibernate.target hybrid-sleep.target
install -d /etc/systemd/logind.conf.d
cat > /etc/systemd/logind.conf.d/kingblocker.conf <<'EOF2'
[Login]
IdleAction=ignore
HandleSuspendKey=ignore
HandleHibernateKey=ignore
HandleLidSwitch=ignore
HandleLidSwitchExternalPower=ignore
KillUserProcesses=no
EOF2

# Silent boot: the HDMI screen stays black from the bootloader until Minus
# draws. Kernel and systemd messages go to the serial debug port only
# (ttyFIQ0 stays in the cmdline), no cursor, no Tux logo. MINUS_VERBOSE_BOOT=1
# leaves the boot text on screen.
if [ "${MINUS_VERBOSE_BOOT:-0}" != "1" ] && [ -f /etc/kernel/cmdline ]; then
    CMDLINE="$(cat /etc/kernel/cmdline)"
    NEW="$(echo "${CMDLINE}" | sed -E 's/(^| )console=tty[0-9]+( |$)/ /g; s/(^| )loglevel=[0-9]+( |$)/ /g; s/  +/ /g; s/^ //; s/ $//')"
    for opt in quiet loglevel=0 vt.global_cursor_default=0 logo.nologo \
               systemd.show_status=false rd.systemd.show_status=false \
               udev.log_level=3 rd.udev.log_level=3; do
        case " ${NEW} " in *" ${opt} "*) ;; *) NEW="${NEW} ${opt}" ;; esac
    done
    if [ "${NEW}" != "${CMDLINE}" ]; then
        cp /etc/kernel/cmdline /etc/kernel/cmdline.bak-minus
        echo "${NEW}" > /etc/kernel/cmdline
        /usr/sbin/u-boot-update >/dev/null
        REBOOT_NEEDED=1
        echo "    quiet boot: kernel/systemd text off the screen (backup: /etc/kernel/cmdline.bak-minus)"
    fi
    # No login prompt or shell on tty1 (the screen Minus takes over). A
    # keyboard user gets an auto-logged-in shell on tty2 (Ctrl+Alt+F2).
    rm -rf /etc/systemd/system/getty@tty1.service.d
    systemctl mask -q getty@tty1.service autovt@tty1.service
    CONSOLE_TTY=tty2
else
    systemctl unmask -q getty@tty1.service autovt@tty1.service 2>/dev/null || true
    CONSOLE_TTY=tty1
fi
# Console auto-login (no password prompt with a keyboard attached).
install -d /etc/systemd/system/getty@${CONSOLE_TTY}.service.d
cat > /etc/systemd/system/getty@${CONSOLE_TTY}.service.d/autologin.conf <<EOF2
[Service]
ExecStart=
ExecStart=-/sbin/agetty --autologin ${RUN_USER} --noclear %I \$TERM
EOF2
systemctl enable -q getty@${CONSOLE_TTY}.service
# Empty the login banner (kernel version, Debian disclaimer).
: > /etc/motd
chmod -x /etc/update-motd.d/* 2>/dev/null || true

# Desktop login managers: auto-login, so the desktop (if used) never asks.
if [ -f /etc/gdm3/daemon.conf ]; then
    sed -i '/^AutomaticLoginEnable\s*=/d; /^AutomaticLogin\s*=/d' /etc/gdm3/daemon.conf
    sed -i "/^\[daemon\]/a AutomaticLoginEnable=true\nAutomaticLogin=${RUN_USER}" /etc/gdm3/daemon.conf
    echo "    gdm3 auto-login as ${RUN_USER}"
fi
if [ -d /usr/share/xsessions ] && [ -e /usr/share/xsessions/plasma.desktop ] && command -v sddm >/dev/null; then
    install -d /etc/sddm.conf.d
    printf '[Autologin]\nUser=%s\nSession=plasma\n' "${RUN_USER}" > /etc/sddm.conf.d/autologin.conf
    echo "    sddm auto-login as ${RUN_USER}"
fi

# KDE: no screen lock, no screen-off/suspend on idle.
if command -v kwriteconfig5 >/dev/null; then
    KW() { as_user kwriteconfig5 "$@"; }
    KW --file kscreenlockerrc --group Daemon --key Autolock false
    KW --file kscreenlockerrc --group Daemon --key LockOnResume false
    KW --file kscreenlockerrc --group Daemon --key Timeout 0
    # Idle suspend can't happen anyway (sleep.target is masked above); turn
    # off idle dim/screen-off. (kwriteconfig5 can't delete whole groups.)
    for prof in AC Battery LowBattery; do
        KW --file powermanagementprofilesrc --group "${prof}" --group DPMSControl --key idleTime --delete
        KW --file powermanagementprofilesrc --group "${prof}" --group DimDisplay --key idleTime --delete
        KW --file powermanagementprofilesrc --group "${prof}" --group SuspendSession --key idleTime --delete
    done
    echo "    KDE screen lock and idle power actions off"
fi
# GNOME (older images): same, through gsettings defaults.
if [ -d /etc/dconf/db ] && command -v dconf >/dev/null; then
    install -d /etc/dconf/db/local.d
    cat > /etc/dconf/db/local.d/00-kingblocker <<'EOF2'
[org/gnome/desktop/screensaver]
lock-enabled=false
idle-activation-enabled=false
[org/gnome/desktop/session]
idle-delay=uint32 0
[org/gnome/settings-daemon/plugins/power]
sleep-inactive-ac-type='nothing'
EOF2
    dconf update 2>/dev/null || true
fi

# The service takes over HDMI-out (it stops the display manager when it
# starts), so by default boot straight to the console. MINUS_KEEP_DESKTOP=1
# keeps the desktop for development.
if [ "${MINUS_KEEP_DESKTOP:-0}" = "1" ]; then
    echo "    keeping the desktop (MINUS_KEEP_DESKTOP=1)"
else
    for dm in sddm gdm3 gdm lightdm; do
        if systemctl list-unit-files "${dm}.service" 2>/dev/null | grep -q "^${dm}.service"; then
            systemctl disable -q "${dm}" 2>/dev/null || true
            echo "    disabled ${dm}"
        fi
    done
    systemctl set-default -q multi-user.target
fi
systemctl daemon-reload
echo "    console auto-login on ${CONSOLE_TTY} as ${RUN_USER}; sleep/suspend masked"

# ---------------------------------------------------------------------------
step 10 "Installing the systemd service..."
systemctl stop ${SERVICE_NAME} 2>/dev/null || true
install -m 644 "$SERVICE_FILE" /etc/systemd/system/${SERVICE_NAME}.service
systemctl daemon-reload
systemctl enable -q ${SERVICE_NAME}

# ---------------------------------------------------------------------------
step 11 "Screenshot dirs, login shell, unattended-upgrades..."
for d in ads non_ads vlm_spastic static; do
    mkdir -p "${SCRIPT_DIR}/screenshots/${d}"
done
chown -R "${RUN_USER}:${RUN_USER}" "${SCRIPT_DIR}/screenshots" 2>/dev/null || true

# Login shells (tmux, ssh) source ~/.bashrc so ~/.local/bin is on PATH.
BASH_PROFILE="${RUN_HOME}/.bash_profile"
if [ ! -f "$BASH_PROFILE" ] || ! grep -q "\.bashrc" "$BASH_PROFILE"; then
    printf 'if [ -f ~/.bashrc ]; then\n    . ~/.bashrc\nfi\n' >> "$BASH_PROFILE"
    chown "${RUN_USER}:${RUN_USER}" "$BASH_PROFILE"
fi
# The unattended-upgrades policy (no auto-reboot; kernel, u-boot, Rockchip,
# GStreamer, Mesa, RKNN held back) is written by the service at startup from
# src/unattended_upgrades.py; toggle in the web UI under System Updates.

if [ "${MINUS_START:-0}" = "1" ]; then
    systemctl start minus-hdmirx-edid.service 2>/dev/null || true
    systemctl start ${SERVICE_NAME}
fi

echo
echo "=== Installation complete ==="
echo
echo "  Web UI:  http://$(hostname).local   (or http://<ip>)"
echo "  Stream:  http://$(hostname).local:9090/stream"
echo "  Start:   sudo systemctl start ${SERVICE_NAME}"
echo "  Logs:    sudo journalctl -u ${SERVICE_NAME} -f"
echo
echo "Wiring: red mute LED  pin 40 -> 1k -> LED(+), LED(-) -> pin 39 (GND)"
if [ "${REBOOT_NEEDED}" -eq 1 ]; then
    echo
    echo "*** REBOOT REQUIRED: boot settings changed (overlays or kernel command line). ***"
fi
