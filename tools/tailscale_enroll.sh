#!/bin/bash
# Join this unit to the KingBlocker tailnet (minus-tailscale-enroll.timer).
#
# Every unit gets its own Tailscale node: a cloned card has no node state
# (tools/prepare_for_clone.sh wipes it), so on first boot this logs in with
# the shared auth key and a name unique to the board:
#     kingblocker-<last 4 hex of the Wi-Fi MAC>     tag: tag:kingblocker
#
#   /etc/minus/tailscale-authkey   reusable, tagged auth key (root, 0600;
#                                  never in git). No key -> no enrollment.
#   /etc/minus/tailscale-tags      optional, default tag:kingblocker
#
# --accept-dns=false: Tailscale never rewrites /etc/resolv.conf, so a
# tailscaled problem cannot take the unit's DNS down (the units need no
# tailnet names). NetworkManager keeps DNS, with public backups added to
# every Wi-Fi profile (WiFiManager.ensure_profile_defaults).
#
# Offline-safe: nothing waits on this (only its timer starts it); every
# Tailscale/network call is time-limited and hard-killed; with no key or no
# network it exits and the timer tries again in 2 minutes.
# Log: journalctl -u minus-tailscale-enroll

set -u
KEY_FILE=/etc/minus/tailscale-authkey
TAGS="$(tr -d ' \n' < /etc/minus/tailscale-tags 2>/dev/null)"
TAGS="${TAGS:-tag:kingblocker}"
log() { echo "[ts-enroll] $*"; }
once() {   # log a message once per boot
    local m="/run/minus-ts-enroll.$1"; shift
    [ -e "$m" ] || { touch "$m"; log "$*"; }
}

command -v tailscale >/dev/null || { once notailscale "tailscale not installed"; exit 0; }

state="$(timeout -k 5 10 tailscale status --json 2>/dev/null |
         python3 -c 'import json,sys; print(json.load(sys.stdin).get("BackendState",""))' 2>/dev/null)"

if [ "$state" = "Running" ]; then
    # Enrolled. Keep Tailscale out of DNS (also on units enrolled by hand).
    if [ ! -e /run/minus-ts-enroll.dnsoff ]; then
        timeout -k 5 15 tailscale set --accept-dns=false >/dev/null 2>&1 &&
            touch /run/minus-ts-enroll.dnsoff
        once running "enrolled as $(timeout -k 5 10 tailscale status --self --json 2>/dev/null |
            python3 -c 'import json,sys; d=json.load(sys.stdin)["Self"]; print(d["HostName"], d["TailscaleIPs"][0])' 2>/dev/null)"
    fi
    exit 0
fi
if [ "$state" = "Stopped" ]; then
    once stopped "tailscale is logged in but stopped (tailscale down); leaving it"
    exit 0
fi

# NeedsLogin / NoState / unknown: enroll if we can.
if [ ! -s "$KEY_FILE" ]; then
    once nokey "not enrolled and no auth key at $KEY_FILE"
    exit 0
fi
if ! timeout -k 5 10 nmcli -t -f STATE general 2>/dev/null | grep -q '^connected'; then
    once offline "not enrolled; waiting for a network"
    exit 0
fi

mac=""
for dev in /sys/class/net/wl* /sys/class/net/en* /sys/class/net/eth*; do
    [ -e "$dev/address" ] && { mac="$(tr -d ':\n' < "$dev/address")"; break; }
done
suffix="${mac: -4}"
[ -n "$suffix" ] || suffix="$(cut -c1-4 /etc/machine-id)"
NAME="kingblocker-${suffix}"

log "enrolling as $NAME ($TAGS)"
if timeout -k 10 90 tailscale up --reset --auth-key="file:$KEY_FILE" \
        --hostname="$NAME" --advertise-tags="$TAGS" --accept-dns=false \
        --timeout=60s >/dev/null 2>&1; then
    rm -f /run/minus-ts-enroll.*
    touch /run/minus-ts-enroll.dnsoff
    log "enrolled: $(timeout -k 5 10 tailscale ip -4 2>/dev/null)"
else
    log "enrollment failed (network, expired key or wrong tag?); retrying in 2 min"
fi
exit 0
