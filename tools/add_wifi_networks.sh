#!/bin/bash
# Add the factory Wi-Fi networks to NetworkManager (KingBlocker9000 units).
#
# Each network is added as a saved profile only if no profile with that name
# exists yet. Adding a profile never connects to it: NetworkManager joins it
# on its own only when it is in range and no better saved network is (its
# autoconnect priority is below the default 0 a user's own network gets
# from the Wi-Fi setup page). Run as root; install.sh calls it (step 12).
#
#   sudo tools/add_wifi_networks.sh
#
# NOTE: the passphrases below are in the repository in plain text.

set -e

# name|passphrase|autoconnect priority
NETWORKS=(
    "TestNetwork1337|testnet1337!|-10"
)

if [ "$EUID" -ne 0 ]; then
    echo "Please run as root: sudo $0"
    exit 1
fi
if ! command -v nmcli >/dev/null; then
    echo "    nmcli not found; skipping Wi-Fi networks"
    exit 0
fi

# A conf.d drop-in for this was tried and is rejected by NetworkManager 1.42
# ("unknown key"); remove it if an earlier version installed it.
rm -f /etc/NetworkManager/conf.d/90-kingblocker-autoconnect.conf

for entry in "${NETWORKS[@]}"; do
    IFS='|' read -r ssid psk prio <<< "$entry"
    if nmcli -t -f NAME connection show | grep -qxF "$ssid"; then
        echo "    Wi-Fi network already saved: $ssid"
        continue
    fi
    nmcli connection add type wifi con-name "$ssid" ifname '*' ssid "$ssid" \
        wifi-sec.key-mgmt wpa-psk wifi-sec.psk "$psk" \
        connection.autoconnect yes connection.autoconnect-priority "$prio" >/dev/null
    echo "    Wi-Fi network saved (not connected): $ssid"
done

# Retry every saved Wi-Fi network forever (NetworkManager's default is 4
# attempts, then ~5 min without auto-connecting). Minus re-applies this to
# networks added later (WiFiManager.ensure_retry_forever).
nmcli -t -f UUID,TYPE connection show | awk -F: '$2=="802-11-wireless"{print $1}' |
    while read -r u; do nmcli connection modify "$u" connection.autoconnect-retries 0; done
echo "    Saved Wi-Fi networks retry forever"
