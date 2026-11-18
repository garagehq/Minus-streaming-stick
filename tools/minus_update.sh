#!/bin/bash
# Minus auto-update (minus-update.service, started by minus-update.timer).
#
# A few minutes after boot, once the unit is online, bring this checkout to
# the tip of its release branch and rebuild ustreamer if its branch moved,
# then restart Minus if anything changed. Runs once per boot: until it has
# reached the remote, the timer retries every 5 minutes.
#
#   Minus:     origin/$MINUS_UPDATE_BRANCH       (default feature/kingblocker-9000)
#   ustreamer: origin/$MINUS_USTREAMER_BRANCH    (default asr-first)
#
# Development-safe:
#   - local changes to tracked files are stashed first (git stash list);
#   - local commits that are not on the remote branch are kept on a
#     minus-update-backup-<time> branch before the checkout is reset;
#   - touch /etc/minus/no-auto-update to turn it off on a dev box.
# Output goes to the journal: journalctl -u minus-update

set -u

# The checkout to update: this script's repo, or MINUS_DIR (to run a newer
# copy of this script against an older checkout).
MINUS_DIR="${MINUS_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_USER="${MINUS_UPDATE_USER:-radxa}"
RUN_HOME="$(getent passwd "$RUN_USER" | cut -d: -f6)"
BRANCH="${MINUS_UPDATE_BRANCH:-feature/kingblocker-9000}"
US_DIR="${RUN_HOME}/ustreamer-garagehq"
US_BRANCH="${MINUS_USTREAMER_BRANCH:-asr-first}"
US_BIN="${RUN_HOME}/ustreamer-patched"
US_BUILT="${RUN_HOME}/.minus_ustreamer_built_rev"
STAMP=/run/minus-update.done

log() { echo "[minus-update] $*"; }
# setpriv rather than runuser/su: no PAM session, so no journal lines per call.
AS_USER=(setpriv --reuid="$RUN_USER" --regid="$RUN_USER" --init-groups
         env HOME="$RUN_HOME" USER="$RUN_USER")
as_user() { "${AS_USER[@]}" "$@"; }
git_m() { as_user git -C "$MINUS_DIR" "$@"; }
git_u() { as_user git -C "$US_DIR" "$@"; }

if [ -e /etc/minus/no-auto-update ]; then
    log "disabled (/etc/minus/no-auto-update)"; exit 0
fi
if [ -e "$STAMP" ]; then
    exit 0                                    # already updated this boot
fi
if ! nmcli -t -f STATE general 2>/dev/null | grep -q '^connected'; then
    log "not online yet; will retry"; exit 0
fi

# Bring a checkout to origin/<branch>. $1 = git wrapper, $2 = branch, $3 = label.
sync_checkout() {
    local g="$1" br="$2" name="$3" ts
    ts="$(date +%Y%m%d-%H%M%S)"
    if [ -n "$($g status --porcelain --untracked-files=no)" ]; then
        $g stash push -q -m "minus-update $ts" && log "$name: stashed local changes (git stash list)"
    fi
    if [ "$($g rev-parse --abbrev-ref HEAD)" != "$br" ]; then
        if $g rev-parse -q --verify "refs/heads/$br" >/dev/null; then
            $g checkout -q "$br"
        else
            $g checkout -q -b "$br" --track "origin/$br"
        fi
        log "$name: switched to $br"
    fi
    if ! $g merge -q --ff-only "origin/$br" 2>/dev/null; then
        $g branch -q "minus-update-backup-$ts"
        $g reset -q --hard "origin/$br"
        log "$name: local commits not on origin/$br kept on minus-update-backup-$ts; reset to origin"
    fi
}

# ---- Minus -----------------------------------------------------------------
if ! timeout 60 "${AS_USER[@]}" git -C "$MINUS_DIR" fetch -q origin "$BRANCH"; then
    log "can't reach origin; will retry"; exit 0
fi
touch "$STAMP"
OLD="$(git_m rev-parse HEAD)"
sync_checkout git_m "$BRANCH" "Minus"
NEW="$(git_m rev-parse HEAD)"
CHANGED=0
if [ "$OLD" != "$NEW" ]; then
    CHANGED=1
    log "Minus: $(git_m log --oneline -1 "$OLD") -> $(git_m log --oneline -1 "$NEW")"
    if ! git_m diff --quiet "$OLD" "$NEW" -- minus.service; then
        install -m 644 "$MINUS_DIR/minus.service" /etc/systemd/system/minus.service
        systemctl daemon-reload
        log "Minus: reinstalled minus.service"
    fi
    for unit in minus-update.service minus-update.timer; do
        if ! git_m diff --quiet "$OLD" "$NEW" -- "$unit"; then
            install -m 644 "$MINUS_DIR/$unit" "/etc/systemd/system/$unit"
            systemctl daemon-reload
        fi
    done
    if ! git_m diff --quiet "$OLD" "$NEW" -- journald/90-kingblocker.conf; then
        install -d /etc/systemd/journald.conf.d
        install -m 644 "$MINUS_DIR/journald/90-kingblocker.conf" /etc/systemd/journald.conf.d/
        systemctl restart systemd-journald
        log "Minus: updated journald limits"
    fi
else
    log "Minus: up to date ($(git_m log --oneline -1))"
fi

# ---- ustreamer (only rebuilt when its branch moved) ------------------------
if [ -d "$US_DIR/.git" ] && timeout 60 "${AS_USER[@]}" git -C "$US_DIR" fetch -q origin "$US_BRANCH"; then
    sync_checkout git_u "$US_BRANCH" "ustreamer"
    US_REV="$(git_u rev-parse HEAD)"
    if [ "$(cat "$US_BUILT" 2>/dev/null)" != "$US_REV" ]; then
        log "ustreamer: building $(git_u log --oneline -1)"
        if as_user make -s -C "$US_DIR" clean >/dev/null 2>&1 \
           && as_user make -s -C "$US_DIR" WITH_MPP=1 -j"$(nproc)" >/dev/null 2>&1 \
           && "$US_DIR/ustreamer" --version >/dev/null 2>&1; then
            [ -f "$US_BIN" ] && cp -p "$US_BIN" "$US_BIN.prev"
            install -o "$RUN_USER" -g "$RUN_USER" -m 755 "$US_DIR/ustreamer" "$US_BIN"
            echo "$US_REV" > "$US_BUILT"; chown "$RUN_USER:$RUN_USER" "$US_BUILT"
            CHANGED=1
            log "ustreamer: installed $US_BIN (previous kept as $US_BIN.prev)"
        else
            log "ustreamer: build FAILED; keeping the installed binary"
        fi
    fi
fi

if [ "$CHANGED" = 1 ]; then
    log "restarting minus"
    systemctl restart minus
fi
log "done"
