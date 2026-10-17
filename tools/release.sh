#!/bin/bash
# Cut a Minus release by hand (no CI): tag the current main and publish a GitHub
# release. Boxes installed from a release see it in Settings -> Minus Updates.
#
#   tools/release.sh v0.2.0               # tag + push + publish
#   tools/release.sh v0.2.0 --dry-run     # show what would happen
#   tools/release.sh v0.2.0 --repo owner/name --branch main   (test repos)
#
# Refuses unless: the version looks like vMAJOR.MINOR.PATCH and is higher than
# every existing v* tag; you're on the release branch with no local changes and
# exactly at origin/<branch>; every Python file compiles. Needs `gh` logged in.
# Release notes = GitHub's generated notes since the previous release + the VLM
# model named in models/vlm_model.json. See docs/RELEASING.md.
set -euo pipefail

VERSION="${1:-}"
shift || true
DRY=0
REPO=""
BRANCH="main"
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY=1 ;;
        --repo) REPO="$2"; shift ;;
        --branch) BRANCH="$2"; shift ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
    shift
done

die() { echo "release: $*" >&2; exit 1; }
run() { if [ $DRY -eq 1 ]; then echo "  (dry run) $*"; else "$@"; fi; }

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

[[ "$VERSION" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]] || die "usage: tools/release.sh vMAJOR.MINOR.PATCH [--dry-run]"
command -v gh >/dev/null || die "the GitHub CLI (gh) is required"
GH_REPO_ARGS=()
[ -n "$REPO" ] && GH_REPO_ARGS=(-R "$REPO")

git fetch --quiet --tags origin
[ "$(git rev-parse --abbrev-ref HEAD)" = "$BRANCH" ] || die "check out $BRANCH first"
[ -z "$(git status --porcelain --untracked-files=no)" ] || die "local changes to tracked files"
[ "$(git rev-parse HEAD)" = "$(git rev-parse "origin/$BRANCH")" ] || die "$BRANCH is not at origin/$BRANCH (pull/push first)"
git rev-parse -q --verify "refs/tags/$VERSION" >/dev/null && die "$VERSION already exists"

PREV="$(git tag -l 'v[0-9]*' | sort -V | tail -1)"
if [ -n "$PREV" ]; then
    HIGHEST="$(printf '%s\n%s\n' "$PREV" "$VERSION" | sort -V | tail -1)"
    [ "$HIGHEST" = "$VERSION" ] || die "$VERSION is not higher than the latest tag $PREV"
fi

echo "Checking that every Python file compiles..."
python3 - <<'EOF' || die "a Python file does not compile"
import pathlib, sys
bad = []
for p in pathlib.Path('.').rglob('*.py'):
    if any(part.startswith('.') or part in ('node_modules', 'venv') for part in p.parts):
        continue
    try:
        compile(p.read_bytes(), str(p), 'exec')
    except SyntaxError as e:
        bad.append(f'{p}: {e}')
print('\n'.join(bad) or 'ok')
sys.exit(1 if bad else 0)
EOF

MODEL_LINE=""
if [ -f models/vlm_model.json ]; then
    MODEL_LINE="$(python3 - <<'PYEOF'
import json
m = json.load(open('models/vlm_model.json'))
sub = m.get('subfolder') or '/'
print(f"**Vision model:** {m['name']} (https://huggingface.co/{m['hf_repo']}, `{sub}`) - "
      "boxes download it automatically from Settings -> Minus Updates.")
PYEOF
)"
fi
NOTES="$(mktemp)"
trap 'rm -f "$NOTES"' EXIT
{
    [ -n "$MODEL_LINE" ] && { echo "$MODEL_LINE"; echo; }
    # Only releases that contain the updater can be updated from the web UI.
    if git cat-file -e "HEAD:tools/minus_update.py" 2>/dev/null; then
        echo "Update a box from the web UI (Settings -> Minus Updates -> Update now), or from a shell:"
        echo '`sudo python3 tools/minus_update.py`'
    fi
} > "$NOTES"

echo "Releasing $VERSION at $(git rev-parse --short HEAD) (previous: ${PREV:-none})"
run git tag -a "$VERSION" -m "Minus $VERSION"
run git push origin "refs/tags/$VERSION"
GEN=(--generate-notes)
[ -n "$PREV" ] && GEN+=(--notes-start-tag "$PREV")
run gh release create "$VERSION" "${GH_REPO_ARGS[@]}" --verify-tag --title "Minus $VERSION" \
    "${GEN[@]}" --notes-file "$NOTES" --latest
[ $DRY -eq 1 ] || echo "Published: $(gh release view "$VERSION" "${GH_REPO_ARGS[@]}" --json url -q .url)"
