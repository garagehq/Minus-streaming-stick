# Releasing Minus

Releases are cut by hand; there is no CI pipeline. A release is a `vMAJOR.MINOR.PATCH`
git tag on `main` plus a GitHub release. Boxes installed from a release see new ones
in **Settings → Minus Updates** and update with one button.

## Cut a release

```bash
git checkout main && git pull
tools/release.sh v0.2.0 --dry-run    # check what will happen
tools/release.sh v0.2.0
```

`tools/release.sh` refuses unless:

- the version looks like `vMAJOR.MINOR.PATCH` and is higher than every existing `v*` tag;
- you're on `main` with no local changes, exactly at `origin/main` (pull/push first);
- every Python file in the repo compiles.

It then creates an annotated tag, pushes it, and publishes a GitHub release marked
**Latest**. The notes are GitHub's generated notes since the previous release
(merged PRs + contributors), with a line naming the vision model from
[`models/vlm_model.json`](../models/vlm_model.json) and how to update a box.
It needs the GitHub CLI (`gh`) logged in with push access.

## Shipping a new vision model (e.g. Minus v0.2)

1. Upload the model to Hugging Face (for the Axera build, an `axera/` folder like
   [TheGarageDev/Minus-v0.1](https://huggingface.co/TheGarageDev/Minus-v0.1)).
2. Point [`models/vlm_model.json`](../models/vlm_model.json) at it: `name`,
   `hf_repo`, `subfolder`, and a new `local_dir` (so the old model stays on disk
   until the new one is verified):
   ```json
   {"name": "minus-v0.2", "hf_repo": "TheGarageDev/Minus-v0.2", "subfolder": "axera",
    "revision": "main", "local_dir": "/home/radxa/axera_models/minus-v0.2"}
   ```
   plus whatever code changes the new model needs (prompt, token ids, input size).
3. Merge to `main` and cut a release.

On each box, "Update now" pulls the code first, then runs the rest of the update
with the **new** code, which sees the new model isn't installed, downloads it
(checksums verified), and restarts on it. If Minus doesn't come back up, the code
and model roll back. Boxes running from source get it the same way from `main`.

## What boxes do with releases

- **On a release tag:** offered the latest GitHub release only if it is *newer* (it
  contains what's installed). A release marked "Latest" that is older is ignored: a
  newer commit always beats an older release.
- **Running from source** (a branch tracking `origin/main` or another branch):
  compared with that branch, never with a release, so they're never moved back to an
  older release.
- **A branch with local commits or no upstream:** shown, never updated automatically.

## Tested (2026-10-17)

The whole process was run on a throwaway public copy of this repo: dry run, release
v0.1.0, a box on `v0.1.0` reading it through the GitHub API, release v0.2.0 and the box
upgrading to it, v0.1.0 forced back to "Latest" with the box staying on v0.2.0, and
the script refusing a lower version, an existing tag, a bad version string, local
changes, an unpushed commit and the wrong branch. One thing to expect: GitHub's
unauthenticated API can return the previous "latest" for about a minute after
publishing, so a box that checks right away sees the new release on its next check.
