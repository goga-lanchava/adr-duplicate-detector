# Development & Git workflow

All commands assume you are in the project directory:

```bash
cd "C:/Users/gogal/Downloads/adr-duplicate-detector-fixed/adr-duplicate-detector"
```

This folder is the git repository root (the outer `adr-duplicate-detector-fixed`
folder is **not** tracked).

---

## One-time setup on a new machine

```bash
git clone https://github.com/goga-lanchava/adr-duplicate-detector.git
cd adr-duplicate-detector

python -m venv .venv
.venv\Scripts\activate            # PowerShell / cmd
# source .venv/bin/activate       # macOS / Linux / Git Bash

pip install -r requirements-dev.txt   # runtime deps + pytest
```

`.venv/` is already in `.gitignore`, so it never gets committed.

---

## Everyday loop

```bash
git pull                       # 1. get any remote changes first
# ... edit code ...
pytest -q                      # 2. make sure tests still pass
git status                     # 3. see what changed
git diff                       # 4. review the actual changes (q to quit)
git add -A                     # 5. stage everything...
git add app.py src/blocking.py #    ...or stage specific files
git commit -m "Short imperative summary of the change"
git push                       # 6. publish
```

### Writing commit messages

- First line: imperative mood, <= ~72 chars — "Fix age-conflict veto off-by-one",
  not "fixed stuff".
- Optional blank line, then a body explaining **why** if it isn't obvious.
- One commit = one logical change. Don't bundle an unrelated typo fix with a
  feature.

---

## Branches (recommended for anything non-trivial)

Working directly on `main` is fine for tiny fixes. For a feature or a risky
change, isolate it:

```bash
git switch -c feature/better-blocking     # create + switch to a new branch
# ... work, commit as many times as you like ...
git push -u origin feature/better-blocking
```

Then open a Pull Request on GitHub (Compare & pull request button, or
`https://github.com/goga-lanchava/adr-duplicate-detector/pulls`). Review the diff
there, then **Merge**. Afterwards:

```bash
git switch main
git pull                                  # bring the merge into local main
git branch -d feature/better-blocking      # delete the merged local branch
```

---

## Inspecting history

```bash
git log --oneline --graph --decorate -20   # compact recent history
git log -p app.py                          # full diffs touching one file
git show <commit-hash>                      # everything in one commit
git blame src/normalizer.py                 # who/when for each line
```

---

## Undoing things

| Situation | Command |
|---|---|
| Discard unstaged edits to a file | `git restore path/to/file` |
| Unstage a file (keep the edits) | `git restore --staged path/to/file` |
| Change the last commit message | `git commit --amend` (only if **not pushed**) |
| Add a forgotten file to the last commit | `git add f && git commit --amend --no-edit` (only if not pushed) |
| Undo the last commit, keep changes staged | `git reset --soft HEAD~1` |
| Undo the last commit, keep changes unstaged | `git reset HEAD~1` |
| Throw away the last commit entirely | `git reset --hard HEAD~1` (destructive) |
| Revert an already-pushed commit safely | `git revert <commit-hash>` (makes a new "undo" commit) |

Rule of thumb: `amend` / `reset` only on commits that haven't been pushed. Once a
commit is on GitHub, use `revert` instead so history stays consistent.

---

## Syncing when the push is rejected

If `git push` fails with "rejected — remote contains work you do not have":

```bash
git pull --rebase        # replay your local commits on top of the remote ones
# resolve conflicts if any, then:  git rebase --continue
git push
```

---

## Before every push — checklist

1. `pytest -q` passes (6 tests).
2. `git diff --staged` shows only what you intend to commit.
3. No secrets, tokens, or large data files staged (`real_faers_sample.csv` is the
   only data file that belongs in the repo).
4. Commit message describes the change, not the activity.
