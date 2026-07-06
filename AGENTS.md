# AGENTS.md — open-fable

Guidance for **AI agents** (Codex, Claude Code, and any Coven familiar) opening
pull requests against this repo. Humans: your canonical guide is
[`CONTRIBUTING.md`](CONTRIBUTING.md) — this is the agent-specific layer on top.

> **Read first:** [`README.md`](README.md) for what this repo is, and
> [`CONTRIBUTING.md`](CONTRIBUTING.md) for the full contribution bar.

---

## What this repo is (one line)

OpenFable is a Coven-flavored **Recurrent-Depth Transformer** for narrative
reasoning, character consistency, and long-form story coherence (Python).

## Branch & PR workflow (all agents)

- **Never push to `main`.** Every change lands via a PR with green CI. Branch
  from current `origin/main`.
- **Fresh branch per task**; use a worktree if multiple sessions may touch this
  repo:
  ```sh
  git fetch origin main
  git worktree add -b <branch> /tmp/openfable-<branch> origin/main
  ```
- Keep the diff scoped to one concern; conventional-commit subjects (`feat:`,
  `fix:`, `docs:`, `chore:`, `refactor:`).
- After merge: delete the remote branch, remove your local worktree/branch.

## Checks — run locally before opening the PR

```sh
ruff check .        # lint
black --check .     # formatting
pytest              # tests live under tests/
```

(Install the dev extras first: `pip install -e ".[dev]"`.) Fix failures rather
than skipping them.

## Repo-specific invariants (don't break these)

- This is a **research model** repo. Keep training/inference determinism where
  the code relies on it (seeds, dtype, device handling) and document any change
  that affects reproducibility.
- Keep model/config changes and their tests together; don't land shape/interface
  changes without updating the affected tests.

## Attribution — credit contributors correctly

When you re-land or build on someone else's work (a fork PR, an issue author's
proposal, a co-author), **credit the human contributor with a working
GitHub-linked trailer** so they appear in the contributors graph:

```
Co-authored-by: Full Name <ID+username@users.noreply.github.com>
```

- Use the **numeric-id no-reply form**. Get the id with `gh api users/<login> --jq .id`.
- **Never** use a machine or `.local` email in a co-author trailer — it links to
  no account and gives **zero** credit.
- When a squash-merge folds a contributor's PR into an internal branch, preserve
  their `Co-authored-by:` line in the squash commit message.
- Credit **people**, not AI tools.

## Secrets & safety

- Never commit secrets, tokens, model weights, or private emails. Use
  `*.noreply.github.com` for attribution.
- Don't disable CI gates or branch protection to land a change; surface the
  blocker instead.

## Claude Code

`CLAUDE.md` points here — this file is the source of truth for both.
