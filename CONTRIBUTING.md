# Contributing

Thanks for helping improve this tutorial. Make sure you follow these
instructions before you open a PR:

## Pre-commit checks

We use [pre-commit](https://pre-commit.com/) to run a small set of checks before
each commit:

- `pre-commit-hooks` for basic file hygiene: valid YAML, final newlines,
  trailing whitespace, and merge conflict markers.
- `markdownlint-cli2` for Markdown structure and consistency.
- `ruff` and `ruff-format` for Python scripts that will be added later.

Markdown line length is not enforced because tutorial prose often reads better
without hard wrapping. Markdown hard line breaks using two trailing spaces are
preserved by the whitespace hook.

## Setup with uv

From the repository root, create and activate a `uv` virtual environment:

```bash
uv venv
source .venv/bin/activate
```

If you already have an active `uv` virtual environment, skip that step.

Install `pre-commit` into the environment and enable the Git hook:

```bash
uv pip install pre-commit
pre-commit install
```

This creates a local `.git/hooks/pre-commit` hook. After that, every
`git commit` runs the configured checks before the commit is created.

Run the checks manually before opening a PR:

```bash
pre-commit run --all-files
```

If a hook rewrites files, review the changes, stage them, and commit again.

GitHub Actions runs the same pre-commit checks on pushes to `main` and on pull
requests.
