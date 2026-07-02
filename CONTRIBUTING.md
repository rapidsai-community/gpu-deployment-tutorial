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

#### Signing Your Work

* We require that all contributors "sign-off" on their commits. This certifies that the contribution is your original work, or you have rights to submit it under the same license, or a compatible license.

  * Any contribution which contains commits that are not Signed-Off will not be accepted.

* To sign off on a commit you simply use the `--signoff` (or `-s`) option when committing your changes:
  ```bash
  $ git commit -s -m "Add cool feature."
  ```
  This will append the following to your commit message:
  ```
  Signed-off-by: Your Name <your@email.com>
  ```

* Full text of the DCO (https://developercertificate.org/):

  ```
    Developer Certificate of Origin
    Version 1.1

    Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

    Everyone is permitted to copy and distribute verbatim copies of this
    license document, but changing it is not allowed.


    Developer's Certificate of Origin 1.1

    By making a contribution to this project, I certify that:

    (a) The contribution was created in whole or in part by me and I
        have the right to submit it under the open source license
        indicated in the file; or

    (b) The contribution is based upon previous work that, to the best
        of my knowledge, is covered under an appropriate open source
        license and I have the right under that license to submit that
        work with modifications, whether created in whole or in part
        by me, under the same open source license (unless I am
        permitted to submit under a different license), as indicated
        in the file; or

    (c) The contribution was provided directly to me by some other
        person who certified (a), (b) or (c) and I have not modified
        it.

    (d) I understand and agree that this project and the contribution
        are public and that a record of the contribution (including all
        personal information I submit with it, including my sign-off) is
        maintained indefinitely and may be redistributed consistent with
        this project or the open source license(s) involved.
  ```
