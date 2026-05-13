# Contributing

We welcome contributions to EKS!
If you have found a bug or would like to request a change, please
[open an issue](https://github.com/paninski-lab/eks/issues) first.

We strive to maintain a welcoming environment for all contributors.
See our [code of conduct](CODE_OF_CONDUCT.md) for more information.

## Development setup

[Fork](https://guides.github.com/activities/forking/#fork) the repo, then clone your fork and
install in editable mode with dev dependencies:

```bash
pip install -e ".[dev]"
pre-commit install
```

`pre-commit install` registers the ruff linting hook so it runs automatically on each commit.
You only need to do this once per clone.

## Running the tests

```bash
pytest
```

Integration tests compare outputs against golden files stored in `tests/integration/golden/`.
These files are hosted remotely and downloaded automatically when the test suite runs.
If your change intentionally alters numerical outputs, you can regenerate your local copies with:

```bash
pytest --update-golden
```

However, the remote golden files can only be updated by a repo maintainer.
If your PR requires golden file changes, note this in the PR description and a maintainer
will update them before merging.

## Linting

We use [ruff](https://docs.astral.sh/ruff/) for linting and import sorting.
The pre-commit hook runs it automatically, but you can also run it manually:

```bash
ruff check eks tests          # check for violations
ruff check --fix eks tests    # auto-fix where possible
```

Configuration lives in `[tool.ruff]` in `pyproject.toml`.

## Pull requests

- Keep PRs focused — one feature or fix per PR
- Include tests for new functionality
- Ensure `pytest` and `ruff check` both pass before opening the PR
- Open against the `main` branch
