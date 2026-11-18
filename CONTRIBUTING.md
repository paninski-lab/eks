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
pytest tests --ignore=tests/integration    # unit tests
pytest tests/integration                    # integration tests (run the CLI end to end)
```

Run the unit and integration tests as two separate commands. Running them in one `pytest`
session can change numerical outputs slightly, which makes the golden-file comparisons below fail.

### Golden files

Integration tests compare each command's CSV outputs against **golden files**: reference outputs
hosted as a zip in a GitHub release (`GOLDEN_URL` in `tests/conftest.py`). The zip is downloaded
automatically once per test session, and outputs must match to within `atol=1e-4`.

If your change intentionally alters numerical outputs, generate new golden files locally:

```bash
pytest tests/integration --generate-golden --golden-dir /tmp/eks_golden
```

This writes each test's CSVs to `/tmp/eks_golden/<test_name>/`. Only a repo maintainer can publish
new golden files (zip them, upload them to a release, and update `GOLDEN_URL`; see
[tests/integration/README.md](tests/integration/README.md) for the full steps). If your PR requires
golden file changes, say so in the PR description and a maintainer will update them before merging.

## Linting

We use [ruff](https://docs.astral.sh/ruff/) for linting and import sorting.
The pre-commit hook runs it automatically, but you can also run it manually:

```bash
ruff check eks tests          # check for violations
ruff check --fix eks tests    # auto-fix where possible
```

Configuration lives in `[tool.ruff]` in `pyproject.toml`.

## Type checking

We use [pyright](https://github.com/microsoft/pyright) for static type checking:

```bash
pyright eks
```

Configuration lives in `[tool.pyright]` in `pyproject.toml`.
Pyright runs in CI on every pull request but is not in the pre-commit hook (it
checks the whole project on every run, adding ~15s per commit).

## Before opening a pull request

Run each of these from the repository root and make sure they all pass:

1. Lint: `ruff check eks tests`
2. Type check: `pyright eks` (CI runs this on every pull request; the pre-commit hook does not)
3. Unit tests: `pytest tests --ignore=tests/integration`
4. Integration tests, in a separate command: `pytest tests/integration`
5. If your change intentionally alters numerical outputs, follow the golden-file steps above.

## Pull requests

- Keep PRs focused — one feature or fix per PR
- Include tests for new functionality
- Complete the [checklist above](#before-opening-a-pull-request) before opening the PR
- Open against the `main` branch
