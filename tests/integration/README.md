from tests.conftest import GOLDEN_URL

# Integration Tests

These tests run the CLI end-to-end and optionally compare their CSV outputs
against a set of **golden files** — a reference snapshot of known-good outputs.

## How tests work

- **Without golden files**: tests only verify that the CLI exits without error (original behavior).
- **With golden files**: after each command runs, all CSV outputs are compared against the
  corresponding golden CSVs using `numpy.testing.assert_allclose` with `atol=1e-4`.

---

## Generating new golden files

Run this whenever you want to establish a new baseline (e.g. after an intentional change
to the algorithm, or when setting up golden files for the first time).

```bash
pytest tests/integration/ \
    --generate-golden \
    --golden-dir /tmp/eks_golden
```

This runs every integration test and copies the CSV outputs into
`/tmp/eks_golden/<test_name>/`. The directory structure will look like:

```
/tmp/eks_golden/
  test_singlecam_defaults/
    eks_singlecam.csv
  test_singlecam_fixed_smooth_param/
    eks_singlecam.csv
  test_multicam_defaults/
    multicam_top_results.csv
    multicam_bot_results.csv
  test_multicam_defaults_nonlinear/
    multicam_Cam-A_results.csv
    multicam_Cam-B_results.csv
    multicam_Cam-C_results.csv
    multicam_3d_results.csv
  ...
```

### Zip and upload

```bash
cd /tmp/eks_golden
zip -r eks_golden.zip .
```

Upload `eks_golden.zip` to your hosting location. The zip must have the test-name
folders at its root (no extra top-level wrapper directory) — the `cd` + `.` zip
command above ensures this.

### Update the URL in conftest.py

Once uploaded, copy the direct download URL of the zip asset from the GitHub release
and set it as `GOLDEN_URL` near the top of `tests/conftest.py`:

```python
GOLDEN_URL = 'https://github.com/paninski-lab/eks-test-fixtures/releases/download/vX/eks_golden.zip'
```

Commit this change so CI and other contributors pick it up automatically.

---

## Running tests with golden comparison

```bash
pytest tests/integration/
```

The golden zip is downloaded once per test session and cached in a temporary directory.
Golden comparison is skipped automatically when `GOLDEN_URL = None` in `conftest.py`.

### Without golden comparison

Set `GOLDEN_URL = None` in `tests/conftest.py`. Tests will only verify that the CLI
exits without error (original behavior).
