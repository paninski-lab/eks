import io
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pytest

# URL of the zipped golden files. Update this after uploading a new release to GitHub.
GOLDEN_URL = 'https://github.com/paninski-lab/eks-test-fixtures/releases/download/v1/eks_golden.zip'  # noqa: E501


def pytest_addoption(parser):
    parser.addoption(
        '--generate-golden',
        action='store_true',
        default=False,
        help='Generate golden output files instead of comparing against them.',
    )
    parser.addoption(
        '--golden-dir',
        action='store',
        default=None,
        help='Directory to write golden files to (used with --generate-golden).',
    )


@pytest.fixture(scope='session')
def golden_dir(tmp_path_factory, pytestconfig):
    """Return path to golden files directory, downloading and extracting if necessary."""
    if pytestconfig.getoption('--generate-golden'):
        golden_dir_opt = pytestconfig.getoption('--golden-dir')
        if golden_dir_opt is None:
            raise ValueError('--golden-dir must be specified when using --generate-golden')
        path = Path(golden_dir_opt)
        path.mkdir(parents=True, exist_ok=True)
        return path

    if GOLDEN_URL is None:
        return None

    cache_dir = tmp_path_factory.mktemp('golden')
    with urllib.request.urlopen(GOLDEN_URL) as response:
        data = response.read()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        zf.extractall(cache_dir)
    return cache_dir


@pytest.fixture
def run_script() -> Callable:

    def _run_script(script_file, input_dir, output_dir, **kwargs) -> Path:

        command_str = [
            'python',
            script_file,
            '--input-dir', input_dir,
            '--save-dir', str(output_dir),
            '--verbose', 'True',
        ]
        for key, arg in kwargs.items():
            command_str.append(f'--{key.replace("_", "-")}')
            if isinstance(arg, list):  # split list arguments into separate elements
                command_str.extend(map(str, arg))
            else:
                command_str.append(str(arg))

        process = subprocess.run(command_str)
        assert process.returncode == 0
        return Path(str(output_dir))

    return _run_script


@pytest.fixture
def compare_to_golden(golden_dir, pytestconfig):
    """Fixture that either saves CSV outputs as golden files, or compares against them.

    In generate mode (--generate-golden), copies all CSVs from output_dir into
    golden_dir/<test_name>/. In compare mode, downloads golden files from the URL
    and asserts numerical equality against them.
    """

    def _compare(test_name: str, output_dir: Path):
        csv_files = sorted(output_dir.glob('*.csv'))
        assert len(csv_files) > 0, f'No CSV files found in {output_dir}'

        if pytestconfig.getoption('--generate-golden'):
            dest = golden_dir / test_name
            dest.mkdir(parents=True, exist_ok=True)
            for csv_file in csv_files:
                shutil.copy(csv_file, dest / csv_file.name)
            return

        if golden_dir is None:
            pytest.skip('GOLDEN_URL is None in conftest.py; skipping golden comparison.')

        golden_test_dir = golden_dir / test_name
        assert golden_test_dir.exists(), (
            f'Golden directory not found for test "{test_name}": {golden_test_dir}'
        )

        for csv_file in csv_files:
            golden_csv = golden_test_dir / csv_file.name
            assert golden_csv.exists(), (
                f'Golden file not found: {golden_csv}. '
                f'Run with --generate-golden to regenerate.'
            )
            actual = pd.read_csv(csv_file, index_col=0)
            expected = pd.read_csv(golden_csv, index_col=0)
            assert actual.shape == expected.shape, (
                f'{test_name}/{csv_file.name}: shape mismatch '
                f'{actual.shape} != {expected.shape}'
            )
            assert list(actual.columns) == list(expected.columns), (
                f'{test_name}/{csv_file.name}: column mismatch'
            )
            np.testing.assert_allclose(
                actual.select_dtypes('number').values,
                expected.select_dtypes('number').values,
                rtol=0,
                atol=1e-4,
                err_msg=f'{test_name}/{csv_file.name}',
            )

    return _compare
