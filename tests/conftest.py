import subprocess
from typing import Callable

import pytest


@pytest.fixture
def run_script() -> Callable:

    def _run_script(script_file, input_dir, output_dir, **kwargs):

        command_str = [
            'python',
            script_file,
            '--input-dir', input_dir,
            '--save-dir', output_dir,
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

    return _run_script
