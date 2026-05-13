from unittest.mock import patch

import pytest

from eks.cli.main import main


class TestMain:
    """Test the main CLI entry point."""

    def test_no_args_exits_nonzero(self):
        """Invoking eks with no subcommand exits with a non-zero code."""
        with patch('sys.argv', ['eks']):
            with pytest.raises(SystemExit) as exc_info:
                main()
        assert exc_info.value.code != 0

    @pytest.mark.parametrize('subcommand', [
        'singlecam',
        'multicam',
        'ibl-pupil',
        'ibl-paw',
        'mirrored-multicam',
    ])
    def test_subcommand_help_exits_zero(self, subcommand):
        """Each expected subcommand is registered and responds to --help."""
        with patch('sys.argv', ['eks', subcommand, '--help']):
            with pytest.raises(SystemExit) as exc_info:
                main()
        assert exc_info.value.code == 0
