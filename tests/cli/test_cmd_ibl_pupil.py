from unittest.mock import MagicMock, patch

import pytest

from eks.cli.cmd_ibl_pupil import cmd_ibl_pupil

_KEYPOINTS = ['pupil_top', 'pupil_bottom']
_SMOOTHER_RETURN = (MagicMock(), [0.9, 0.8], [MagicMock()], _KEYPOINTS)


_COMMAND_ARGS = {'diameter_s': None, 'com_s': None}


@pytest.fixture
def _args(make_cli_args):
    """Build the argparse.Namespace for the ibl-pupil command."""
    return make_cli_args(_COMMAND_ARGS)


class TestCmdIblPupil:
    """Test the cmd_ibl_pupil handler."""

    def test_calls_smoother(self, tmp_path, _args):
        """Smoother function is invoked when the command runs."""
        with patch('eks.cli.cmd_ibl_pupil.handle_io', return_value=tmp_path), \
                patch('eks.cli.cmd_ibl_pupil.fit_eks_pupil', return_value=_SMOOTHER_RETURN) as mock_fit:
            cmd_ibl_pupil(_args())
            mock_fit.assert_called_once()

    def test_smooth_params_passed_as_list(self, tmp_path, _args):
        """diameter_s and com_s are combined into a single smooth_params list."""
        with patch('eks.cli.cmd_ibl_pupil.handle_io', return_value=tmp_path), \
                patch('eks.cli.cmd_ibl_pupil.fit_eks_pupil', return_value=_SMOOTHER_RETURN) as mock_fit:
            cmd_ibl_pupil(_args(diameter_s=0.99, com_s=0.95))
            assert mock_fit.call_args.kwargs['smooth_params'] == [0.99, 0.95]

    def test_none_smooth_params_passed_as_none_list(self, tmp_path, _args):
        """Unspecified smooth params are passed as [None, None] for auto-tuning."""
        with patch('eks.cli.cmd_ibl_pupil.handle_io', return_value=tmp_path), \
                patch('eks.cli.cmd_ibl_pupil.fit_eks_pupil', return_value=_SMOOTHER_RETURN) as mock_fit:
            cmd_ibl_pupil(_args())
            assert mock_fit.call_args.kwargs['smooth_params'] == [None, None]

    def test_no_plot_by_default(self, tmp_path, _args):
        """plot_results is not called when --make-plot is not set."""
        with patch('eks.cli.cmd_ibl_pupil.handle_io', return_value=tmp_path), \
                patch('eks.cli.cmd_ibl_pupil.fit_eks_pupil', return_value=_SMOOTHER_RETURN), \
                patch('eks.cli.cmd_ibl_pupil.plot_results') as mock_plot:
            cmd_ibl_pupil(_args())
            mock_plot.assert_not_called()

    def test_make_plot(self, tmp_path, _args):
        """plot_results is called once when --make-plot is set."""
        with patch('eks.cli.cmd_ibl_pupil.handle_io', return_value=tmp_path), \
                patch('eks.cli.cmd_ibl_pupil.fit_eks_pupil', return_value=_SMOOTHER_RETURN), \
                patch('eks.cli.cmd_ibl_pupil.plot_results') as mock_plot:
            cmd_ibl_pupil(_args(make_plot=True))
            mock_plot.assert_called_once()
