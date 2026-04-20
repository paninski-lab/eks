import argparse
from unittest.mock import MagicMock, patch

from eks.cli.cmd_ibl_pupil import cmd_ibl_pupil

_KEYPOINTS = ['pupil_top', 'pupil_bottom']
_SMOOTHER_RETURN = (MagicMock(), [0.9, 0.8], [MagicMock()], _KEYPOINTS)


def _args(
    input_dir='/tmp/input',
    input_files=None,
    save_dir='/tmp/output',
    save_filename=None,
    s_frames=None,
    blocks=[],
    verbose=False,
    make_plot=False,
    diameter_s=None,
    com_s=None,
):
    return argparse.Namespace(
        input_dir=input_dir,
        input_files=input_files,
        save_dir=save_dir,
        save_filename=save_filename,
        s_frames=s_frames,
        blocks=blocks,
        verbose=verbose,
        make_plot=make_plot,
        diameter_s=diameter_s,
        com_s=com_s,
    )


class TestCmdIblPupil:
    """Test the cmd_ibl_pupil handler."""

    def test_calls_smoother(self, tmp_path):
        """Smoother function is invoked when the command runs."""
        with patch('eks.cli.cmd_ibl_pupil.handle_io', return_value=tmp_path), \
                patch('eks.cli.cmd_ibl_pupil.fit_eks_pupil', return_value=_SMOOTHER_RETURN) as mock_fit:
            cmd_ibl_pupil(_args())
            mock_fit.assert_called_once()

    def test_smooth_params_passed_as_list(self, tmp_path):
        """diameter_s and com_s are combined into a single smooth_params list."""
        with patch('eks.cli.cmd_ibl_pupil.handle_io', return_value=tmp_path), \
                patch('eks.cli.cmd_ibl_pupil.fit_eks_pupil', return_value=_SMOOTHER_RETURN) as mock_fit:
            cmd_ibl_pupil(_args(diameter_s=0.99, com_s=0.95))
            assert mock_fit.call_args.kwargs['smooth_params'] == [0.99, 0.95]

    def test_none_smooth_params_passed_as_none_list(self, tmp_path):
        """Unspecified smooth params are passed as [None, None] for auto-tuning."""
        with patch('eks.cli.cmd_ibl_pupil.handle_io', return_value=tmp_path), \
                patch('eks.cli.cmd_ibl_pupil.fit_eks_pupil', return_value=_SMOOTHER_RETURN) as mock_fit:
            cmd_ibl_pupil(_args())
            assert mock_fit.call_args.kwargs['smooth_params'] == [None, None]

    def test_no_plot_by_default(self, tmp_path):
        """plot_results is not called when --make-plot is not set."""
        with patch('eks.cli.cmd_ibl_pupil.handle_io', return_value=tmp_path), \
                patch('eks.cli.cmd_ibl_pupil.fit_eks_pupil', return_value=_SMOOTHER_RETURN), \
                patch('eks.cli.cmd_ibl_pupil.plot_results') as mock_plot:
            cmd_ibl_pupil(_args())
            mock_plot.assert_not_called()

    def test_make_plot(self, tmp_path):
        """plot_results is called once when --make-plot is set."""
        with patch('eks.cli.cmd_ibl_pupil.handle_io', return_value=tmp_path), \
                patch('eks.cli.cmd_ibl_pupil.fit_eks_pupil', return_value=_SMOOTHER_RETURN), \
                patch('eks.cli.cmd_ibl_pupil.plot_results') as mock_plot:
            cmd_ibl_pupil(_args(make_plot=True))
            mock_plot.assert_called_once()
