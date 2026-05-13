import argparse
from unittest.mock import MagicMock, patch

from eks.cli.cmd_singlecam import cmd_singlecam

_BODYPARTS = ['nose', 'tail']
_SMOOTHER_RETURN = (MagicMock(), [0.5, 0.5], [MagicMock(), MagicMock()], _BODYPARTS)


def _args(
    input_dir='/tmp/input',
    input_files=None,
    save_dir='/tmp/output',
    save_filename=None,
    s_frames=None,
    blocks=[],
    verbose=False,
    make_plot=False,
    bodypart_list=None,
    s=None,
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
        bodypart_list=bodypart_list,
        s=s,
    )


class TestCmdSinglecam:
    """Test the cmd_singlecam handler."""

    def test_calls_smoother(self, tmp_path):
        """Smoother function is invoked when the command runs."""
        with patch('eks.cli.cmd_singlecam.handle_io', return_value=tmp_path), \
                patch('eks.cli.cmd_singlecam.fit_eks_singlecam', return_value=_SMOOTHER_RETURN) as mock_fit:
            cmd_singlecam(_args())
            mock_fit.assert_called_once()

    def test_passes_args_to_smoother(self, tmp_path):
        """CLI arguments are forwarded to the smoother correctly."""
        with patch('eks.cli.cmd_singlecam.handle_io', return_value=tmp_path), \
                patch('eks.cli.cmd_singlecam.fit_eks_singlecam', return_value=_SMOOTHER_RETURN) as mock_fit:
            cmd_singlecam(_args(bodypart_list=['nose'], s=[5.0], verbose=True, blocks=[[0, 1]]))
            kwargs = mock_fit.call_args.kwargs
            assert kwargs['bodypart_list'] == ['nose']
            assert kwargs['smooth_param'] == [5.0]
            assert kwargs['blocks'] == [[0, 1]]

    def test_uses_input_files_when_no_input_dir(self, tmp_path):
        """input_source falls back to input_files when input_dir is None."""
        input_files = [str(tmp_path / 'a.csv'), str(tmp_path / 'b.csv')]
        with patch('eks.cli.cmd_singlecam.handle_io', return_value=tmp_path), \
                patch('eks.cli.cmd_singlecam.fit_eks_singlecam', return_value=_SMOOTHER_RETURN) as mock_fit:
            cmd_singlecam(_args(input_dir=None, input_files=input_files))
            assert mock_fit.call_args.kwargs['input_source'] == input_files

    def test_no_plot_by_default(self, tmp_path):
        """plot_results is not called when --make-plot is not set."""
        with patch('eks.cli.cmd_singlecam.handle_io', return_value=tmp_path), \
                patch('eks.cli.cmd_singlecam.fit_eks_singlecam', return_value=_SMOOTHER_RETURN), \
                patch('eks.cli.cmd_singlecam.plot_results') as mock_plot:
            cmd_singlecam(_args())
            mock_plot.assert_not_called()

    def test_make_plot(self, tmp_path):
        """plot_results is called once when --make-plot is set."""
        with patch('eks.cli.cmd_singlecam.handle_io', return_value=tmp_path), \
                patch('eks.cli.cmd_singlecam.fit_eks_singlecam', return_value=_SMOOTHER_RETURN), \
                patch('eks.cli.cmd_singlecam.plot_results') as mock_plot:
            cmd_singlecam(_args(make_plot=True))
            mock_plot.assert_called_once()
