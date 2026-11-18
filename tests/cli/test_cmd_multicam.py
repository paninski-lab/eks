from unittest.mock import MagicMock, patch

import pytest

from eks.cli.cmd_multicam import cmd_multicam

_BODYPARTS = ['nose', 'tail']
_CAMERA_DFS = [MagicMock(), MagicMock()]
_INPUT_DFS = [[MagicMock()], [MagicMock()]]
_SMOOTHER_RETURN = (_CAMERA_DFS, [0.5, 0.5], _INPUT_DFS, _BODYPARTS, MagicMock())


_COMMAND_ARGS = {
    'bodypart_list': None,
    's': None,
    'camera_names': ['top', 'bot'],
    'quantile_keep_pca': 95,
    'inflate_vars': True,
    'n_latent': 3,
    'calibration': None,
}


@pytest.fixture
def _args(make_cli_args):
    """Build the argparse.Namespace for the multicam command."""
    return make_cli_args(_COMMAND_ARGS)


class TestCmdMulticam:
    """Test the cmd_multicam handler."""

    def test_calls_smoother(self, tmp_path, _args):
        """Smoother function is invoked when the command runs."""
        with patch('eks.cli.cmd_multicam.handle_io', return_value=tmp_path), \
                patch('eks.cli.cmd_multicam.fit_eks_multicam', return_value=_SMOOTHER_RETURN) as mock_fit:
            cmd_multicam(_args())
            mock_fit.assert_called_once()

    def test_passes_args_to_smoother(self, tmp_path, _args):
        """CLI arguments are forwarded to the smoother correctly."""
        with patch('eks.cli.cmd_multicam.handle_io', return_value=tmp_path), \
                patch('eks.cli.cmd_multicam.fit_eks_multicam', return_value=_SMOOTHER_RETURN) as mock_fit:
            cmd_multicam(_args(
                bodypart_list=['nose'],
                camera_names=['left', 'right'],
                s=[5.0],
                n_latent=5,
                calibration='/tmp/cal.toml',
            ))
            kwargs = mock_fit.call_args.kwargs
            assert kwargs['bodypart_list'] == ['nose']
            assert kwargs['camera_names'] == ['left', 'right']
            assert kwargs['smooth_param'] == [5.0]
            assert kwargs['n_latent'] == 5
            assert kwargs['calibration'] == '/tmp/cal.toml'

    def test_inflate_vars_default_true(self, tmp_path, _args):
        """inflate_vars defaults to True (enabled by default)."""
        with patch('eks.cli.cmd_multicam.handle_io', return_value=tmp_path), \
                patch('eks.cli.cmd_multicam.fit_eks_multicam', return_value=_SMOOTHER_RETURN) as mock_fit:
            cmd_multicam(_args())
            assert mock_fit.call_args.kwargs['inflate_vars'] is True

    def test_no_inflate_vars(self, tmp_path, _args):
        """inflate_vars is False when --no-inflate-vars is set."""
        with patch('eks.cli.cmd_multicam.handle_io', return_value=tmp_path), \
                patch('eks.cli.cmd_multicam.fit_eks_multicam', return_value=_SMOOTHER_RETURN) as mock_fit:
            cmd_multicam(_args(inflate_vars=False))
            assert mock_fit.call_args.kwargs['inflate_vars'] is False

    def test_no_plot_by_default(self, tmp_path, _args):
        """plot_results is not called when --make-plot is not set."""
        with patch('eks.cli.cmd_multicam.handle_io', return_value=tmp_path), \
                patch('eks.cli.cmd_multicam.fit_eks_multicam', return_value=_SMOOTHER_RETURN), \
                patch('eks.cli.cmd_multicam.plot_results') as mock_plot:
            cmd_multicam(_args())
            mock_plot.assert_not_called()

    def test_make_plot(self, tmp_path, _args):
        """plot_results is called once when --make-plot is set."""
        with patch('eks.cli.cmd_multicam.handle_io', return_value=tmp_path), \
                patch('eks.cli.cmd_multicam.fit_eks_multicam', return_value=_SMOOTHER_RETURN), \
                patch('eks.cli.cmd_multicam.plot_results') as mock_plot:
            cmd_multicam(_args(make_plot=True))
            mock_plot.assert_called_once()
