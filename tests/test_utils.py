import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eks.utils import crop_frames, format_data, plot_results


def _make_dlc_csv(directory: Path, filename: str, keypoints: list, n_frames: int = 5) -> Path:
    """Write a minimal DLC-format CSV (3-row MultiIndex header) to directory/filename."""
    scorer = 'scorer'
    arrays = [
        [scorer] * (len(keypoints) * 3),
        [kp for kp in keypoints for _ in range(3)],
        ['x', 'y', 'likelihood'] * len(keypoints),
    ]
    columns = pd.MultiIndex.from_arrays(arrays, names=['scorer', 'bodyparts', 'coords'])
    rng = np.random.default_rng(0)
    data = rng.random((n_frames, len(keypoints) * 3))
    df = pd.DataFrame(data, columns=columns)
    filepath = directory / filename
    df.to_csv(filepath)
    return filepath


class TestCropFrames:
    """Test the crop_frames function."""

    def test_crop_frames_no_crop_none(self):
        # Arrange
        y = np.arange(20)

        # Act
        out = crop_frames(y, None)

        # Assert
        assert np.shares_memory(out, y) or np.array_equal(out, y)
        assert out.shape == y.shape

    def test_crop_frames_no_crop_none_none(self):
        # Arrange
        y = np.arange(20)

        # Act
        out = crop_frames(y, [(None, None)])

        # Assert
        assert np.shares_memory(out, y) or np.array_equal(out, y)
        assert out.shape == y.shape

    def test_crop_frames_single_span(self):
        # Arrange
        y = np.arange(10)

        # Act
        out = crop_frames(y, [(2, 5)])

        # Assert
        np.testing.assert_array_equal(out, np.array([2, 3, 4]))

    def test_crop_frames_open_ended_spans(self):
        # Arrange
        y = np.arange(10)

        # Act
        out = crop_frames(y, [(None, 3), (7, None)])

        # Assert
        np.testing.assert_array_equal(out, np.array([0, 1, 2, 7, 8, 9]))

    def test_crop_frames_invalid_tuple_shape(self):
        # Arrange
        y = np.arange(10)

        # Act / Assert
        with pytest.raises(ValueError):
            crop_frames(y, [(1, 3, 5)])

    def test_crop_frames_out_of_bounds(self):
        # Arrange
        y = np.arange(10)

        # Act / Assert
        with pytest.raises(ValueError):
            crop_frames(y, [(1, 20)])

        with pytest.raises(ValueError):
            crop_frames(y, [(6, 5)])

    def test_crop_frames_overlap_raises(self):
        # Arrange
        y = np.arange(20)

        # Act / Assert
        with pytest.raises(ValueError):
            crop_frames(y, [(2, 6), (5, 10)])


class TestFormatData:
    """Test the format_data function."""

    def test_format_data_directory_input(self, tmp_path):
        # Arrange
        _make_dlc_csv(tmp_path, 'model0.csv', ['nose', 'tail'])
        _make_dlc_csv(tmp_path, 'model1.csv', ['nose', 'tail'])

        # Act
        input_dfs_list, keypoint_names = format_data(str(tmp_path))

        # Assert
        assert len(input_dfs_list) == 2
        assert keypoint_names == ['nose', 'tail']

    def test_format_data_list_input(self, tmp_path):
        # Arrange
        path0 = _make_dlc_csv(tmp_path, 'model0.csv', ['nose', 'tail'])
        path1 = _make_dlc_csv(tmp_path, 'model1.csv', ['nose', 'tail'])

        # Act
        input_dfs_list, keypoint_names = format_data([str(path0), str(path1)])

        # Assert
        assert len(input_dfs_list) == 2
        assert keypoint_names == ['nose', 'tail']

    def test_format_data_invalid_source_raises(self):
        # Act / Assert
        with pytest.raises(ValueError, match='input_source must be'):
            format_data('/nonexistent/path/that/does/not/exist')

    def test_format_data_no_valid_files_raises(self, tmp_path):
        # Arrange
        (tmp_path / 'readme.txt').write_text('not a csv')

        # Act / Assert
        with pytest.raises(FileNotFoundError, match='no valid marker input files'):
            format_data(str(tmp_path))

    def test_format_data_skips_non_csv_files(self, tmp_path):
        # Arrange
        _make_dlc_csv(tmp_path, 'model0.csv', ['nose'])
        (tmp_path / 'ignore.txt').write_text('not a csv')

        # Act
        input_dfs_list, _ = format_data(str(tmp_path))

        # Assert
        assert len(input_dfs_list) == 1

    def test_format_data_output_columns(self, tmp_path):
        # Arrange
        _make_dlc_csv(tmp_path, 'model0.csv', ['nose', 'tail'])

        # Act
        input_dfs_list, keypoint_names = format_data(str(tmp_path))

        # Assert - flat column names produced by convert_lp_dlc
        df = input_dfs_list[0]
        expected_cols = ['nose_x', 'nose_y', 'nose_likelihood', 'tail_x', 'tail_y', 'tail_likelihood']
        assert list(df.columns) == expected_cols

    def test_format_data_camera_names_returns_list_of_lists(self, tmp_path):
        # Arrange
        _make_dlc_csv(tmp_path, 'model0_top.csv', ['nose', 'tail'])
        _make_dlc_csv(tmp_path, 'model0_bot.csv', ['nose', 'tail'])

        # Act
        input_dfs_list, keypoint_names = format_data(
            str(tmp_path), camera_names=['top', 'bot'],
        )

        # Assert - one list per camera, one DataFrame per model
        assert len(input_dfs_list) == 2
        assert len(input_dfs_list[0]) == 1
        assert len(input_dfs_list[1]) == 1
        assert keypoint_names == ['nose', 'tail']

    def test_format_data_camera_names_multiple_models(self, tmp_path):
        # Arrange - two models per camera
        _make_dlc_csv(tmp_path, 'model0_top.csv', ['nose'])
        _make_dlc_csv(tmp_path, 'model1_top.csv', ['nose'])
        _make_dlc_csv(tmp_path, 'model0_bot.csv', ['nose'])
        _make_dlc_csv(tmp_path, 'model1_bot.csv', ['nose'])

        # Act
        input_dfs_list, _ = format_data(str(tmp_path), camera_names=['top', 'bot'])

        # Assert
        assert len(input_dfs_list[0]) == 2
        assert len(input_dfs_list[1]) == 2

    def test_format_data_camera_not_found_raises(self, tmp_path):
        # Arrange - only 'top' files exist
        _make_dlc_csv(tmp_path, 'model0_top.csv', ['nose'])

        # Act / Assert
        with pytest.raises(FileNotFoundError, match="no files matching camera 'bot'"):
            format_data(str(tmp_path), camera_names=['top', 'bot'])

    def test_format_data_unequal_seed_counts_warns(self, tmp_path, caplog):
        # Arrange - 'top' has 2 models, 'bot' has 1
        _make_dlc_csv(tmp_path, 'model0_top.csv', ['nose'])
        _make_dlc_csv(tmp_path, 'model1_top.csv', ['nose'])
        _make_dlc_csv(tmp_path, 'model0_bot.csv', ['nose'])

        # Act
        with caplog.at_level(logging.WARNING, logger='eks.utils'):
            format_data(str(tmp_path), camera_names=['top', 'bot'])

        # Assert
        assert 'unequal number of seed files per camera' in caplog.text


class TestPlotResults:
    """Test the plot_results function."""

    def _make_output_df(self, key: str, n_frames: int = 20) -> pd.DataFrame:
        scorer = 'ensemble-kalman_tracker'
        columns = pd.MultiIndex.from_arrays([
            [scorer] * 3,
            [key] * 3,
            ['x', 'y', 'likelihood'],
        ])
        rng = np.random.default_rng(0)
        return pd.DataFrame(rng.random((n_frames, 3)), columns=columns)

    def _make_input_df(self, key: str, n_frames: int = 20) -> pd.DataFrame:
        rng = np.random.default_rng(1)
        return pd.DataFrame({
            f'{key}_x': rng.random(n_frames),
            f'{key}_y': rng.random(n_frames),
            f'{key}_likelihood': rng.random(n_frames),
        })

    def test_plot_results_creates_pdf(self, tmp_path):
        # Arrange
        key = 'nose'
        output_df = self._make_output_df(key)
        input_dfs_list = [self._make_input_df(key)]

        # Act
        plot_results(
            output_df=output_df,
            input_dfs_list=input_dfs_list,
            key=key,
            s_final=1.5,
            nll_values=None,
            idxs=(0, 10),
            save_dir=str(tmp_path),
            smoother_type='singlecam',
        )

        # Assert
        assert (tmp_path / f'singlecam_{key}.pdf').exists()

    def test_plot_results_tuple_s_final_and_nll_values(self, tmp_path):
        # Arrange — covers the s_final tuple branch and the nll_values plotting branch
        key = 'nose'
        n_frames = 20
        output_df = self._make_output_df(key, n_frames)
        input_dfs_list = [self._make_input_df(key, n_frames)]
        nll_values = np.random.default_rng(2).random(n_frames)

        # Act
        plot_results(
            output_df=output_df,
            input_dfs_list=input_dfs_list,
            key=key,
            s_final=(0.5, 1.0),
            nll_values=nll_values,
            idxs=(0, 10),
            save_dir=str(tmp_path),
            smoother_type='ibl_pupil',
        )

        # Assert
        assert (tmp_path / f'ibl_pupil_{key}.pdf').exists()
