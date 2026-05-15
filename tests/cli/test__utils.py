"""Tests for eks.cli._utils."""

import argparse
from pathlib import Path

import pytest

from eks.cli._utils import handle_io, parse_blocks, parse_s_frames


class TestHandleIo:
    """Test the handle_io function."""

    def test_handle_io_valid_dir_returns_save_dir(self, tmp_path):
        # Arrange
        input_dir = tmp_path / 'input'
        input_dir.mkdir()
        save_dir = tmp_path / 'output'

        # Act / Assert
        assert handle_io(input_dir, save_dir) == Path(save_dir)

    def test_handle_io_invalid_input_dir_raises(self, tmp_path):
        # Arrange
        input_dir = tmp_path / 'nonexistent'

        # Act / Assert
        with pytest.raises(ValueError, match='--input-dir must be a valid directory'):
            handle_io(input_dir, tmp_path)

    def test_handle_io_none_save_dir_creates_outputs(self, tmp_path, monkeypatch):
        # Arrange
        input_dir = tmp_path / 'input'
        input_dir.mkdir()
        monkeypatch.chdir(tmp_path)

        # Act
        result = handle_io(input_dir, None)

        # Assert
        assert result == tmp_path / 'outputs'
        assert result.is_dir()


class TestParseSFrames:
    """Test the parse_s_frames function."""

    def test_parse_s_frames_digit_string(self):
        # bare integer N → [(1, N)]
        assert parse_s_frames('100') == [(1, 100)]

    def test_parse_s_frames_single_tuple(self):
        assert parse_s_frames('[(10,200)]') == [(10, 200)]

    def test_parse_s_frames_multiple_tuples(self):
        assert parse_s_frames('[(0,100),(200,300)]') == [(0, 100), (200, 300)]

    def test_parse_s_frames_open_ends(self):
        assert parse_s_frames('[(,100),(200,)]') == [(None, 100), (200, None)]

    def test_parse_s_frames_no_valid_tuples_raises(self):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_s_frames('not_valid')

    def test_parse_s_frames_start_greater_than_end_raises(self):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_s_frames('[(200,100)]')


class TestParseBlocks:
    """Test the parse_blocks function."""

    def test_parse_blocks_single_block(self):
        assert parse_blocks('0,1,2') == [[0, 1, 2]]

    def test_parse_blocks_multiple_blocks(self):
        assert parse_blocks('0,1;2,3') == [[0, 1], [2, 3]]

    def test_parse_blocks_invalid_raises(self):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_blocks('a,b')
