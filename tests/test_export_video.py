import json
from unittest.mock import Mock, patch

import pytest
from replay.export_video import main, parse_args


@pytest.fixture
def mock_env():
    env = Mock()
    env.reset = Mock(return_value=(Mock(), {}))
    env.step = Mock()
    env.close = Mock()
    env._env = Mock()
    env._env.render = Mock(return_value=Mock())
    return env


@pytest.fixture
def sample_run_data():
    return {
        "spec": {
            "env_id": "ALE/Pacman-v5",
            "seed": 42,
            "frameskip": 4,
            "repeat_action_probability": 0.0,
        },
        "actions": [0, 1, 2, 3, 4],
        "total_reward": 10.0,
        "final_obs_hash": "abc123",
    }


class TestParseArgs:
    def test_parse_args_required_run(self):
        """Test parsing with required --run argument."""
        with patch("sys.argv", ["export_video.py", "--run", "runs/test/run.json"]):
            args = parse_args()
            assert args.run == "runs/test/run.json"
            assert args.fps == 60
            assert args.out is None
            assert args.max_frames is None

    def test_parse_args_all_options(self):
        """Test parsing with all options."""
        with patch(
            "sys.argv",
            [
                "export_video.py",
                "--run",
                "runs/test/run.json",
                "--out",
                "output.mp4",
                "--fps",
                "30",
                "--max-frames",
                "1000",
            ],
        ):
            args = parse_args()
            assert args.run == "runs/test/run.json"
            assert args.out == "output.mp4"
            assert args.fps == 30
            assert args.max_frames == 1000

    def test_parse_args_missing_run(self):
        """Test that missing --run argument raises SystemExit."""
        with patch("sys.argv", ["export_video.py"]):
            with pytest.raises(SystemExit):
                parse_args()


class TestMain:
    @patch("replay.export_video.iio.imwrite")
    @patch("replay.export_video.ALEEnv")
    def test_main_default_output_path(
        self, mock_ale_env, mock_imwrite, tmp_path, sample_run_data, mock_env
    ):
        """Test video export with default output path."""
        mock_ale_env.return_value = mock_env

        run_file = tmp_path / "run.json"
        run_file.write_text(json.dumps(sample_run_data))

        step_result = Mock(terminated=False, truncated=False)
        mock_env.step.return_value = step_result

        with patch("sys.argv", ["export_video.py", "--run", str(run_file)]):
            main()

        mock_env.reset.assert_called_once_with(seed=42)
        assert mock_env.step.call_count == 5  # 5 actions
        mock_imwrite.assert_called_once()

        # Check that default output path is in same directory as run.json
        call_args = mock_imwrite.call_args
        output_path = call_args[0][0]
        assert output_path == tmp_path / "replay.mp4"

    @patch("replay.export_video.iio.imwrite")
    @patch("replay.export_video.ALEEnv")
    def test_main_custom_output_path(
        self, mock_ale_env, mock_imwrite, tmp_path, sample_run_data, mock_env
    ):
        """Test video export with custom output path."""
        mock_ale_env.return_value = mock_env

        run_file = tmp_path / "run.json"
        run_file.write_text(json.dumps(sample_run_data))
        custom_out = tmp_path / "custom_video.mp4"

        step_result = Mock(terminated=False, truncated=False)
        mock_env.step.return_value = step_result

        with patch(
            "sys.argv", ["export_video.py", "--run", str(run_file), "--out", str(custom_out)]
        ):
            main()

        call_args = mock_imwrite.call_args
        output_path = call_args[0][0]
        assert output_path == custom_out

    @patch("replay.export_video.iio.imwrite")
    @patch("replay.export_video.ALEEnv")
    def test_main_custom_fps(self, mock_ale_env, mock_imwrite, tmp_path, sample_run_data, mock_env):
        """Test video export with custom FPS."""
        mock_ale_env.return_value = mock_env

        run_file = tmp_path / "run.json"
        run_file.write_text(json.dumps(sample_run_data))

        step_result = Mock(terminated=False, truncated=False)
        mock_env.step.return_value = step_result

        with patch("sys.argv", ["export_video.py", "--run", str(run_file), "--fps", "30"]):
            main()

        call_args = mock_imwrite.call_args
        assert call_args[1]["fps"] == 30

    @patch("replay.export_video.iio.imwrite")
    @patch("replay.export_video.ALEEnv")
    def test_main_max_frames_limit(
        self, mock_ale_env, mock_imwrite, tmp_path, sample_run_data, mock_env
    ):
        """Test video export stops at max_frames limit."""
        mock_ale_env.return_value = mock_env

        run_file = tmp_path / "run.json"
        run_file.write_text(json.dumps(sample_run_data))

        step_result = Mock(terminated=False, truncated=False)
        mock_env.step.return_value = step_result

        with patch("sys.argv", ["export_video.py", "--run", str(run_file), "--max-frames", "3"]):
            main()

        # Should stop early due to max_frames limit
        # Initial frame (1) + 2 steps = 3 frames total
        assert mock_env.step.call_count == 2

    @patch("replay.export_video.iio.imwrite")
    @patch("replay.export_video.ALEEnv")
    def test_main_handles_done_and_reset(
        self, mock_ale_env, mock_imwrite, tmp_path, sample_run_data, mock_env
    ):
        """Test video export handles episode termination and reset."""
        mock_ale_env.return_value = mock_env

        run_file = tmp_path / "run.json"
        run_file.write_text(json.dumps(sample_run_data))

        # First action terminates episode
        step_result_done = Mock(terminated=True, truncated=False)
        step_result_normal = Mock(terminated=False, truncated=False)
        mock_env.step.side_effect = [step_result_done] + [step_result_normal] * 4

        with patch("sys.argv", ["export_video.py", "--run", str(run_file)]):
            main()

        # Should call reset twice: initial + after done
        assert mock_env.reset.call_count == 2
        reset_calls = mock_env.reset.call_args_list
        assert reset_calls[0].kwargs["seed"] == 42
        assert reset_calls[1].kwargs["seed"] == 43  # seed + step + 1

    @patch("replay.export_video.iio.imwrite")
    @patch("replay.export_video.ALEEnv")
    def test_main_handles_none_frames(
        self, mock_ale_env, mock_imwrite, tmp_path, sample_run_data, mock_env
    ):
        """Test video export handles None frames gracefully."""
        mock_ale_env.return_value = mock_env
        mock_env.render_rgb.return_value = None  # No frame available

        run_file = tmp_path / "run.json"
        run_file.write_text(json.dumps(sample_run_data))

        step_result = Mock(terminated=False, truncated=False)
        mock_env.step.return_value = step_result

        with patch("sys.argv", ["export_video.py", "--run", str(run_file)]):
            with pytest.raises(RuntimeError, match="No frames captured"):
                main()

        # Should not call imwrite if no frames were captured
        mock_imwrite.assert_not_called()

    @patch("replay.export_video.iio.imwrite")
    @patch("replay.export_video.ALEEnv")
    def test_main_env_closed(self, mock_ale_env, mock_imwrite, tmp_path, sample_run_data, mock_env):
        """Test that environment is properly closed."""
        mock_ale_env.return_value = mock_env

        run_file = tmp_path / "run.json"
        run_file.write_text(json.dumps(sample_run_data))

        step_result = Mock(terminated=False, truncated=False)
        mock_env.step.return_value = step_result

        with patch("sys.argv", ["export_video.py", "--run", str(run_file)]):
            main()

        mock_env.close.assert_called_once()
