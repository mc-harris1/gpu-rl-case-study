import json
from unittest.mock import Mock, patch

import pytest
from replay.replay import main, parse_args


@pytest.fixture
def mock_env():
    env = Mock()
    env.reset = Mock(return_value=(Mock(), {}))
    env.step = Mock()
    env.close = Mock()
    return env


@pytest.fixture
def sample_run_data():
    return {
        "spec": {
            "env_key": "pacman",
            "env_id": "ALE/Pong-v5",
            "obs_type": "pixels",
            "seed": 42,
            "policy": "sticky_dir",
            "frameskip": 4,
            "repeat_action_probability": 0.0,
            "single_episode": False,
        },
        "actions": [0, 1, 2],
        "total_reward": 5.0,
        "final_obs_hash": "abc123",
    }


def test_parse_args_with_run_argument(monkeypatch):
    monkeypatch.setattr("sys.argv", ["replay.py", "--run", "runs/1/run.json"])
    args = parse_args()
    assert args.run == "runs/1/run.json"
    assert args.render is False


def test_parse_args_with_render_flag(monkeypatch):
    monkeypatch.setattr("sys.argv", ["replay.py", "--run", "runs/1/run.json", "--render"])
    args = parse_args()
    assert args.render is True


def test_parse_args_missing_run_argument(monkeypatch):
    monkeypatch.setattr("sys.argv", ["replay.py"])
    with pytest.raises(SystemExit):
        parse_args()


@patch("replay.replay.ALEEnv")
@patch("replay.replay.obs_hash")
def test_main_successful_replay(mock_obs_hash, mock_ale_env, tmp_path, sample_run_data, mock_env):
    mock_ale_env.return_value = mock_env
    mock_obs_hash.side_effect = ["hash0", "hash1", "hash2", "abc123"]

    run_file = tmp_path / "run.json"
    run_file.write_text(json.dumps(sample_run_data))

    # Create step results with rewards that sum to 5.0 (3 actions: 1.0 + 2.0 + 2.0 = 5.0)
    step_results = [
        Mock(reward=1.0, obs=Mock(), terminated=False, truncated=False),
        Mock(reward=2.0, obs=Mock(), terminated=False, truncated=False),
        Mock(reward=2.0, obs=Mock(), terminated=False, truncated=False),
    ]
    mock_env.step.side_effect = step_results

    with patch("sys.argv", ["replay.py", "--run", str(run_file)]):
        main()

    mock_env.reset.assert_called()
    assert mock_env.step.call_count == 3
    mock_env.close.assert_called_once()


@patch("replay.replay.ALEEnv")
@patch("replay.replay.obs_hash")
def test_main_reward_mismatch(mock_obs_hash, mock_ale_env, tmp_path, sample_run_data, mock_env):
    mock_ale_env.return_value = mock_env
    mock_obs_hash.side_effect = ["hash0", "hash1", "hash2", "hash3"]

    sample_run_data["total_reward"] = 10.0
    run_file = tmp_path / "run.json"
    run_file.write_text(json.dumps(sample_run_data))

    step_result = Mock(reward=5.0, obs=Mock(), terminated=False, truncated=False)
    mock_env.step.return_value = step_result

    with patch("sys.argv", ["replay.py", "--run", str(run_file)]):
        with pytest.raises(SystemExit, match="2"):
            main()


@patch("replay.replay.ALEEnv")
@patch("replay.replay.obs_hash")
def test_main_hash_mismatch(mock_obs_hash, mock_ale_env, tmp_path, sample_run_data, mock_env):
    mock_ale_env.return_value = mock_env
    mock_obs_hash.side_effect = ["hash0", "hash1", "hash2", "wrong_hash"]

    run_file = tmp_path / "run.json"
    run_file.write_text(json.dumps(sample_run_data))

    step_result = Mock(reward=5.0, obs=Mock(), terminated=False, truncated=False)
    mock_env.step.return_value = step_result

    with patch("sys.argv", ["replay.py", "--run", str(run_file)]):
        with pytest.raises(SystemExit, match="2"):
            main()


@patch("replay.replay.ALEEnv")
@patch("replay.replay.obs_hash")
def test_main_reset_on_done(mock_obs_hash, mock_ale_env, tmp_path, sample_run_data, mock_env):
    mock_ale_env.return_value = mock_env
    # Need more hashes: initial obs + 3 steps (one with done, which adds extra hash after reset)
    mock_obs_hash.side_effect = ["hash0", "hash1", "hash_after_reset", "hash2", "abc123"]

    run_file = tmp_path / "run.json"
    run_file.write_text(json.dumps(sample_run_data))

    # 3 actions total, rewards sum to 5.0
    step_result_done = Mock(reward=2.5, obs=Mock(), terminated=True, truncated=False)
    step_result_normal_1 = Mock(reward=2.0, obs=Mock(), terminated=False, truncated=False)
    step_result_normal_2 = Mock(reward=0.5, obs=Mock(), terminated=False, truncated=False)
    mock_env.step.side_effect = [step_result_done, step_result_normal_1, step_result_normal_2]

    with patch("sys.argv", ["replay.py", "--run", str(run_file)]):
        main()

    assert mock_env.reset.call_count == 2
    reset_calls = mock_env.reset.call_args_list
    assert reset_calls[1].kwargs["seed"] == 43
