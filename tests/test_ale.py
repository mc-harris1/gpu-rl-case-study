from unittest.mock import Mock, patch

import gymnasium as gym
import numpy as np
import pytest
from envs.ale import ALEEnv
from envs.base import StepResult
from gymnasium.spaces import Box, Discrete


@pytest.fixture
def mock_env():
    """Create a mock gymnasium environment."""
    env = Mock(spec=gym.Env)
    env.action_space = Mock()
    env.action_space.n = 18
    env.action_space.seed = Mock()
    return env


@pytest.fixture
def ale_env(mock_env):
    """Create an ALEEnv instance with mocked gymnasium environment."""
    with patch("envs.ale.gym.make", return_value=mock_env):
        env = ALEEnv(env_id="ALE/Pong-v5")
        env._env = mock_env
    return env


def test_ale_env_initialization():
    """Test ALEEnv initialization with default parameters."""
    with patch("envs.ale.gym.make") as mock_make:
        env = ALEEnv(env_id="ALE/Pong-v5")  # noqa: F841
        mock_make.assert_called_once_with(
            "ALE/Pong-v5",
            render_mode=None,
            frameskip=4,
            repeat_action_probability=0.0,
        )


def test_ale_env_custom_parameters():
    """Test ALEEnv initialization with custom parameters."""
    with patch("envs.ale.gym.make") as mock_make:
        env = ALEEnv(  # noqa: F841
            env_id="ALE/Breakout-v5",
            render_mode="human",
            frameskip=2,
            repeat_action_probability=0.25,
        )
        mock_make.assert_called_once_with(
            "ALE/Breakout-v5",
            render_mode="human",
            frameskip=2,
            repeat_action_probability=0.25,
        )


def test_reset_without_seed(ale_env):
    """Test reset without seed."""
    obs_array = np.random.rand(84, 84, 1).astype(np.uint8)
    ale_env._env.reset.return_value = (obs_array, {"info": "test"})

    obs, info = ale_env.reset()

    assert isinstance(obs, np.ndarray)
    np.testing.assert_array_equal(obs, obs_array)
    assert info == {"info": "test"}
    ale_env._env.reset.assert_called_once_with(seed=None)


def test_reset_with_seed(ale_env):
    """Test reset with seed."""
    obs_array = np.random.rand(84, 84, 1).astype(np.uint8)
    ale_env._env.reset.return_value = (obs_array, {"episode": 1})

    obs, info = ale_env.reset(seed=42)

    assert isinstance(obs, np.ndarray)
    ale_env._env.reset.assert_called_once_with(seed=42)
    ale_env._env.action_space.seed.assert_called_once_with(42)


def test_reset_seed_exception_handling(ale_env):
    """Test reset handles action space seed exceptions gracefully."""
    obs_array = np.random.rand(84, 84, 1).astype(np.uint8)
    ale_env._env.reset.return_value = (obs_array, {})
    ale_env._env.action_space.seed.side_effect = Exception("Seed not supported")

    obs, info = ale_env.reset(seed=42)

    assert isinstance(obs, np.ndarray)
    ale_env._env.action_space.seed.assert_called_once_with(42)


def test_step(ale_env):
    """Test step function."""
    obs_array = np.random.rand(84, 84, 1).astype(np.uint8)
    ale_env._env.step.return_value = (obs_array, 1.0, False, False, {"step": 1})

    result = ale_env.step(action=0)

    assert isinstance(result, StepResult)
    np.testing.assert_array_equal(result.obs, obs_array)
    assert result.reward == 1.0
    assert result.terminated is False
    assert result.truncated is False
    assert result.info == {"step": 1}
    ale_env._env.step.assert_called_once_with(0)


def test_step_with_termination(ale_env):
    """Test step function with terminal state."""
    obs_array = np.zeros((84, 84, 1), dtype=np.uint8)
    ale_env._env.step.return_value = (obs_array, 10.0, True, False, {})

    result = ale_env.step(action=5)

    assert result.terminated is True
    assert result.truncated is False
    assert result.reward == 10.0


def test_step_with_truncation(ale_env):
    """Test step function with truncation."""
    obs_array = np.zeros((84, 84, 1), dtype=np.uint8)
    ale_env._env.step.return_value = (obs_array, 0.0, False, True, {})

    result = ale_env.step(action=1)

    assert result.terminated is False
    assert result.truncated is True


def test_close(ale_env):
    """Test close function."""
    ale_env.close()
    ale_env._env.close.assert_called_once()


def test_action_space_n(ale_env):
    """Test action_space_n property."""
    ale_env._env.action_space = Discrete(18)

    assert ale_env.action_space_n == 18


def test_action_space_n_non_discrete(ale_env):
    """Test action_space_n raises error for non-Discrete space."""
    ale_env._env.action_space = Box(low=0, high=1, shape=(2,))

    with pytest.raises(TypeError, match="Expected Discrete action space"):
        _ = ale_env.action_space_n


def test_action_meanings(ale_env):
    """Test action_meanings property."""
    mock_unwrapped = Mock()
    mock_unwrapped.get_action_meanings.return_value = ["NOOP", "FIRE", "RIGHT", "LEFT"]
    ale_env._env.unwrapped = mock_unwrapped

    meanings = ale_env.action_meanings

    assert isinstance(meanings, list)
    assert meanings == ["NOOP", "FIRE", "RIGHT", "LEFT"]
    mock_unwrapped.get_action_meanings.assert_called_once()


def test_action_meanings_fallback(ale_env):
    """Test action_meanings fallback when unavailable."""
    from gymnasium.spaces import Discrete

    ale_env._env.action_space = Discrete(6)
    ale_env._env.unwrapped = Mock()
    ale_env._env.unwrapped.get_action_meanings.side_effect = Exception("Not supported")

    meanings = ale_env.action_meanings

    assert isinstance(meanings, list)
    assert meanings == ["0", "1", "2", "3", "4", "5"]
