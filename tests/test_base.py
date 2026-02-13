from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from envs.base import StepResult, obs_hash


class TestStepResult:
    def test_step_result_creation(self):
        obs = np.array([1.0, 2.0, 3.0])
        result = StepResult(
            obs=obs, reward=1.5, terminated=False, truncated=False, info={"key": "value"}
        )
        assert np.array_equal(result.obs, obs)
        assert result.reward == 1.5
        assert result.terminated is False
        assert result.truncated is False
        assert result.info == {"key": "value"}

    def test_step_result_immutable(self):
        result = StepResult(
            obs=np.array([1.0]), reward=0.0, terminated=False, truncated=False, info={}
        )
        with pytest.raises(FrozenInstanceError):
            result.reward = 5.0  # type: ignore


class TestObsHash:
    def test_obs_hash_consistency(self):
        obs = np.array([1.0, 2.0, 3.0])
        hash1 = obs_hash(obs)
        hash2 = obs_hash(obs)
        assert hash1 == hash2

    def test_obs_hash_different_arrays(self):
        obs1 = np.array([1.0, 2.0, 3.0])
        obs2 = np.array([1.0, 2.0, 4.0])

        assert obs_hash(obs1) != obs_hash(obs2)

    def test_obs_hash_different_shapes(self):
        obs1 = np.array([1.0, 2.0, 3.0])
        obs2 = np.array([[1.0, 2.0, 3.0]])

        assert obs_hash(obs1) != obs_hash(obs2)

    def test_obs_hash_2d_array(self):
        obs = np.array([[1, 2, 3], [4, 5, 6]])
        hash_val = obs_hash(obs)
        assert isinstance(hash_val, str)
        assert len(hash_val) == 64  # SHA256 hex length

    def test_obs_hash_uint8_pixels(self):
        obs = np.array([[0, 128, 255], [64, 32, 16]], dtype=np.uint8)
        hash_val = obs_hash(obs)
        assert isinstance(hash_val, str)
