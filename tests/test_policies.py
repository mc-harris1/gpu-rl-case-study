import numpy as np
import pytest
from replay.policies import (
    RandomPolicy,
    StickyDirectionalPolicy,
    list_policies,
    make_policy,
)


class TestListPolicies:
    def test_list_policies_returns_list(self):
        """Test that list_policies returns a list."""
        policies = list_policies()
        assert isinstance(policies, list)
        assert len(policies) > 0

    def test_list_policies_contains_expected(self):
        """Test that list_policies contains expected policy names."""
        policies = list_policies()
        assert "random" in policies
        assert "sticky_dir" in policies

    def test_list_policies_is_sorted(self):
        """Test that list_policies returns sorted list."""
        policies = list_policies()
        assert policies == sorted(policies)


class TestMakePolicy:
    def test_make_policy_random(self):
        """Test making a random policy."""
        policy = make_policy("random")
        assert isinstance(policy, RandomPolicy)
        assert policy.name == "random"

    def test_make_policy_sticky_dir(self):
        """Test making a sticky directional policy."""
        policy = make_policy("sticky_dir")
        assert isinstance(policy, StickyDirectionalPolicy)
        assert policy.name == "sticky_dir"

    def test_make_policy_invalid_name(self):
        """Test that invalid policy name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown policy"):
            make_policy("nonexistent_policy")

    def test_make_policy_error_message(self):
        """Test error message contains valid policy names."""
        with pytest.raises(ValueError, match="random"):
            make_policy("invalid")

    def test_make_policy_creates_fresh_instance(self):
        """Test that make_policy creates fresh instances."""
        policy1 = make_policy("random")
        policy2 = make_policy("random")
        # Should be different instances
        assert policy1 is not policy2


class TestRandomPolicy:
    def test_random_policy_initialization(self):
        """Test RandomPolicy initialization."""
        policy = RandomPolicy()
        assert policy.name == "random"

    def test_random_policy_reset(self):
        """Test RandomPolicy reset."""
        policy = RandomPolicy()
        policy.reset(seed=42, action_meanings=["NOOP", "FIRE"], action_space_n=2)

        assert hasattr(policy, "rng")
        assert hasattr(policy, "n")
        assert policy.n == 2

    def test_random_policy_act(self):
        """Test RandomPolicy act returns valid action."""
        policy = RandomPolicy()
        policy.reset(seed=42, action_meanings=["NOOP", "FIRE"], action_space_n=2)

        obs = np.zeros((84, 84, 3), dtype=np.uint8)
        action = policy.act(step=0, obs=obs, reward=0.0, done=False)

        assert isinstance(action, int)
        assert 0 <= action < 2

    def test_random_policy_deterministic_with_seed(self):
        """Test RandomPolicy is deterministic with same seed."""
        policy1 = RandomPolicy()
        policy1.reset(seed=123, action_meanings=["A", "B", "C"], action_space_n=3)

        policy2 = RandomPolicy()
        policy2.reset(seed=123, action_meanings=["A", "B", "C"], action_space_n=3)

        obs = np.zeros((84, 84, 3), dtype=np.uint8)

        actions1 = [policy1.act(step=i, obs=obs, reward=0.0, done=False) for i in range(10)]
        actions2 = [policy2.act(step=i, obs=obs, reward=0.0, done=False) for i in range(10)]

        assert actions1 == actions2

    def test_random_policy_different_seeds(self):
        """Test RandomPolicy produces different sequences with different seeds."""
        policy1 = RandomPolicy()
        policy1.reset(seed=42, action_meanings=["A", "B", "C"], action_space_n=3)

        policy2 = RandomPolicy()
        policy2.reset(seed=123, action_meanings=["A", "B", "C"], action_space_n=3)

        obs = np.zeros((84, 84, 3), dtype=np.uint8)

        actions1 = [policy1.act(step=i, obs=obs, reward=0.0, done=False) for i in range(10)]
        actions2 = [policy2.act(step=i, obs=obs, reward=0.0, done=False) for i in range(10)]

        # Very unlikely to be identical with different seeds
        assert actions1 != actions2


class TestStickyDirectionalPolicy:
    def test_sticky_dir_policy_initialization(self):
        """Test StickyDirectionalPolicy initialization with default params."""
        policy = StickyDirectionalPolicy()
        assert policy.name == "sticky_dir"
        assert policy.stuck_window == 30
        assert policy.jitter_prob == 0.02

    def test_sticky_dir_policy_custom_params(self):
        """Test StickyDirectionalPolicy with custom parameters."""
        policy = StickyDirectionalPolicy(stuck_window=50, jitter_prob=0.05)
        assert policy.stuck_window == 50
        assert policy.jitter_prob == 0.05

    def test_sticky_dir_policy_reset_with_directions(self):
        """Test StickyDirectionalPolicy reset with directional action meanings."""
        policy = StickyDirectionalPolicy()
        action_meanings = ["NOOP", "UP", "RIGHT", "DOWN", "LEFT", "FIRE"]
        policy.reset(seed=42, action_meanings=action_meanings, action_space_n=6)

        assert hasattr(policy, "rng")
        assert hasattr(policy, "dir_actions")
        assert hasattr(policy, "cur_action")
        # Should find UP, RIGHT, DOWN, LEFT
        assert len(policy.dir_actions) == 4
        assert 1 in policy.dir_actions  # UP
        assert 2 in policy.dir_actions  # RIGHT
        assert 3 in policy.dir_actions  # DOWN
        assert 4 in policy.dir_actions  # LEFT

    def test_sticky_dir_policy_reset_without_directions(self):
        """Test StickyDirectionalPolicy reset without directional meanings (fallback)."""
        policy = StickyDirectionalPolicy()
        action_meanings = ["ACTION_0", "ACTION_1", "ACTION_2", "ACTION_3", "ACTION_4"]
        policy.reset(seed=42, action_meanings=action_meanings, action_space_n=5)

        # Should fall back to first 4 actions
        assert len(policy.dir_actions) == 4
        assert policy.dir_actions == [0, 1, 2, 3]

    def test_sticky_dir_policy_fallback_with_few_actions(self):
        """Test StickyDirectionalPolicy fallback with fewer than 4 actions."""
        policy = StickyDirectionalPolicy()
        action_meanings = ["ACT_0", "ACT_1"]
        policy.reset(seed=42, action_meanings=action_meanings, action_space_n=2)

        # Should handle fewer than 4 actions
        assert len(policy.dir_actions) == 2

    def test_sticky_dir_policy_act(self):
        """Test StickyDirectionalPolicy act returns valid action."""
        policy = StickyDirectionalPolicy()
        action_meanings = ["NOOP", "UP", "RIGHT", "DOWN", "LEFT"]
        policy.reset(seed=42, action_meanings=action_meanings, action_space_n=5)

        obs = np.zeros((84, 84, 3), dtype=np.uint8)
        action = policy.act(step=0, obs=obs, reward=0.0, done=False)

        assert isinstance(action, int)
        assert action in policy.dir_actions

    def test_sticky_dir_policy_resets_progress_on_positive_reward(self):
        """Test that positive reward resets progress counter."""
        policy = StickyDirectionalPolicy()
        action_meanings = ["NOOP", "UP", "RIGHT", "DOWN", "LEFT"]
        policy.reset(seed=42, action_meanings=action_meanings, action_space_n=5)

        obs = np.zeros((84, 84, 3), dtype=np.uint8)

        # Take some steps without reward
        for i in range(5):
            policy.act(step=i, obs=obs, reward=0.0, done=False)

        initial_action = policy.cur_action

        # Positive reward should reset progress
        policy.act(step=5, obs=obs, reward=1.0, done=False)
        assert policy._since_progress == 0

        # Should still be on same action (sticky behavior)
        action = policy.act(step=6, obs=obs, reward=0.0, done=False)
        assert action == initial_action

    def test_sticky_dir_policy_rotates_when_stuck(self):
        """Test that policy rotates direction when stuck."""
        policy = StickyDirectionalPolicy(stuck_window=5, jitter_prob=0.0)
        action_meanings = ["UP", "RIGHT", "DOWN", "LEFT"]
        policy.reset(seed=42, action_meanings=action_meanings, action_space_n=4)

        obs = np.zeros((84, 84, 3), dtype=np.uint8)

        initial_action = policy.cur_action

        # Take steps without reward to get stuck
        for i in range(6):
            policy.act(step=i, obs=obs, reward=0.0, done=False)

        # Should rotate to next direction after stuck_window steps
        assert policy.cur_action != initial_action

    def test_sticky_dir_policy_resets_on_done(self):
        """Test that policy resets progress counter on done."""
        policy = StickyDirectionalPolicy()
        action_meanings = ["UP", "RIGHT", "DOWN", "LEFT"]
        policy.reset(seed=42, action_meanings=action_meanings, action_space_n=4)

        obs = np.zeros((84, 84, 3), dtype=np.uint8)

        # Build up progress
        for i in range(10):
            policy.act(step=i, obs=obs, reward=0.0, done=False)

        # done=True should reset progress
        policy.act(step=10, obs=obs, reward=0.0, done=True)
        assert policy._since_progress == 0

    def test_sticky_dir_policy_deterministic_with_seed(self):
        """Test StickyDirectionalPolicy is deterministic with same seed."""
        policy1 = StickyDirectionalPolicy(jitter_prob=0.0, stuck_window=10)
        policy1.reset(seed=123, action_meanings=["UP", "RIGHT", "DOWN", "LEFT"], action_space_n=4)

        policy2 = StickyDirectionalPolicy(jitter_prob=0.0, stuck_window=10)
        policy2.reset(seed=123, action_meanings=["UP", "RIGHT", "DOWN", "LEFT"], action_space_n=4)

        obs = np.zeros((84, 84, 3), dtype=np.uint8)

        actions1 = [policy1.act(step=i, obs=obs, reward=0.0, done=False) for i in range(20)]
        actions2 = [policy2.act(step=i, obs=obs, reward=0.0, done=False) for i in range(20)]

        assert actions1 == actions2

    def test_sticky_dir_policy_jitter_changes_action(self):
        """Test that jitter probability can change action."""
        # Use high jitter probability to ensure it triggers
        policy = StickyDirectionalPolicy(jitter_prob=1.0, stuck_window=100)
        action_meanings = ["UP", "RIGHT", "DOWN", "LEFT"]
        policy.reset(seed=42, action_meanings=action_meanings, action_space_n=4)

        obs = np.zeros((84, 84, 3), dtype=np.uint8)

        # With jitter_prob=1.0, every step should potentially select new random direction
        # Take enough steps to see some variation
        actions = [policy.act(step=i, obs=obs, reward=0.0, done=False) for i in range(10)]

        # Should have some action (all from dir_actions)
        assert all(a in policy.dir_actions for a in actions)
