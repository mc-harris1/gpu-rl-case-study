import pytest
from envs.registry import EnvSpec, get_env_spec, list_envs, make_env


class TestEnvSpec:
    def test_env_spec_frozen(self):
        """Test that EnvSpec is immutable."""
        spec = EnvSpec(key="test", env_id="test-v5", obs_type="pixels", description="Test env")
        with pytest.raises(AttributeError):
            spec.key = "modified"


class TestListEnvs:
    def test_list_envs_returns_list(self):
        """Test that list_envs returns a list."""
        envs = list_envs()
        assert isinstance(envs, list)
        assert len(envs) > 0

    def test_list_envs_contains_specs(self):
        """Test that list_envs returns EnvSpec objects."""
        envs = list_envs()
        assert all(isinstance(env, EnvSpec) for env in envs)


class TestGetEnvSpec:
    def test_get_env_spec_valid_key(self):
        """Test getting an environment spec with valid key."""
        spec = get_env_spec("pacman")
        assert spec.key == "pacman"
        assert spec.env_id == "ALE/Pacman-v5"
        assert spec.obs_type == "pixels"

    def test_get_env_spec_valid_ram_key(self):
        """Test getting RAM environment spec."""
        spec = get_env_spec("pacman-ram")
        assert spec.key == "pacman-ram"
        assert spec.obs_type == "state"

    def test_get_env_spec_invalid_key(self):
        """Test that invalid key raises ValueError."""
        with pytest.raises(ValueError, match="Unknown env key"):
            get_env_spec("invalid-env")

    def test_get_env_spec_error_message(self):
        """Test error message contains valid keys."""
        with pytest.raises(ValueError, match="pacman"):
            get_env_spec("nonexistent")


class TestMakeEnv:
    def test_make_env_returns_tuple(self):
        """Test that make_env returns spec and env."""
        spec, env = make_env(
            env_key="pacman", render_mode=None, frameskip=4, repeat_action_probability=0.0
        )
        assert isinstance(spec, EnvSpec)
        assert spec.key == "pacman"

    def test_make_env_invalid_key(self):
        """Test make_env with invalid key raises ValueError."""
        with pytest.raises(ValueError):
            make_env(
                env_key="invalid", render_mode=None, frameskip=4, repeat_action_probability=0.0
            )
