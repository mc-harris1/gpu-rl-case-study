import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from replay.record import RunArtifact, RunSpec, _make_run_dir, parse_args


class TestRunSpec:
    def test_run_spec_creation(self):
        spec = RunSpec(
            env_key="pacman",
            env_id="ALE/Pacman-v5",
            obs_type="pixels",
            seed=123,
            steps=5000,
            policy="random",
            frameskip=4,
            repeat_action_probability=0.0,
            single_episode=False,
        )
        assert spec.env_key == "pacman"
        assert spec.env_id == "ALE/Pacman-v5"
        assert spec.obs_type == "pixels"
        assert spec.seed == 123
        assert spec.steps == 5000
        assert spec.policy == "random"
        assert spec.single_episode is False


class TestRunArtifact:
    def test_run_artifact_creation(self):
        spec = RunSpec(
            env_key="pacman",
            env_id="ALE/Pacman-v5",
            obs_type="pixels",
            seed=123,
            steps=100,
            policy="random",
            frameskip=4,
            repeat_action_probability=0.0,
            single_episode=False,
        )
        artifact = RunArtifact(
            run_id="test_run_123",
            created_unix_s=1234567890.0,
            spec=spec,
            actions=[0, 1, 2],
            total_reward=100.5,
            final_obs_hash="abc123",
        )
        assert artifact.run_id == "test_run_123"
        assert artifact.total_reward == 100.5
        assert len(artifact.actions) == 3


class TestMakeRunDir:
    def test_make_run_dir_creates_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            run_id = "test_run_123"
            result = _make_run_dir(base, run_id)

            assert result.exists()
            assert result.name == run_id
            assert result.parent == base

    def test_make_run_dir_fails_if_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            run_id = "test_run_123"
            d = base / run_id
            d.mkdir(parents=True)

            with pytest.raises(FileExistsError):
                _make_run_dir(base, run_id)


class TestParseArgs:
    def test_parse_args_defaults(self):
        with patch("sys.argv", ["record.py"]):
            args = parse_args()
            assert args.env_key == "pacman"
            assert args.seed == 123
            assert args.steps == 5000
            assert args.frameskip == 4
            assert args.sticky == 0.0
            assert args.runs_dir == "runs"
            assert args.render is False
            assert args.policy == "sticky_dir"
            assert args.single_episode is False

    def test_parse_args_custom_values(self):
        with patch(
            "sys.argv",
            [
                "record.py",
                "--env",
                "pacman-ram",
                "--seed",
                "456",
                "--steps",
                "1000",
                "--frameskip",
                "2",
                "--sticky",
                "0.25",
                "--runs-dir",
                "custom_runs",
                "--render",
                "--policy",
                "random",
                "--single-episode",
            ],
        ):
            args = parse_args()
            assert args.env_key == "pacman-ram"
            assert args.seed == 456
            assert args.steps == 1000
            assert args.frameskip == 2
            assert args.sticky == 0.25
            assert args.runs_dir == "custom_runs"
            assert args.render is True
            assert args.policy == "random"
            assert args.single_episode is True
