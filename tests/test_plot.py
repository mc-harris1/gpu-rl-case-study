from unittest.mock import patch

import pandas as pd
import pytest
from metrics.plot import main, parse_args


class TestParseArgs:
    def test_parse_args_with_run_dir(self):
        with patch("sys.argv", ["prog", "--run-dir", "/path/to/run"]):
            args = parse_args()
            assert args.run_dir == "/path/to/run"

    def test_parse_args_missing_run_dir(self):
        with patch("sys.argv", ["prog"]):
            with pytest.raises(SystemExit):
                parse_args()


class TestMain:
    @patch("metrics.plot.plt.savefig")
    @patch("metrics.plot.plt.figure")
    @patch("metrics.plot.pd.read_csv")
    def test_main_success(self, mock_read_csv, mock_figure, mock_savefig, tmp_path, capsys):
        df = pd.DataFrame(
            {"episode_id": [0, 0, 0], "step": [1, 2, 3], "reward": ["1.0", "2.0", "3.0"]}
        )
        mock_read_csv.return_value = df

        run_dir = tmp_path / "runs" / "test_run"
        run_dir.mkdir(parents=True)
        (run_dir / "telemetry.csv").touch()

        with patch("sys.argv", ["prog", "--run-dir", str(run_dir)]):
            main()

        captured = capsys.readouterr()
        assert "Saved:" in captured.out
        # Should create both plots
        assert mock_savefig.call_count == 2

    def test_main_missing_telemetry(self, tmp_path):
        run_dir = tmp_path / "runs" / "test_run"
        run_dir.mkdir(parents=True)

        with patch("sys.argv", ["prog", "--run-dir", str(run_dir)]):
            with pytest.raises(FileNotFoundError, match="Missing telemetry.csv"):
                main()

    @patch("metrics.plot.plt.savefig")
    @patch("metrics.plot.plt.figure")
    @patch("metrics.plot.pd.read_csv")
    def test_main_handles_non_numeric_rewards(
        self, mock_read_csv, mock_figure, mock_savefig, tmp_path
    ):
        df = pd.DataFrame(
            {"episode_id": [0, 0, 0], "step": [1, 2, 3], "reward": ["1.5", "invalid", "2.5"]}
        )
        mock_read_csv.return_value = df

        run_dir = tmp_path / "runs" / "test_run"
        run_dir.mkdir(parents=True)
        (run_dir / "telemetry.csv").touch()

        with patch("sys.argv", ["prog", "--run-dir", str(run_dir)]):
            main()

        # Both plots should be created
        assert mock_savefig.call_count == 2

    @patch("metrics.plot.plt.savefig")
    @patch("metrics.plot.plt.figure")
    @patch("metrics.plot.pd.read_csv")
    def test_main_backward_compatibility_no_episode_id(
        self, mock_read_csv, mock_figure, mock_savefig, tmp_path
    ):
        """Test backward compatibility with old CSV files without episode_id."""
        df = pd.DataFrame({"step": [1, 2, 3], "reward": ["1.0", "2.0", "3.0"]})
        mock_read_csv.return_value = df

        run_dir = tmp_path / "runs" / "test_run"
        run_dir.mkdir(parents=True)
        (run_dir / "telemetry.csv").touch()

        with patch("sys.argv", ["prog", "--run-dir", str(run_dir)]):
            main()

        # Both plots should be created
        assert mock_savefig.call_count == 2

    @patch("metrics.plot.plt.savefig")
    @patch("metrics.plot.plt.figure")
    @patch("metrics.plot.pd.read_csv")
    def test_main_handles_wall_ms_column(self, mock_read_csv, mock_figure, mock_savefig, tmp_path):
        """Test that wall_ms column is coerced to numeric."""
        df = pd.DataFrame(
            {
                "episode_id": [0, 0, 0],
                "step": [1, 2, 3],
                "reward": ["1.0", "2.0", "3.0"],
                "wall_ms": ["10.5", "invalid", "12.3"],
            }
        )
        mock_read_csv.return_value = df

        run_dir = tmp_path / "runs" / "test_run"
        run_dir.mkdir(parents=True)
        (run_dir / "telemetry.csv").touch()

        with patch("sys.argv", ["prog", "--run-dir", str(run_dir)]):
            main()

        # Both plots should be created
        assert mock_savefig.call_count == 2
