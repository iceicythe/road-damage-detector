from pathlib import Path
import json
import subprocess
import sys

from road_damage.experiments import train_config

ROOT = Path(__file__).resolve().parents[1]


def test_dry_run_works_outside_repo_without_loading_model(tmp_path):
    completed = subprocess.run([sys.executable, str(ROOT / "run.py"), "train", "--preset", "n960", "--dry-run"],
                               cwd=tmp_path, check=True, text=True, capture_output=True)
    config = json.loads(completed.stdout)
    assert config["imgsz"] == 960
    assert config["batch"] == 4
    assert Path(config["data"]).is_absolute()


def test_presets_keep_common_training_settings():
    n, s = train_config("n640"), train_config("s640")
    assert n["model"] == "yolo11n.pt"
    assert s["model"] == "yolo11s.pt"
    assert {k: v for k, v in n.items() if k not in {"name", "model"}} == {k: v for k, v in s.items() if k not in {"name", "model"}}

