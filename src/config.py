from dataclasses import dataclass
from pathlib import Path

@dataclass
class Settings:
    base_model: str = "google/flan-t5-small"
    project_root: Path = Path(__file__).resolve().parents[1]
    data_dir: Path = project_root / "data"
    out_dir: Path = project_root / "outputs"
    model_dir: Path = out_dir / "fine_tuned_model"
    max_input_len: int = 256
    max_output_len: int = 160

SETTINGS = Settings()
