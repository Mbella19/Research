"""Project paths. Everything is relative to the project root so the whole
directory can be copied/renamed for another instrument and still work."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
PARQUET_DIR = DATA_DIR / "parquet"
RUNS_DIR = PROJECT_ROOT / "runs"
NOTES_DIR = PROJECT_ROOT / "notes"
MODELS_DIR = PROJECT_ROOT / "models"


def ensure_dirs() -> None:
    for d in (DATA_DIR, PARQUET_DIR, RUNS_DIR, NOTES_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)
