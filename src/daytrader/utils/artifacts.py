"""Per-run artifact directories: config snapshots, hashes, logs, plots."""
import shutil
from datetime import datetime
from pathlib import Path

from . import paths
from .hashio import save_json, sha256_file
from .log import add_run_file_handler


def new_run_dir(kind: str) -> Path:
    """Create runs/<kind>_<timestamp>/ with config snapshot + file logging."""
    paths.ensure_dirs()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = paths.RUNS_DIR / f"{kind}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "plots").mkdir()
    cfg_snap = run_dir / "config"
    cfg_snap.mkdir()
    hashes = {}
    for f in sorted(paths.CONFIG_DIR.glob("*.yaml")):
        shutil.copy2(f, cfg_snap / f.name)
        hashes[f.name] = sha256_file(f)
    save_json(hashes, run_dir / "config" / "hashes.json")
    add_run_file_handler(run_dir)
    return run_dir
