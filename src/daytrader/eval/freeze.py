"""Freeze the validated recipe into an immutable FINAL artifact record."""
from datetime import datetime

from ..utils import paths
from ..utils.hashio import load_json, save_json, sha256_file
from ..utils.log import get_logger

log = get_logger("eval.freeze")


def run_freeze() -> None:
    champ = load_json(paths.MODELS_DIR / "champion.json")
    files = {}
    for f in sorted(paths.MODELS_DIR.rglob("*")):
        if f.is_file() and f.suffix in (".txt", ".pkl", ".pt", ".json", ".parquet"):
            files[str(f.relative_to(paths.MODELS_DIR))] = sha256_file(f)
    frozen = {
        "frozen_at": datetime.now().isoformat(),
        "champion": champ["champion"],
        "champion_evidence": champ["results"],
        "recipe": champ.get("recipe"),
        "file_hashes": files,
    }
    out = paths.MODELS_DIR / "FINAL_FROZEN.json"
    if out.exists():
        raise SystemExit("REFUSED: FINAL_FROZEN.json already exists. "
                         "Delete it consciously if you truly intend to re-freeze "
                         "(and note it in the ledger).")
    save_json(frozen, out)
    log.info(f"FROZEN: champion={champ['champion']} → {out}")
