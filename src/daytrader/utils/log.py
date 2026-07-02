"""Logging: rich console at INFO + per-run file at DEBUG."""
import logging
from pathlib import Path

from rich.logging import RichHandler

_CONSOLE_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    global _CONSOLE_CONFIGURED
    root = logging.getLogger("daytrader")
    if not _CONSOLE_CONFIGURED:
        root.setLevel(logging.DEBUG)
        console = RichHandler(rich_tracebacks=True, show_path=False, markup=False)
        console.setLevel(logging.INFO)
        console.setFormatter(logging.Formatter("%(message)s", datefmt="%H:%M:%S"))
        root.addHandler(console)
        _CONSOLE_CONFIGURED = True
    return root.getChild(name) if name else root


def add_run_file_handler(run_dir: Path) -> logging.Handler:
    """Attach a DEBUG file handler writing into the run directory."""
    run_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    )
    logging.getLogger("daytrader").addHandler(fh)
    return fh
