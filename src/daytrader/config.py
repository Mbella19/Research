"""Config access: instrument.yaml (physical facts) + experiment.yaml (knobs).

`override_experiment` lets pipeline stages test alternate knobs (e.g. barrier
geometries) in-process; overrides nest one level deep (dict-of-dict merge).
"""
import copy
from functools import lru_cache
from pathlib import Path

from .utils import paths
from .utils.hashio import load_yaml

_exp_overrides: dict = {}


@lru_cache(maxsize=None)
def instrument() -> dict:
    return load_yaml(paths.CONFIG_DIR / "instrument.yaml")


@lru_cache(maxsize=None)
def _experiment_raw() -> dict:
    return load_yaml(paths.CONFIG_DIR / "experiment.yaml")


def experiment() -> dict:
    base = copy.deepcopy(_experiment_raw())
    for k, v in _exp_overrides.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k].update(v)
        else:
            base[k] = v
    return base


def override_experiment(**sections) -> None:
    """e.g. override_experiment(labels={'tp_atr': 3.0, 'sl_atr': 1.5})."""
    _exp_overrides.update(sections)


def clear_overrides() -> None:
    _exp_overrides.clear()


def costs(profile: str | None = None) -> dict:
    """Cost profile: named, or the active one from experiment.yaml."""
    ins = instrument()
    if "cost_profiles" in ins:
        profiles = ins["cost_profiles"]
        name = profile or experiment().get("cost_profile") or next(iter(profiles))
        return profiles[name]
    return ins["costs"]  # legacy single-profile layout


def data_path(rel: str) -> Path:
    """Data files are referenced relative to the project root."""
    return paths.PROJECT_ROOT / rel


def real_sources(include_locked_oos: bool = False) -> dict[str, Path]:
    d = instrument()["data"]["real"]
    out = {
        "real_training": data_path(d["training"]),
        "real_validation": data_path(d["validation"]),
    }
    if include_locked_oos:
        out["real_locked_oos"] = data_path(d["locked_oos"])
    return out


def synth_sources() -> dict[str, Path]:
    import re

    glob = instrument()["data"]["synthetic"]["glob"]
    files = sorted(paths.PROJECT_ROOT.glob(glob))
    out: dict[str, Path] = {}
    for f in files:
        m = re.search(r"[_\- ]U(\d+)", f.stem, flags=re.IGNORECASE)
        slug = re.sub(r"\W+", "_", f.stem.lower())
        key = f"synth_u{m.group(1)}" if m else f"synth_{slug}"
        out[key] = f
    return out
