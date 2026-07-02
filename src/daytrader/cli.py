"""daytrader CLI — every stage of the pipeline is a subcommand.

Subcommands are added as the corresponding pipeline stage is implemented,
so the CLI never advertises functionality that does not exist yet.
"""
import argparse

from .utils import paths
from .utils.log import get_logger

log = get_logger("cli")


def cmd_ingest(args: argparse.Namespace) -> None:
    from .data.loader import ingest_all
    from .utils.artifacts import new_run_dir
    from .utils.hashio import save_json

    run_dir = new_run_dir("ingest")
    log.info(f"artifacts → {run_dir}")
    metas = ingest_all(force=args.force)
    save_json([m["integrity"] for m in metas], run_dir / "integrity_report.json")

    lines = [
        "# Ingest integrity report", "",
        "| source | rows | span | days | dup | mono | ohlc_bad | gaps>1m | spread p50/p95 | clean |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for m in metas:
        r = m["integrity"]
        lines.append(
            f"| {r['name']} | {r['rows']:,} | {r['start'][:10]} → {r['end'][:10]} "
            f"| {r['trading_days']} | {r['dup_ts']} | {r['non_monotonic']} "
            f"| {r['ohlc_violations']} | {r['intraday_gaps_gt1m']} "
            f"| {r['spread_pts']['p50']}/{r['spread_pts']['p95']} "
            f"| {'✅' if r['clean'] else '⚠️'} |"
        )
    (run_dir / "integrity_report.md").write_text("\n".join(lines), encoding="utf-8")
    log.info(f"integrity report written: {run_dir / 'integrity_report.md'}")


def cmd_profile(args: argparse.Namespace) -> None:
    from .eval.profile import run_profile

    run_profile()


def cmd_audit_synth(args: argparse.Namespace) -> None:
    from .eval.leakage import run_leakage_audit

    run_leakage_audit()


def cmd_features(args: argparse.Namespace) -> None:
    from .config import synth_sources
    from .features.registry import build_features

    sources = (["real_training", "real_validation"] + list(synth_sources().keys())
               if args.source == "all" else [args.source])
    for s in sources:
        build_features(s, refresh=args.refresh)


def cmd_labels(args: argparse.Namespace) -> None:
    from .config import synth_sources
    from .labels.triple_barrier import build_labels

    sources = (["real_training", "real_validation"] + list(synth_sources().keys())
               if args.source == "all" else [args.source])
    for s in sources:
        build_labels(s, refresh=args.refresh)


def cmd_train_lgbm(args: argparse.Namespace) -> None:
    from .models import pipeline

    if args.stage == "search":
        pipeline.run_search(n_configs=args.n_configs)
    elif args.stage == "evreg":
        from .models import evreg

        evreg.run_evreg()
    elif args.stage == "barriers":
        pipeline.run_barriers()
    elif args.stage == "synth":
        pipeline.run_synth_gate()
    elif args.stage == "arena":
        from .models import arena

        arena.run_arena()
    else:
        if args.geometry == "b":     # pre-registered challenger tp2/sl1/H24 (D-026 B2)
            from .config import experiment as _exp
            from .config import override_experiment

            override_experiment(labels={**_exp()["labels"], "tp_atr": 2.0,
                                        "sl_atr": 1.0, "horizon_bars": 24})
        pipeline.run_final(out_name=args.out)


def cmd_train_tcn(args: argparse.Namespace) -> None:
    from .models import tcn_pipeline

    if args.stage == "cv":
        tcn_pipeline.run_tcn_cv()
    else:
        tcn_pipeline.run_tcn_final()


def cmd_champion(args: argparse.Namespace) -> None:
    from .models.blend import run_champion

    run_champion()


def cmd_validate(args: argparse.Namespace) -> None:
    from .eval import validate

    if args.look == "champion":
        validate.run_champion_look(args.model)
    elif args.look == "sizing":
        validate.run_sizing_look(args.model, min_ev=args.min_ev)
    else:
        validate.run_threshold_sweep(args.model)


def cmd_robustness(args: argparse.Namespace) -> None:
    from .eval.robustness import run_battery

    run_battery(args.model)


def cmd_freeze(args: argparse.Namespace) -> None:
    from .eval.freeze import run_freeze

    run_freeze()


def cmd_oos(args: argparse.Namespace) -> None:
    from .eval.oos import run_oos

    run_oos(confirm=args.confirm_single_shot)


def cmd_forward(args: argparse.Namespace) -> None:
    from .eval.oos import run_forward

    run_forward(args.csv)


def cmd_signal(args: argparse.Namespace) -> None:
    from .eval.live import run_signal

    run_signal(args.csv, args.equity)


def cmd_direction(args: argparse.Namespace) -> None:
    from .models import direction

    if args.stage == "search":
        direction.run_search()
    elif args.stage == "final":
        direction.run_final()
    else:
        direction.validation_look(args.gate, args.risk)


def cmd_portfolio(args: argparse.Namespace) -> None:
    if args.stage == "grid":
        from .portfolio import grid

        grid.run_grid()
        return
    from .portfolio import book

    if args.stage == "oof":
        book.run_oof()
    elif args.stage == "validate":
        book.run_validate(which=args.which)
    elif args.stage == "robustness":
        book.run_robustness()
    elif args.stage == "freeze-v2":
        book.run_freeze_v2()
    elif args.stage == "freeze-v3":
        book.run_freeze_v3()
    elif args.stage == "finaltest":
        book.run_finaltest(confirm=args.confirm_single_shot)
    elif args.stage == "fullhistory":
        book.run_fullhistory()


def cmd_benchmark(args: argparse.Namespace) -> None:
    from .backtest import signals as sigmod
    from .backtest.engine import make_cost_cfg, make_risk_cfg, run_backtest
    from .config import experiment
    from .data.loader import load_bars
    from .eval.report import render_backtest
    from .features.registry import build_features
    from .utils.artifacts import new_run_dir

    run_dir = new_run_dir(f"benchmark_{args.source}")
    log.info(f"artifacts → {run_dir}")
    feat = build_features(args.source)
    df1m = load_bars(args.source)
    cost, risk = make_cost_cfg(), make_risk_cfg()
    strategies = ["ms", "momentum"] if args.strategy == "all" else [args.strategy]
    for name in strategies:
        sig = sigmod.ms_rule(feat) if name == "ms" else sigmod.momentum(feat)
        n_sig = int((sig["side"] != 0).sum())
        log.info(f"{name}: {n_sig:,} raw signals")
        res = run_backtest(df1m, sig, cost, risk)
        render_backtest(run_dir, name, res, experiment()["backtest"]["equity0"], df1m)


def main() -> None:
    paths.ensure_dirs()
    p = argparse.ArgumentParser(prog="daytrader")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_ing = sub.add_parser("ingest", help="CSV → parquet + integrity report")
    p_ing.add_argument("--force", action="store_true", help="re-ingest even if unchanged")
    p_ing.set_defaults(fn=cmd_ingest)

    p_prof = sub.add_parser("profile", help="session/spread profiling + synth realism report")
    p_prof.set_defaults(fn=cmd_profile)

    p_aud = sub.add_parser("audit-synth", help="synthetic↔real leakage audit")
    p_aud.set_defaults(fn=cmd_audit_synth)

    p_feat = sub.add_parser("features", help="build + cache feature matrices")
    p_feat.add_argument("--source", default="all",
                        help="source name or 'all' (never includes locked OOS)")
    p_feat.add_argument("--refresh", action="store_true")
    p_feat.set_defaults(fn=cmd_features)

    p_lab = sub.add_parser("labels", help="build + cache triple-barrier labels")
    p_lab.add_argument("--source", default="all")
    p_lab.add_argument("--refresh", action="store_true")
    p_lab.set_defaults(fn=cmd_labels)

    p_bm = sub.add_parser("benchmark", help="rule-based benchmark backtests")
    p_bm.add_argument("--source", default="real_training")
    p_bm.add_argument("--strategy", default="all", choices=["all", "ms", "momentum"])
    p_bm.set_defaults(fn=cmd_benchmark)

    p_tr = sub.add_parser("train-lgbm", help="LightGBM purged-CV pipeline")
    p_tr.add_argument("--stage", required=True,
                      choices=["search", "barriers", "synth", "final", "evreg", "arena"])
    p_tr.add_argument("--n-configs", type=int, default=None)
    p_tr.add_argument("--out", default="lgbm_final",
                      help="final-stage artifact dir under models/ (v3: lgbm_final_v3)")
    p_tr.add_argument("--geometry", default="a", choices=["a", "b"],
                      help="final stage label geometry (b = tp2/sl1/H24 challenger)")
    p_tr.set_defaults(fn=cmd_train_lgbm)

    p_tc = sub.add_parser("train-tcn", help="TCN purged-CV / final ensemble")
    p_tc.add_argument("--stage", required=True, choices=["cv", "final"])
    p_tc.set_defaults(fn=cmd_train_tcn)

    p_ch = sub.add_parser("champion", help="stacking blend + champion selection")
    p_ch.set_defaults(fn=cmd_champion)

    p_val = sub.add_parser("validate", help="pre-registered VALIDATION looks (ledgered)")
    p_val.add_argument("--look", required=True,
                       choices=["champion", "threshold", "sizing"])
    p_val.add_argument("--model", default=None, choices=[None, "gbt", "tcn", "blend"])
    p_val.add_argument("--min-ev", type=float, default=None)
    p_val.set_defaults(fn=cmd_validate)

    p_rob = sub.add_parser("robustness", help="stress battery on chosen recipe (one look)")
    p_rob.add_argument("--model", default=None, choices=[None, "gbt", "tcn", "blend"])
    p_rob.set_defaults(fn=cmd_robustness)

    p_fz = sub.add_parser("freeze", help="freeze FINAL artifact (immutable)")
    p_fz.set_defaults(fn=cmd_freeze)

    p_oos = sub.add_parser("oos", help="LOCKED OOS — runs exactly once")
    p_oos.add_argument("--confirm-single-shot", action="store_true")
    p_oos.set_defaults(fn=cmd_oos)

    p_fw = sub.add_parser("forward", help="frozen artifact on a new MT5 export")
    p_fw.add_argument("--csv", required=True)
    p_fw.set_defaults(fn=cmd_forward)

    p_sg = sub.add_parser("signal", help="live decision from a fresh MT5 export (frozen policy)")
    p_sg.add_argument("--csv", required=True)
    p_sg.add_argument("--equity", type=float, default=None)
    p_sg.set_defaults(fn=cmd_signal)

    p_dir = sub.add_parser("direction", help="Phase-3 to-EOD direction system")
    p_dir.add_argument("--stage", required=True, choices=["search", "final", "validate"])
    p_dir.add_argument("--gate", type=float, default=0.15)
    p_dir.add_argument("--risk", type=float, default=None)
    p_dir.set_defaults(fn=cmd_direction)

    p_pf = sub.add_parser("portfolio", help="v2 two-sleeve portfolio (D-021)")
    p_pf.add_argument("--stage", required=True,
                      choices=["grid", "oof", "validate", "robustness",
                               "freeze-v2", "freeze-v3", "finaltest", "fullhistory"])
    p_pf.add_argument("--which", default="s2", choices=["s2", "portfolio"],
                      help="validate stage: sleeve-2 alone or full portfolio")
    p_pf.add_argument("--confirm-single-shot", action="store_true")
    p_pf.set_defaults(fn=cmd_portfolio)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
