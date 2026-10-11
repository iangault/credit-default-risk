"""Command-line entry point: one subcommand per pipeline stage.

Usage: credit-risk [--config PATH] {data,split,experiments,decide,train,evaluate}

Add a subcommand here as each new stage is built.
"""

from __future__ import annotations

import argparse
import sys

from credit_risk.config import Config, ConfigError, default_config_path, load_config
from credit_risk.data import (
    TARGET,
    DataFileError,
    clean,
    load_raw,
    load_split,
    save_splits,
    split,
    verify_raw,
)
from credit_risk.decision import load_model, run_decision, save_decision, save_model, train_final
from credit_risk.evaluate import evaluate, save_evaluation
from credit_risk.experiments import run_experiments, save_experiments


def _run_data(cfg: Config) -> None:
    verify_raw(cfg)
    print(f"OK: {cfg.paths.raw} matches the expected sha256.")


def _run_split(cfg: Config) -> None:
    train, test = split(clean(load_raw(cfg)), cfg)
    save_splits(train, test, cfg)
    print(
        f"Wrote {len(train)} train and {len(test)} test rows to {cfg.paths.processed_dir} "
        f"(default rate: train {train[TARGET].mean():.3f}, test {test[TARGET].mean():.3f})."
    )


def _run_experiments(cfg: Config) -> None:
    table = run_experiments(cfg)
    path = save_experiments(table, cfg)
    summary = table.set_index("experiment")[
        ["roc_auc_mean", "pr_auc_mean", "recall_mean", "precision_mean"]
    ]
    print(summary.round(3).to_string())
    print(f"Wrote {path}")


def _run_decide(cfg: Config) -> None:
    table, curves = run_decision(cfg)
    table_path, plot_path = save_decision(table, curves, cfg)
    print(table.round(3).to_string(index=False))
    print(f"Wrote {table_path} and {plot_path}")


def _run_train(cfg: Config) -> None:
    model = train_final(cfg)
    path = save_model(model, cfg)
    final, costs = cfg.final, cfg.costs
    print(
        f"Final setup: {final.experiment} with {final.calibration} calibration; costs: "
        f"missed defaulter {costs.false_negative:g}, false alarm {costs.false_positive:g}"
    )
    print(
        f"Threshold: {model.best_threshold_:.3f}; cross-validated expected cost per client: "
        f"{-model.best_score_:.3f}"
    )
    print(f"Wrote {path}")


def _run_evaluate(cfg: Config) -> None:
    model = load_model(cfg)  # stops if the model is missing or stale
    test = load_split("test", cfg)
    table, proba = evaluate(model, test, cfg)
    table_path, plot_path = save_evaluation(table, test[TARGET], proba, cfg)
    print(table.set_index("setup").round(3).to_string())
    # The last row is the train estimate; its recall, precision and counts are always
    # blank, so check only the four columns it should have.
    estimate = table.iloc[-1][["threshold", "roc_auc", "pr_auc", "expected_cost"]]
    if estimate.isna().any():
        print(
            "Note: part of the train estimate was not found; run `make experiments` and "
            "`make decide` with the current config to fill it."
        )
    print(f"Wrote {table_path} and {plot_path}")


_STAGES = {
    "data": _run_data,
    "split": _run_split,
    "experiments": _run_experiments,
    "decide": _run_decide,
    "train": _run_train,
    "evaluate": _run_evaluate,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="credit-risk", description="Run one stage of the credit default pipeline."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="YAML config file (default: configs/base.yaml in this project)",
    )
    stages = parser.add_subparsers(dest="stage", required=True, metavar="STAGE")
    stages.add_parser("data", help="check the raw CSV against its expected sha256")
    stages.add_parser("split", help="clean the raw data and write a stratified train/test split")
    stages.add_parser("experiments", help="cross-validate each experiment in the config on train")
    stages.add_parser(
        "decide", help="compare calibrated, cost-tuned candidates on train (config 'decision')"
    )
    stages.add_parser("train", help="fit the 'final' setup on all of train and save it")
    stages.add_parser("evaluate", help="score the saved model on the test set")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config(args.config or default_config_path())
        _STAGES[args.stage](cfg)
    except (ConfigError, DataFileError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
