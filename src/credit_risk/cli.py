"""Command-line entry point: one subcommand per pipeline stage.

Usage: credit-risk [--config PATH] {data,split,experiments,train,evaluate,explain}
"""

from __future__ import annotations

import argparse
import sys

from credit_risk.config import Config, ConfigError, default_config_path, load_config
from credit_risk.data import TARGET, DataFileError, clean, load_raw, save_splits, split, verify_raw

# Stages not built yet, and the milestone that builds each one.
_PENDING = {
    "experiments": "M2",
    "train": "M5",
    "evaluate": "M6",
    "explain": "M6",
}


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


_STAGES = {"data": _run_data, "split": _run_split}


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
    for name, milestone in _PENDING.items():
        stages.add_parser(name, help=f"not implemented yet (milestone {milestone})")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config(args.config or default_config_path())
        if args.stage in _STAGES:
            _STAGES[args.stage](cfg)
        else:
            print(
                f"Stage '{args.stage}' is not implemented yet "
                f"(planned for milestone {_PENDING[args.stage]})."
            )
    except (ConfigError, DataFileError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
