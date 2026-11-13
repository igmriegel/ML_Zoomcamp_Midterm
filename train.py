"""CLI entrypoint to train final binary and multi-label models."""

from __future__ import annotations

import argparse
from pathlib import Path

from pdm.config import DATA_PATH, METRICS_PATH, MODEL_PATH, SEED
from pdm.training import train_and_evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Train predictive maintenance models.")
    parser.add_argument("--data", type=Path, default=DATA_PATH, help="Path to AI4I 2020 CSV.")
    parser.add_argument("--out", type=Path, default=MODEL_PATH, help="Output model.bin path.")
    parser.add_argument(
        "--metrics", type=Path, default=METRICS_PATH, help="Output metrics.json path."
    )
    parser.add_argument("--fast", action="store_true", help="Use reduced CV iterations for CI.")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed.")
    args = parser.parse_args()

    metrics = train_and_evaluate(
        data_path=args.data,
        model_path=args.out,
        metrics_path=args.metrics,
        fast=args.fast,
        seed=args.seed,
    )
    tb = metrics["test_binary"]
    print(
        f"Winner: {metrics['winner']} | thr={metrics['threshold']} | "
        f"Test PR-AUC={tb['pr_auc']:.4f} | ROC-AUC={tb['roc_auc']:.4f} | "
        f"Recall={tb['recall']:.4f} | F1={tb['f1']:.4f}"
    )


if __name__ == "__main__":
    main()
