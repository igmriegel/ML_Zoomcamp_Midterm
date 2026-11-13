"""Download AI4I 2020 Predictive Maintenance dataset from UCI ML Repository (id=601)."""

from __future__ import annotations

import argparse
from pathlib import Path

from ucimlrepo import fetch_ucirepo

from pdm.config import DATA_PATH


def download(output_path: Path = DATA_PATH) -> Path:
    """Fetch AI4I 2020 dataset from UCI and write it to CSV."""
    dataset = fetch_ucirepo(id=601)
    df = dataset.data.original
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download AI4I 2020 dataset from UCI.")
    parser.add_argument("--out", type=Path, default=DATA_PATH, help="Destination CSV path.")
    args = parser.parse_args()
    path = download(args.out)
    print(f"Saved dataset to {path}")


if __name__ == "__main__":
    main()
