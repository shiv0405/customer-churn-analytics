from __future__ import annotations

from pathlib import Path
import csv


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "sample_metrics.csv"


def load_metrics() -> list[dict[str, str]]:
    if not DATA_PATH.exists():
        return []
    with DATA_PATH.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def summarize(metrics: list[dict[str, str]]) -> str:
    if not metrics:
        return "No input metrics found. Add files under data/raw to begin analysis."
    return f"Loaded {len(metrics)} rows for customer-churn-analytics."


if __name__ == "__main__":
    print(summarize(load_metrics()))

