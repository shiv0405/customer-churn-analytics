from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

from .config import ProjectPaths
from .pipeline import score_batch_frame


def load_trained_model(paths: ProjectPaths) -> object:
    if not paths.model_path.exists():
        raise FileNotFoundError(f"Expected trained model at {paths.model_path}")
    with paths.model_path.open("rb") as handle:
        return pickle.load(handle)


def score_batch_file(
    *,
    paths: ProjectPaths,
    input_path: Path,
    threshold: float,
    output_path: Path | None = None,
) -> Path:
    dataset = pd.read_csv(input_path)
    model = load_trained_model(paths)
    scored = score_batch_frame(model, dataset, threshold)
    destination = output_path or paths.processed_dir / "batch_scoring_output.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(destination, index=False)
    return destination
