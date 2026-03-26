from __future__ import annotations

from pathlib import Path

from customer_churn_analytics.config import ProjectPaths
from customer_churn_analytics.data_generation import generate_customer_dataset, write_dataset
from customer_churn_analytics.pipeline import run_training_pipeline


def test_training_pipeline_writes_expected_outputs(tmp_path: Path) -> None:
    paths = ProjectPaths.from_root(tmp_path)
    dataset = generate_customer_dataset(rows=1800, seed=11)
    write_dataset(dataset, paths.raw_data_path)

    result = run_training_pipeline(paths)

    assert result.dataset_rows == 1800
    assert result.metrics["roc_auc"] >= 0.85
    assert Path(result.artifacts["metrics_path"]).exists()
    assert Path(result.artifacts["scored_output_path"]).exists()
    assert Path(result.artifacts["segment_risk_path"]).exists()
    assert Path(result.artifacts["retention_playbook_path"]).exists()
