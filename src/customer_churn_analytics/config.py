from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    raw_data_path: Path
    scoring_input_path: Path
    processed_dir: Path
    artifacts_dir: Path
    docs_dir: Path
    model_path: Path
    metrics_path: Path
    portfolio_kpis_path: Path
    leaderboard_path: Path
    feature_importance_path: Path
    segment_risk_path: Path
    retention_playbook_path: Path
    summary_markdown_path: Path
    executive_html_path: Path
    scored_output_path: Path

    @classmethod
    def from_root(cls, root: Path | None = None) -> "ProjectPaths":
        resolved_root = (root or Path(__file__).resolve().parents[2]).resolve()
        processed_dir = resolved_root / "data" / "processed"
        artifacts_dir = resolved_root / "artifacts"
        docs_dir = resolved_root / "docs"
        return cls(
            root=resolved_root,
            raw_data_path=resolved_root / "data" / "raw" / "customer_churn_synthetic.csv",
            scoring_input_path=resolved_root / "data" / "raw" / "scoring_input_sample.csv",
            processed_dir=processed_dir,
            artifacts_dir=artifacts_dir,
            docs_dir=docs_dir,
            model_path=artifacts_dir / "churn_model.pkl",
            metrics_path=artifacts_dir / "model_metrics.json",
            portfolio_kpis_path=artifacts_dir / "portfolio_kpis.json",
            leaderboard_path=artifacts_dir / "model_leaderboard.csv",
            feature_importance_path=artifacts_dir / "feature_importance.csv",
            segment_risk_path=artifacts_dir / "segment_risk.csv",
            retention_playbook_path=artifacts_dir / "retention_playbook.md",
            summary_markdown_path=artifacts_dir / "model_summary.md",
            executive_html_path=artifacts_dir / "executive_summary.html",
            scored_output_path=processed_dir / "customer_risk_scores.csv",
        )

    def ensure_directories(self) -> None:
        self.raw_data_path.parent.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.docs_dir.mkdir(parents=True, exist_ok=True)
