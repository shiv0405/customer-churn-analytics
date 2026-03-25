# Customer Churn Analytics

A production-friendly churn analytics project built for portfolio and stakeholder review. It includes synthetic enterprise-scale sample data, a training and evaluation pipeline, risk scoring outputs, model governance artifacts, and an executive-ready HTML summary.

## Highlights

- Large synthetic customer base with realistic churn drivers and segment mix
- End-to-end training pipeline with validation, model selection, threshold tuning, and scoring
- Exported artifacts for metrics, leaderboard comparison, feature importance, and executive reporting
- Modular package structure that is easy to evolve into batch jobs or API-serving workflows

## Project Layout

- `src/customer_churn_analytics/` contains the package code
- `data/raw/` stores the generated training dataset
- `data/processed/` stores scored customer outputs
- `artifacts/` stores metrics, model assets, and summary reports

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m customer_churn_analytics.cli run-all --rows 15000
```

## Commands

Generate only the sample dataset:

```bash
python -m customer_churn_analytics.cli generate-data --rows 15000
```

Train models, select the best one, and export reports:

```bash
python -m customer_churn_analytics.cli train
```

## Outputs

After `run-all`, the project writes:

- `data/raw/customer_churn_synthetic.csv`
- `data/processed/customer_risk_scores.csv`
- `artifacts/model_metrics.json`
- `artifacts/model_leaderboard.csv`
- `artifacts/feature_importance.csv`
- `artifacts/model_summary.md`
- `artifacts/executive_summary.html`
- `artifacts/churn_model.pkl`

## Notes

- The dataset is synthetic by design, but the feature interactions are intentionally realistic so the project supports senior-level discussion around retention strategy, operational risk, and model governance.
- This project is built to be transparent about automation and suitable for extension into orchestration, monitoring, or API-serving work.

## Automation Disclosure

**Note:** This repository uses automation and AI assistance for planning, initial scaffolding, routine maintenance, and selected code or documentation generation. I review and curate the outputs as part of my portfolio workflow.
