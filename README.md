# Customer Retention Intelligence Platform

Executive-ready churn analytics project designed to look and behave like a retention decisioning accelerator rather than a notebook demo. It generates an enterprise-scale customer portfolio, trains and evaluates a champion churn model, scores accounts into intervention bands, and exports leadership-facing reporting artifacts that can support conversations around product adoption, billing risk, customer operations, and model governance.

## Why This Project Stands Out

- Positions the work around retention strategy, not just binary classification
- Produces a champion model, segment risk views, intervention queues, and stakeholder-ready HTML and markdown briefings
- Includes a reusable batch-scoring workflow that makes the project feel closer to production operations
- Uses synthetic data intentionally, but with realistic feature interactions that support senior-level discussion

## Core Capabilities

- Synthetic portfolio generation for 15,000+ customers with realistic commercial, support, product usage, and payment signals
- End-to-end model selection across multiple classifiers with threshold tuning and exported evaluation metrics
- Batch scoring output with risk bands, reason summaries, and recommended next actions
- Artifact pack for leadership review, including portfolio KPIs, retention playbook, segment risk summary, and executive HTML

## Project Layout

- `src/customer_churn_analytics/` contains the package code, training pipeline, and batch inference workflow
- `data/raw/` contains the generated training data and a ready-to-score sample batch file
- `data/processed/` contains scored outputs for operational review
- `artifacts/` contains the model, metrics, segment analysis, playbook, and executive summaries
- `docs/` contains architecture and operating-model notes

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m customer_churn_analytics.cli run-all --rows 15000
```

## Useful Commands

Generate the synthetic portfolio only:

```bash
python -m customer_churn_analytics.cli generate-data --rows 15000
```

Rebuild the full report pack from the current dataset:

```bash
python -m customer_churn_analytics.cli build-report
```

Score a new batch file with the trained model:

```bash
python -m customer_churn_analytics.cli score-batch --input data/raw/scoring_input_sample.csv
```

## Outputs

After `run-all`, the project writes:

- `data/raw/customer_churn_synthetic.csv`
- `data/raw/scoring_input_sample.csv`
- `data/processed/customer_risk_scores.csv`
- `artifacts/model_metrics.json`
- `artifacts/portfolio_kpis.json`
- `artifacts/model_leaderboard.csv`
- `artifacts/feature_importance.csv`
- `artifacts/segment_risk.csv`
- `artifacts/retention_playbook.md`
- `artifacts/model_summary.md`
- `artifacts/executive_summary.html`
- `artifacts/churn_model.pkl`

## Business Questions It Answers

- Which customer segments carry the highest renewal risk and why?
- How much of the portfolio needs immediate intervention versus automated nurture?
- Which operational signals matter most: billing friction, service reliability, engagement, or contract structure?
- How would this pipeline plug into a retention operations or customer success workflow?

## Production Path

- Replace the synthetic generator with warehouse extracts or CRM snapshots
- Publish the batch scoring output into downstream CRM or case-management workflows
- Add scheduled retraining, model monitoring, and champion/challenger governance
- Wrap the batch scorer in an API or job orchestration layer for daily operational use
