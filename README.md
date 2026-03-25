# customer-churn-analytics

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Template](https://img.shields.io/badge/Template-Data%20Science-teal)
![License](https://img.shields.io/badge/License-MIT-brightgreen)

Predictive churn modeling, KPI analysis, and stakeholder-ready insights.

## Overview

This repository is structured for exploratory analysis, feature engineering, and stakeholder-ready deliverables. It includes a starter module, notebook scaffold, Power BI placeholders, and a baseline churn modeling pipeline.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/analysis.py
```

## Run the churn pipeline

Place a CSV at `data/raw/customer_churn_sample.csv` and run:

```bash
python src/churn_pipeline.py
```

The pipeline writes:

- `data/processed/churn_model_metrics.csv`
- `data/processed/churn_scored_customers.csv`

## Expected churn dataset schema

The baseline model expects these columns:

- `customer_id`
- `tenure_months`
- `monthly_charges`
- `support_tickets_90d`
- `payment_failures_90d`
- `usage_score`
- `contract_type`
- `region`
- `churned`

Notes:

- `churned` must be binary (`0` or `1`).
- Both churn classes must be present for training and evaluation.
- Missing values in feature columns are handled by the preprocessing pipeline.

## Project Structure

- `src/analysis.py` contains a starter loading and summarization workflow for sample operational metrics.
- `src/churn_pipeline.py` trains a baseline logistic regression churn model and exports evaluation outputs.
- `notebooks/01_exploration.ipynb` is the initial analysis notebook.
- `docs/architecture.md` describes KPI and delivery expectations.
- `powerbi/` contains Power BI placeholders and metadata for stakeholder reporting.

## Validation and evaluation improvements

The churn pipeline includes:

- required-column validation before training
- stratified train/test splitting for more stable class balance
- exported metrics including accuracy, precision, recall, F1, ROC AUC, and average precision
- scored-customer output with probability and simple risk bands

## Automation Disclosure

**Note:** This repository uses automation and AI assistance for planning, initial scaffolding, routine maintenance, and selected code or documentation generation. I review and curate the outputs as part of my portfolio workflow.

## Reporting Starter Kit

This repository includes a practical Power BI build kit:

- `data/operations_kpis.csv` and `data/incident_log.csv` sample the reporting model
- `powerbi/measures.dax` contains reusable DAX starter measures
- `powerbi/semantic_model.json` describes the tables and fields
- `powerbi/dashboard_preview.html` gives a browser-based KPI preview before the real `.pbix` is built
