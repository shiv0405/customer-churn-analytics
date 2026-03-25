# customer-churn-analytics

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Template](https://img.shields.io/badge/Template-Data%20Science-teal)
![License](https://img.shields.io/badge/License-MIT-brightgreen)

Predictive churn modeling, KPI analysis, and stakeholder-ready insights.

## Overview

This repository is structured for exploratory analysis, feature engineering, and stakeholder-ready deliverables. It includes a starter module, notebook scaffold, and folders for raw and processed datasets.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/analysis.py
```

## Project Structure

- `src/analysis.py` contains a starter loading and summarization workflow.
- `notebooks/01_exploration.ipynb` is the initial analysis notebook.
- `data/raw/` and `data/processed/` separate incoming data from derived outputs.

## Automation Disclosure

**Note:** This repository uses automation and AI assistance for planning, initial scaffolding, routine maintenance, and selected code or documentation generation. I review and curate the outputs as part of my portfolio workflow.

## Added Starter Assets

These files make the project more immediately usable for churn analysis and stakeholder reporting:

- `src/churn_pipeline.py` - baseline churn modeling workflow with preprocessing, train/test split, and CSV metric export.
- `docs/architecture.md` - concise project architecture, KPI definitions, and stakeholder delivery guidance.
- `data/raw/customer_churn_sample.csv` - small synthetic sample dataset for local testing and demos.

## Run The Baseline Model

```bash
python src/churn_pipeline.py
```

Expected outputs:

- `data/processed/churn_model_metrics.csv`
- `data/processed/churn_scored_customers.csv`

## Suggested Next Steps

- Replace the sample CSV with production-safe source extracts.
- Add retention intervention labels for uplift or next-best-action analysis.
- Connect model outputs to Power BI for churn risk segmentation and KPI tracking.
