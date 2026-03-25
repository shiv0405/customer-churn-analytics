# Customer Churn Analytics Architecture

## Objective

This project starter supports two parallel outcomes:

1. Predict customer churn risk.
2. Translate model and KPI outputs into stakeholder-ready reporting.

## Recommended Data Flow

1. Land source extracts in `data/raw/`.
2. Validate schema and business definitions.
3. Train and score with `src/churn_pipeline.py`.
4. Save derived outputs to `data/processed/`.
5. Load processed outputs into Power BI for executive reporting.

## Core Analytical Entities

### Customer-level features

- `customer_id`: unique customer key
- `tenure_months`: how long the customer has been active
- `monthly_charges`: recurring billing amount
- `contract_type`: month-to-month, annual, or multi-year style segment
- `support_tickets_90d`: recent support burden
- `payment_failures_90d`: recent billing friction signal
- `usage_score`: normalized engagement indicator
- `region`: reporting segment

### Target

- `churned`: binary outcome where `1` means the customer churned

## KPI Definitions

Keep KPI logic consistent between Python outputs and BI dashboards.

- Churn rate = churned customers / active customer population in scope
- Predicted high-risk customers = count of customers above an agreed churn probability threshold
- Revenue at risk = sum of monthly charges for high-risk customers
- Retention opportunity rate = retained-after-intervention customers / targeted customers
- Support burden by risk tier = average support tickets grouped by predicted risk segment

## Modeling Notes

The baseline script uses logistic regression because it is:

- fast to run
- easy to explain
- suitable as a benchmark before more complex models

Suggested next experiments:

- tree-based models for nonlinear behavior
- threshold tuning based on retention team capacity
- calibration analysis for probability quality
- segment-specific models if business rules differ by product or region

## Stakeholder Deliverables

### Executive audience

Focus on:

- current churn trend
- revenue at risk
- high-risk segment breakdown
- top controllable drivers

### Operations or retention teams

Focus on:

- customer-level risk lists
- contact prioritization thresholds
- region or contract-type hotspots
- intervention tracking

## Output Contract

Current starter outputs:

- `data/processed/churn_model_metrics.csv`
- `data/processed/churn_scored_customers.csv`

Recommended future outputs:

- feature importance summary
- threshold scenario table
- monthly monitoring report
- model input data dictionary
