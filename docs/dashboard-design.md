# customer-churn-analytics Dashboard Design

Predictive churn modeling, KPI analysis, and stakeholder-ready insights.

## Pages
- Executive Overview with KPI cards and weekly trend
- Site Performance with throughput, quality, and backlog comparison
- Incident Detail with severity and resolution time views

## Modeling Notes
- Treat `operations_kpis.csv` as the weekly fact table
- Treat `incident_log.csv` as the operational incident table
- Add a date dimension linked to `reporting_week` in the real Power BI model
