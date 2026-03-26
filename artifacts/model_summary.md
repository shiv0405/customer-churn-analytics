# Retention Intelligence Summary

- Champion model: `logistic_regression`
- Operating threshold: `0.75`

## Model Performance

- `accuracy`: 0.9193
- `precision`: 0.3095
- `recall`: 0.6894
- `f1`: 0.4272
- `roc_auc`: 0.9397
- `average_precision`: 0.4845

## Portfolio KPIs

- Customers scored: `15000`
- Critical account share: `9.7%`
- Elevated or critical share: `17.7%`
- Average churn probability: `24.9%`

## Top Features

- `boolean__autopay_enabled`: 1.150019
- `categorical__contract_type_month-to-month`: 1.13603
- `categorical__contract_type_two-year`: 1.001268
- `numeric__nps_score`: 0.84063
- `numeric__engagement_score`: 0.779412
- `numeric__avg_monthly_usage_gb`: 0.723211
- `numeric__monthly_charges`: 0.680105
- `categorical__contract_type_one-year`: 0.663158
- `numeric__tenure_months`: 0.585899
- `categorical__payment_method_invoice`: 0.48566

## Highest-Risk Segments

- `LATAM / month-to-month`: probability 0.4197, critical accounts 182
- `APAC / month-to-month`: probability 0.3675, critical accounts 311
- `North America / month-to-month`: probability 0.3587, critical accounts 438
- `Europe / month-to-month`: probability 0.3579, critical accounts 334
- `LATAM / one-year`: probability 0.179, critical accounts 15
- `Europe / one-year`: probability 0.1499, critical accounts 29
- `APAC / one-year`: probability 0.1473, critical accounts 35
- `North America / one-year`: probability 0.1356, critical accounts 39
- `LATAM / two-year`: probability 0.1308, critical accounts 11
- `Europe / two-year`: probability 0.112, critical accounts 19