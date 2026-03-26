# Retention Playbook

## Portfolio Snapshot

- Customers scored: `15000`
- Critical share: `9.7%`
- Elevated or critical share: `17.7%`
- Average churn probability: `24.9%`

## Priority Segments

- `LATAM / month-to-month` with average churn probability `0.4197` and `182` critical accounts
- `APAC / month-to-month` with average churn probability `0.3675` and `311` critical accounts
- `North America / month-to-month` with average churn probability `0.3587` and `438` critical accounts
- `Europe / month-to-month` with average churn probability `0.3579` and `334` critical accounts
- `LATAM / one-year` with average churn probability `0.179` and `15` critical accounts

## Primary Risk Drivers

- `boolean__autopay_enabled` (1.150019)
- `categorical__contract_type_month-to-month` (1.13603)
- `categorical__contract_type_two-year` (1.001268)
- `numeric__nps_score` (0.84063)
- `numeric__engagement_score` (0.779412)
- `numeric__avg_monthly_usage_gb` (0.723211)
- `numeric__monthly_charges` (0.680105)
- `categorical__contract_type_one-year` (0.663158)

## Example Intervention Queue

- `CUST-013331`: Immediate retention review | Month-to-month commitment; High support friction; Weak customer advocacy | Route to retention specialist with offer guardrails and executive outreach review.
- `CUST-000348`: Immediate retention review | Month-to-month commitment; Billing instability; Weak customer advocacy | Route to retention specialist with offer guardrails and executive outreach review.
- `CUST-000235`: Immediate retention review | Month-to-month commitment; Billing instability; High support friction | Route to retention specialist with offer guardrails and executive outreach review.
- `CUST-003220`: Immediate retention review | Month-to-month commitment; Billing instability; High support friction | Route to retention specialist with offer guardrails and executive outreach review.
- `CUST-013885`: Immediate retention review | Month-to-month commitment; Billing instability; Weak customer advocacy | Route to retention specialist with offer guardrails and executive outreach review.
- `CUST-007062`: Immediate retention review | Month-to-month commitment; Billing instability; High support friction | Route to retention specialist with offer guardrails and executive outreach review.
- `CUST-003146`: Immediate retention review | Month-to-month commitment; Billing instability; Recent service disruption | Route to retention specialist with offer guardrails and executive outreach review.
- `CUST-013766`: Immediate retention review | Month-to-month commitment; High support friction; Falling product engagement | Route to retention specialist with offer guardrails and executive outreach review.