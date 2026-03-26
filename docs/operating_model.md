# Operating Model Notes

This project is intentionally framed around how a retention program would use model outputs in practice.

## Intervention Tiers

- `Immediate retention review`
  - high-probability churn accounts requiring specialist attention
- `Manager intervention`
  - accounts that should trigger billing, support, or account-management review
- `Automated nurture`
  - accounts appropriate for lifecycle messaging and adoption plays
- `Monitor only`
  - accounts kept in standard success cadence with watch-list monitoring

## Example Workflow

1. Generate or ingest the current customer portfolio
2. Train or refresh the champion model
3. Export scored accounts and portfolio KPIs
4. Route critical and elevated accounts into intervention queues
5. Review segment-level themes and refresh the playbook for the next planning cycle
