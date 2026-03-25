from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path

SITES = ("Berlin", "London", "New York")
TEAMS = ("Platform", "Customer Ops", "Revenue Ops")
SHIFTS = ("Day", "Swing")
ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    rows = []
    incidents = []
    start = datetime.now(timezone.utc).date() - timedelta(days=77)
    for period in range(12):
        week = start + timedelta(days=period * 7)
        for site_index, site in enumerate(SITES):
            for team_index, team in enumerate(TEAMS):
                for shift_index, shift in enumerate(SHIFTS):
                    target = 420 + site_index * 32 + team_index * 24 + shift_index * 16
                    row = {
                        "reporting_week": week.isoformat(),
                        "site": site,
                        "team": team,
                        "shift": shift,
                        "throughput_units": target - 24 + ((period + site_index + team_index + shift_index) % 7) * 8,
                        "throughput_target": target,
                        "downtime_minutes": 34 + site_index * 6 + team_index * 4 + shift_index * 8 + (period % 4) * 5,
                        "quality_pct": round(95.7 + ((period + team_index + shift_index) % 6) * 0.45 - site_index * 0.15, 2),
                        "on_time_pct": round(91.8 + ((period + site_index + shift_index) % 5) * 0.6 - team_index * 0.1, 2),
                        "backlog_items": 42 + site_index * 5 + team_index * 6 + shift_index * 4 + ((12 - period) % 5) * 3,
                        "escalations_open": 3 + site_index + ((period + team_index + shift_index) % 4),
                    }
                    rows.append(row)
                    if row["downtime_minutes"] >= 50 or (len(rows) % 3 == 0):
                        incidents.append({
                            "incident_id": f"OPS-{datetime.now(timezone.utc).year}-{len(incidents) + 1:03d}",
                            "opened_date": row["reporting_week"],
                            "site": row["site"],
                            "team": row["team"],
                            "severity": "high" if row["downtime_minutes"] >= 65 else "medium",
                            "status": "closed" if len(incidents) % 2 == 0 else "monitoring",
                            "resolution_hours": 5 + (len(incidents) % 7) * 2,
                            "owner": f"{row['team']} lead",
                        })
    destination = ROOT / "data" / "operations_kpis.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    incident_destination = ROOT / "data" / "incident_log.csv"
    with incident_destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(incidents[0].keys()))
        writer.writeheader()
        writer.writerows(incidents)


if __name__ == "__main__":
    main()
