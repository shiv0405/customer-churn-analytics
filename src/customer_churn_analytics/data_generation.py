from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


REGIONS = np.array(["North America", "Europe", "APAC", "LATAM"])
CONTRACT_TYPES = np.array(["month-to-month", "one-year", "two-year"])
PAYMENT_METHODS = np.array(["credit-card", "bank-transfer", "digital-wallet", "invoice"])
INTERNET_TIERS = np.array(["fiber", "cable", "5g"])


def generate_customer_dataset(rows: int = 15000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    customer_ids = [f"CUST-{index:06d}" for index in range(1, rows + 1)]
    regions = rng.choice(REGIONS, size=rows, p=[0.36, 0.28, 0.24, 0.12])
    contracts = rng.choice(CONTRACT_TYPES, size=rows, p=[0.5, 0.28, 0.22])
    payment_methods = rng.choice(PAYMENT_METHODS, size=rows, p=[0.42, 0.27, 0.21, 0.10])
    internet_tiers = rng.choice(INTERNET_TIERS, size=rows, p=[0.52, 0.32, 0.16])

    ages = rng.integers(19, 78, size=rows)
    tenure_months = np.clip(rng.gamma(shape=2.8, scale=10.5, size=rows).round(), 1, 96).astype(int)
    monthly_charges = np.clip(rng.normal(loc=92, scale=24, size=rows), 25, 210).round(2)
    avg_monthly_usage_gb = np.clip(rng.normal(loc=410, scale=125, size=rows), 40, 980).round(2)

    autopay_enabled = rng.random(rows) < np.where(payment_methods == "invoice", 0.24, 0.72)
    paperless_billing = rng.random(rows) < 0.81
    premium_support = rng.random(rows) < np.where(contracts == "two-year", 0.38, 0.19)
    streaming_bundle = rng.random(rows) < np.where(internet_tiers == "fiber", 0.56, 0.31)

    support_tickets_90d = rng.poisson(lam=1.4 + (contracts == "month-to-month") * 0.6 + (~autopay_enabled) * 0.4)
    payment_failures_90d = rng.poisson(lam=0.35 + (~autopay_enabled) * 0.55 + (payment_methods == "invoice") * 0.35)
    service_incidents_30d = rng.poisson(lam=0.6 + (internet_tiers == "5g") * 0.22 + (regions == "LATAM") * 0.18)
    late_deliveries_90d = rng.poisson(lam=0.45 + (regions == "APAC") * 0.14 + (regions == "LATAM") * 0.20)

    discount_pct = np.clip(rng.normal(loc=9, scale=5, size=rows), 0, 30).round(2)
    engagement_score = np.clip(rng.normal(loc=69, scale=14, size=rows), 8, 99).round(1)
    nps_score = np.clip(rng.normal(loc=34, scale=28, size=rows), -80, 95).round(0).astype(int)
    days_since_last_login = np.clip(rng.normal(loc=8, scale=6, size=rows), 0, 45).round().astype(int)
    cross_sell_products = np.clip(rng.poisson(lam=1.4 + streaming_bundle * 0.7 + premium_support * 0.3), 0, 6)

    churn_logit = (
        -1.15
        + (contracts == "month-to-month") * 1.25
        + (contracts == "one-year") * 0.22
        + (~autopay_enabled) * 0.55
        + (payment_methods == "invoice") * 0.48
        + support_tickets_90d * 0.18
        + payment_failures_90d * 0.33
        + service_incidents_30d * 0.21
        + late_deliveries_90d * 0.12
        + (monthly_charges - 92) * 0.014
        - (avg_monthly_usage_gb - 410) * 0.0035
        + discount_pct * 0.018
        + days_since_last_login * 0.035
        - tenure_months * 0.022
        - engagement_score * 0.032
        - nps_score * 0.018
        - premium_support * 0.25
        - streaming_bundle * 0.14
        - cross_sell_products * 0.16
        + (regions == "LATAM") * 0.18
        + (regions == "APAC") * 0.08
    )

    churn_probability = 1 / (1 + np.exp(-(churn_logit * 1.75)))
    churned = (rng.random(rows) < churn_probability).astype(int)

    dataset = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "region": regions,
            "contract_type": contracts,
            "payment_method": payment_methods,
            "internet_tier": internet_tiers,
            "age": ages,
            "tenure_months": tenure_months,
            "monthly_charges": monthly_charges,
            "avg_monthly_usage_gb": avg_monthly_usage_gb,
            "autopay_enabled": autopay_enabled.astype(int),
            "paperless_billing": paperless_billing.astype(int),
            "premium_support": premium_support.astype(int),
            "streaming_bundle": streaming_bundle.astype(int),
            "support_tickets_90d": support_tickets_90d.astype(int),
            "payment_failures_90d": payment_failures_90d.astype(int),
            "service_incidents_30d": service_incidents_30d.astype(int),
            "late_deliveries_90d": late_deliveries_90d.astype(int),
            "discount_pct": discount_pct,
            "engagement_score": engagement_score,
            "nps_score": nps_score,
            "days_since_last_login": days_since_last_login,
            "cross_sell_products": cross_sell_products.astype(int),
            "churn_probability_signal": churn_probability.round(6),
            "churned": churned.astype(int),
        }
    )

    for column in ("avg_monthly_usage_gb", "discount_pct", "engagement_score"):
        missing_mask = rng.random(rows) < 0.025
        dataset.loc[missing_mask, column] = np.nan

    return dataset


def write_dataset(dataset: pd.DataFrame, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(destination, index=False)
    return destination
