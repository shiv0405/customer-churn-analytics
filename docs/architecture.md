# Architecture Overview

Customer Retention Intelligence Platform is structured as a portfolio-ready analytics accelerator with clear separation between data generation, model training, batch scoring, and stakeholder reporting.

## Layers

1. `data_generation.py`
   Creates the synthetic customer portfolio with retention, usage, support, and payment behavior signals.
2. `pipeline.py`
   Validates the dataset, selects a champion model, tunes the operating threshold, and exports performance and business-facing artifacts.
3. `inference.py`
   Loads the trained model and scores new batch files into intervention-ready outputs.
4. `artifacts/`
   Holds the report pack used for model review, business prioritization, and executive communication.

## Production Extension Path

- Replace synthetic inputs with warehouse or CRM extracts
- Persist models and outputs in a governed artifact store
- Schedule batch scoring on a fixed cadence
- Route high-risk accounts into CRM, lifecycle orchestration, or success-manager workflows
