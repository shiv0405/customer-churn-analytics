"""Customer churn analytics package."""

from .config import ProjectPaths
from .data_generation import generate_customer_dataset, write_dataset
from .pipeline import run_training_pipeline

__all__ = [
    "ProjectPaths",
    "generate_customer_dataset",
    "write_dataset",
    "run_training_pipeline",
]
