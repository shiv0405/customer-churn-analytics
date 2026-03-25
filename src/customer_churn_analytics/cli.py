from __future__ import annotations

import argparse
from pathlib import Path

from .config import ProjectPaths
from .data_generation import generate_customer_dataset, write_dataset
from .pipeline import run_training_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Customer churn analytics project CLI")
    parser.add_argument("--project-root", default=".", help="Project root containing data/ and artifacts/")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate-data", help="Generate the synthetic customer churn dataset")
    generate_parser.add_argument("--rows", type=int, default=15000, help="Number of synthetic customer rows to generate")
    generate_parser.add_argument("--seed", type=int, default=42, help="Random seed")

    subparsers.add_parser("train", help="Train the churn model and export artifacts")

    run_all_parser = subparsers.add_parser("run-all", help="Generate data and run the full training pipeline")
    run_all_parser.add_argument("--rows", type=int, default=15000, help="Number of synthetic customer rows to generate")
    run_all_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    paths = ProjectPaths.from_root(project_root)

    if args.command == "generate-data":
        dataset = generate_customer_dataset(rows=args.rows, seed=args.seed)
        destination = write_dataset(dataset, paths.raw_data_path)
        print(f"Wrote dataset to {destination} ({len(dataset)} rows)")
        return 0

    if args.command == "train":
        result = run_training_pipeline(paths)
        print(f"Champion model: {result.model_name}")
        print(f"Threshold: {result.threshold}")
        print(f"Artifacts: {result.artifacts}")
        return 0

    if args.command == "run-all":
        dataset = generate_customer_dataset(rows=args.rows, seed=args.seed)
        write_dataset(dataset, paths.raw_data_path)
        result = run_training_pipeline(paths)
        print(f"Wrote dataset to {paths.raw_data_path} ({len(dataset)} rows)")
        print(f"Champion model: {result.model_name}")
        print(f"Threshold: {result.threshold}")
        print(f"Artifacts: {result.artifacts}")
        return 0

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
