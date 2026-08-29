import argparse

from src.training.config import TrainingConfig
from src.training.runner import GENERATORS, run_experiment


def main():
    parser = argparse.ArgumentParser(description="Run a single generator/dataset experiment.")
    parser.add_argument("--dataset", required=True, choices=["adult", "diabetes", "heart"])
    parser.add_argument("--generator", required=True, choices=list(GENERATORS.keys()))
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=300, help="passed through as a generator kwarg")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = TrainingConfig(
        dataset_name=args.dataset,
        generator_name=args.generator,
        num_samples=args.num_samples,
        seed=args.seed,
        generator_kwargs={"epochs": args.epochs},
    )
    report = run_experiment(config)
    print(f"Saved results for '{config.run_name}' to experiments/results/")
    print(report)


if __name__ == "__main__":
    main()
