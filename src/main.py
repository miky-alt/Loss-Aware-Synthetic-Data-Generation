import argparse

from src.experiments.config import ExperimentMode, TrainingConfig
from src.experiments.experiment import run_experiment
from src.experiments.report import summarize_report
from src.generators.registry import GENERATORS

_DATASETS = ["adult", "diabetes", "heart"]
_GENERATORS = list(GENERATORS.keys())


def main():
    parser = argparse.ArgumentParser(
        description="Loss-aware synthetic data experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  Train and evaluate:\n"
            "    python -m src.main run --dataset adult --generator ctgan\n\n"
            "  Re-evaluate a pre-trained generator without retraining:\n"
            "    python -m src.main evaluate --pretrained-run adult_ctgan_seed42 "
            "--dataset adult --generator ctgan\n\n"
            "  Print the summary of a saved report:\n"
            "    python -m src.main show adult_ctgan_seed42"
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Train a generator and evaluate it.")
    run_p.add_argument("--dataset", required=True, choices=_DATASETS)
    run_p.add_argument("--generator", required=True, choices=_GENERATORS)
    run_p.add_argument("--num-samples", type=int, default=1000)
    run_p.add_argument("--epochs", type=int, default=300, help="passed through as a generator kwarg")
    run_p.add_argument("--seed", type=int, default=42)
    run_p.add_argument("--output-dir", default="experiments/results")

    eval_p = sub.add_parser(
        "evaluate",
        help="Load a pre-trained generator from disk and re-evaluate it (no retraining).",
    )
    eval_p.add_argument(
        "--pretrained-run",
        required=True,
        metavar="RUN_NAME",
        help="run_name of the saved .pkl to load (e.g. adult_ctgan_seed42)",
    )
    eval_p.add_argument("--dataset", required=True, choices=_DATASETS)
    eval_p.add_argument("--generator", required=True, choices=_GENERATORS)
    eval_p.add_argument("--num-samples", type=int, default=1000)
    eval_p.add_argument("--seed", type=int, default=42)
    eval_p.add_argument("--output-dir", default="experiments/results")

    show_p = sub.add_parser("show", help="Print the summary of a saved experiment report.")
    show_p.add_argument("run_name", help="run_name of the saved report (e.g. adult_ctgan_seed42)")
    show_p.add_argument("--output-dir", default="experiments/results")

    args = parser.parse_args()

    if args.command == "run":
        config = TrainingConfig(
            dataset_name=args.dataset,
            generator_name=args.generator,
            num_samples=args.num_samples,
            seed=args.seed,
            generator_kwargs={"epochs": args.epochs},
        )
        report = run_experiment(config, output_dir=args.output_dir)
        print(summarize_report(report))

    elif args.command == "evaluate":
        config = TrainingConfig(
            dataset_name=args.dataset,
            generator_name=args.generator,
            num_samples=args.num_samples,
            seed=args.seed,
        )
        report = run_experiment(
            config,
            output_dir=args.output_dir,
            mode=ExperimentMode.EVALUATE_ONLY,
            pretrained_run_name=args.pretrained_run,
        )
        print(summarize_report(report))

    elif args.command == "show":
        print(summarize_report(args.run_name, output_dir=args.output_dir))


if __name__ == "__main__":
    main()
