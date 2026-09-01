import argparse

from src.experiments.config import ExperimentMode, TrainingConfig
from src.experiments.experiment import run_experiment
from src.experiments.query import query_index
from src.experiments.report import summarize_report
from src.generators.registry import GENERATORS

_DATASETS = ["adult", "diabetes", "heart"]
_GENERATORS = list(GENERATORS.keys())


def _parse_kwargs(kwarg_list: list[str] | None) -> dict:
    """Parse ['key=value', ...] into a dict, auto-casting ints and floats."""
    if not kwarg_list:
        return {}
    result = {}
    for item in kwarg_list:
        if "=" not in item:
            raise ValueError(f"--kwarg must be in KEY=VALUE format, got: {item!r}")
        key, raw = item.split("=", 1)
        for cast in (int, float):
            try:
                result[key] = cast(raw)
                break
            except ValueError:
                pass
        else:
            result[key] = raw
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Loss-aware synthetic data experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  Train with default kwargs:\n"
            "    python -m src.main run --dataset adult --generator ctgan\n\n"
            "  Train with custom kwargs (neural-network generators):\n"
            "    python -m src.main run --dataset adult --generator ctgan "
            "--kwarg epochs=50 --kwarg batch_size=500\n\n"
            "  Train with GaussianCopula (no epochs):\n"
            "    python -m src.main run --dataset heart --generator gaussian_copula\n\n"
            "  Re-evaluate a pre-trained generator without retraining:\n"
            "    python -m src.main evaluate --pretrained-run adult_ctgan_seed42 "
            "--dataset adult --generator ctgan\n\n"
            "  Print the summary of a saved report:\n"
            "    python -m src.main show adult_ctgan_seed42\n\n"
            "  Find runs matching given criteria:\n"
            "    python -m src.main find --dataset adult --generator ctgan --kwarg epochs=300"
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    _kwarg_help = (
        "Generator constructor kwarg as KEY=VALUE (repeatable). "
        "e.g. --kwarg epochs=50 --kwarg batch_size=500. "
        "Not needed for gaussian_copula (statistical model)."
    )

    run_p = sub.add_parser("run", help="Train a generator and evaluate it.")
    run_p.add_argument("--dataset", required=True, choices=_DATASETS)
    run_p.add_argument("--generator", required=True, choices=_GENERATORS)
    run_p.add_argument("--num-samples", type=int, default=1000)
    run_p.add_argument("--kwarg", action="append", metavar="KEY=VALUE", help=_kwarg_help)
    run_p.add_argument("--seed", type=int, default=42)
    run_p.add_argument("--test-size", type=float, default=0.2, help="fraction held out from training, used for evaluation")
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
    eval_p.add_argument("--test-size", type=float, default=0.2, help="must match the value used for the original training run")
    eval_p.add_argument("--output-dir", default="experiments/results")

    show_p = sub.add_parser("show", help="Print the summary of a saved experiment report.")
    show_p.add_argument("run_name", help="run_name of the saved report (e.g. adult_ctgan_seed42)")
    show_p.add_argument("--output-dir", default="experiments/results")

    find_p = sub.add_parser("find", help="Query index.jsonl for runs matching given criteria.")
    find_p.add_argument("--dataset", choices=_DATASETS)
    find_p.add_argument("--generator", choices=_GENERATORS)
    find_p.add_argument("--seed", type=int)
    find_p.add_argument("--test-size", type=float)
    find_p.add_argument("--code-version", help="short git commit hash, e.g. abcdef1")
    find_p.add_argument("--kwarg", action="append", metavar="KEY=VALUE",
                         help="match generator_kwargs entries (repeatable)")
    find_p.add_argument("--output-dir", default="experiments/results")

    args = parser.parse_args()

    if args.command == "run":
        config = TrainingConfig(
            dataset_name=args.dataset,
            generator_name=args.generator,
            num_samples=args.num_samples,
            seed=args.seed,
            test_size=args.test_size,
            generator_kwargs=_parse_kwargs(args.kwarg),
        )
        report = run_experiment(config, output_dir=args.output_dir)
        print(summarize_report(report))

    elif args.command == "evaluate":
        config = TrainingConfig(
            dataset_name=args.dataset,
            generator_name=args.generator,
            num_samples=args.num_samples,
            seed=args.seed,
            test_size=args.test_size,
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

    elif args.command == "find":
        matches = query_index(
            output_dir=args.output_dir,
            dataset_name=args.dataset,
            generator_name=args.generator,
            seed=args.seed,
            test_size=args.test_size,
            code_version=args.code_version,
            kwargs=_parse_kwargs(args.kwarg),
        )
        if not matches:
            print("No matching runs found.")
        else:
            for row in matches:
                print(
                    f"{row['run_name']}  "
                    f"(dataset={row.get('dataset_name')}, generator={row.get('generator_name')}, "
                    f"kwargs={row.get('generator_kwargs')}, code_version={row.get('code_version')})"
                )


if __name__ == "__main__":
    main()
