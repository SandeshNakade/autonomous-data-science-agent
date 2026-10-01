"""Command-line entrypoint used by local demos and optional DVC stages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .monitoring import (
    build_drift_baseline,
    compare_to_baseline,
    load_baseline,
    retraining_request,
    save_baseline,
)
from .orchestrator import AutonomousDataScienceOrchestrator
from .profiling import profile_dataset
from .registry import LocalModelRegistry
from .schemas import RunConfig


def _write_output(path: str | None, payload: dict) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    if path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(encoded, encoding="utf-8")
    print(encoded)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Controlled autonomous data-science prototype")
    commands = parser.add_subparsers(dest="command", required=True)

    profile = commands.add_parser("profile", help="Profile a CSV without training")
    profile.add_argument("--dataset", required=True)
    profile.add_argument("--output")

    run = commands.add_parser("run", help="Run a bounded agent-driven experiment")
    run.add_argument("--dataset", required=True)
    run.add_argument("--objective", required=True)
    run.add_argument("--target")
    run.add_argument("--threshold", type=float)
    run.add_argument("--max-experiments", type=int, default=3)
    run.add_argument("--use-mlflow", action="store_true")
    run.add_argument("--artifact-root", default="artifacts")
    run.add_argument("--model-root", default="models")
    run.add_argument("--output")

    approve = commands.add_parser("approve", help="Explicitly promote a pending model")
    approve.add_argument("--proposal", required=True)
    approve.add_argument("--approved-by", required=True)
    approve.add_argument("--model-root", default="models")
    approve.add_argument("--output")

    baseline = commands.add_parser("baseline", help="Save a compact drift baseline")
    baseline.add_argument("--dataset", required=True)
    baseline.add_argument("--exclude", nargs="*", default=[])
    baseline.add_argument("--output", required=True)

    monitor = commands.add_parser("monitor", help="Check drift and only request retraining")
    monitor.add_argument("--baseline", required=True)
    monitor.add_argument("--dataset", required=True)
    monitor.add_argument("--threshold", type=float, default=0.20)
    monitor.add_argument("--source-run-id")
    monitor.add_argument("--output")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "profile":
        _write_output(args.output, profile_dataset(args.dataset).to_dict())
        return
    if args.command == "run":
        summary = AutonomousDataScienceOrchestrator(
            artifact_root=args.artifact_root,
            model_root=args.model_root,
        ).run(
            args.dataset,
            args.objective,
            RunConfig(
                target_column=args.target,
                acceptance_threshold=args.threshold,
                max_experiments=args.max_experiments,
                use_mlflow=args.use_mlflow,
            ),
        )
        _write_output(args.output, summary.to_dict())
        return
    if args.command == "approve":
        result = LocalModelRegistry(args.model_root).approve(args.proposal, args.approved_by)
        _write_output(args.output, result)
        return
    if args.command == "baseline":
        baseline = build_drift_baseline(args.dataset, tuple(args.exclude))
        save_baseline(baseline, args.output)
        print(json.dumps(baseline.to_dict(), indent=2, sort_keys=True))
        return
    if args.command == "monitor":
        report = compare_to_baseline(load_baseline(args.baseline), args.dataset, args.threshold)
        _write_output(
            args.output,
            {"report": report.to_dict(), "retraining_request": retraining_request(report, args.source_run_id)},
        )


if __name__ == "__main__":
    main()
