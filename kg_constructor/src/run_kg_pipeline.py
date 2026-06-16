"""
Main entry point for the Knowledge Graph construction pipeline.

This script parses CLI arguments, loads configuration, resolves paths,
initializes reproducibility logging, and invokes the pipeline runner.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from kg_constructor.src.pipeline import run_pipeline
from kg_constructor.src.utils.config import build_paths, load_config
from kg_constructor.src.utils.log import add_file_handler, get_logger
from kg_constructor.src.utils.reproducibility import make_run_id, write_run_summary

logger = get_logger(__name__)


def resolve_selected_steps(
    pipeline_order: list[str],
    only: str | None,
    from_step: str | None,
    to_step: str | None,
) -> list[str]:
    """
    Resolve the effective list of steps selected by the CLI arguments.
    """
    if only:
        return [only]

    start_index = 0
    end_index = len(pipeline_order)

    if from_step:
        start_index = pipeline_order.index(from_step)

    if to_step:
        end_index = pipeline_order.index(to_step) + 1

    return pipeline_order[start_index:end_index]


def main() -> None:
    """
    Parse command-line arguments, load config, initialize reproducibility
    artefacts, and run the pipeline.
    """
    parser = argparse.ArgumentParser(description="Knowledge Graph construction pipeline")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--only", default=None, help="Run a single pipeline step (e.g., 01_stats)")
    parser.add_argument("--from_step", default=None, help="Run pipeline starting from this step")
    parser.add_argument("--to_step", default=None, help="Run pipeline until this step")
    parser.add_argument("--force", action="store_true", help="Force execution even if outputs exist")

    args = parser.parse_args()

    cfg = load_config(args.config)
    project_root = Path(".").resolve()
    paths = build_paths(cfg, project_root=project_root)

    pipeline_order = cfg["pipeline"]["steps"]
    selected_steps = resolve_selected_steps(
        pipeline_order=pipeline_order,
        only=args.only,
        from_step=args.from_step,
        to_step=args.to_step,
    )

    run_id = make_run_id()
    runs_dir = Path(paths.output_dir) / "reports" / "runs"
    log_path = runs_dir / f"run_{run_id}.log"
    summary_path = runs_dir / f"run_{run_id}_summary.json"

    add_file_handler(log_path)

    write_run_summary(
        summary_path=summary_path,
        run_id=run_id,
        config_path=Path(args.config),
        cfg=cfg,
        input_dir=Path(paths.input_dir),
        selected_steps=selected_steps,
        force=args.force,
    )

    logger.info("Run ID: %s", run_id)
    logger.info("Run log: %s", log_path)
    logger.info("Run summary: %s", summary_path)

    run_pipeline(
        cfg=cfg,
        paths=paths,
        pipeline_order=pipeline_order,
        only=args.only,
        from_step=args.from_step,
        to_step=args.to_step,
        force=args.force,
    )


if __name__ == "__main__":
    main()
    