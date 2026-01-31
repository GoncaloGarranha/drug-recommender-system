from __future__ import annotations

import argparse
from pathlib import Path

from kg_constructor.src.pipeline import run_pipeline
from kg_constructor.src.utils.config import build_paths, load_config


def main():
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
