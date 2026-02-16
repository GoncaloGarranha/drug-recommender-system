"""
KG Pipeline Orchestration Module

This module provides dynamic loading and execution of pipeline steps for the Knowledge Graph constructor.
Each step is a Python script in src/steps/ with a `run(cfg, paths, force=False)` function.
The pipeline can execute all steps or a selected subset, supporting flexible workflows.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Callable, List, Optional

from kg_constructor.src.utils.log import get_logger

logger = get_logger(__name__)


def _load_step_run(steps_dir: Path, step_name: str) -> Callable[..., None]:
    """
    Dynamically import a pipeline step script and return its `run` function.
    Args:
        steps_dir: Path to the steps directory (src/steps/).
        step_name: Name of the step (e.g., '01_stats').
    Returns:
        Callable run function for the step.
    Raises:
        FileNotFoundError: If the step file does not exist.
        ImportError: If the module cannot be loaded.
        AttributeError: If the module does not define a callable `run`.
    """
    step_path = steps_dir / f"{step_name}.py"
    if not step_path.exists():
        raise FileNotFoundError(f"Step file not found: {step_path}")

    module_name = f"kg_step_{step_name}"
    spec = importlib.util.spec_from_file_location(module_name, str(step_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec for: {step_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore

    run_fn = getattr(module, "run", None)
    if run_fn is None or not callable(run_fn):
        raise AttributeError(
            f"Step '{step_name}' must define a callable: run(cfg, paths, force=False)"
        )

    return run_fn


def _select_steps(
    pipeline_order: List[str],
    only: Optional[str] = None,
    from_step: Optional[str] = None,
    to_step: Optional[str] = None,
) -> List[str]:
    """
    Select which steps to run based on CLI flags.
    Args:
        pipeline_order: List of all step names in order.
        only: If set, run only this step.
        from_step: If set, start from this step (inclusive).
        to_step: If set, end at this step (inclusive).
    Returns:
        List of step names to execute.
    Raises:
        ValueError: If any step name is invalid or range is inconsistent.
    """
    if only:
        if only not in pipeline_order:
            raise ValueError(f"Invalid --only step: {only}")
        return [only]

    start = 0
    end = len(pipeline_order) - 1

    if from_step:
        if from_step not in pipeline_order:
            raise ValueError(f"Invalid --from_step: {from_step}")
        start = pipeline_order.index(from_step)

    if to_step:
        if to_step not in pipeline_order:
            raise ValueError(f"Invalid --to_step: {to_step}")
        end = pipeline_order.index(to_step)

    if start > end:
        raise ValueError("Invalid step range: --from_step is after --to_step.")

    return pipeline_order[start : end + 1]


def run_pipeline(
    cfg: dict,
    paths,
    pipeline_order: List[str],
    only: Optional[str] = None,
    from_step: Optional[str] = None,
    to_step: Optional[str] = None,
    force: bool = False,
) -> None:
    """
    Run the Knowledge Graph pipeline by dynamically loading and executing each step.
    Args:
        cfg: Configuration dictionary (from YAML or CLI).
        paths: Object with resolved paths for input/output directories.
        pipeline_order: List of all step names in execution order.
        only: If set, run only this step.
        from_step: If set, start from this step (inclusive).
        to_step: If set, end at this step (inclusive).
        force: If True, force re-execution of steps even if outputs exist.
    """
    # Resolve the steps directory (src/steps/)
    steps_dir = Path(paths.kg_root) / "src" / "steps"
    if not steps_dir.exists():
        raise FileNotFoundError(f"Steps directory not found: {steps_dir}")

    # Determine which steps to run based on CLI flags
    selected = _select_steps(pipeline_order, only=only, from_step=from_step, to_step=to_step)

    logger.info("Starting KG pipeline")
    logger.info(f"Selected steps: {selected}")
    logger.info(f"Force execution: {force}")

    # Execute each selected step in order
    for step_name in selected:
        logger.info(f"--> Running step: {step_name}")
        run_fn = _load_step_run(steps_dir, step_name)
        # Each step must define a function: run(cfg, paths, force=False)
        run_fn(cfg=cfg, paths=paths, force=force)
        logger.info(f"<-- Finished step: {step_name}")

    logger.info("KG pipeline completed successfully")
