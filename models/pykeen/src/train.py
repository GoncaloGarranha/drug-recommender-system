from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pykeen
import torch
import yaml
from pykeen.models import ComplEx, DistMult, RGCN, TransE
from pykeen.losses import BCEWithLogitsLoss, MarginRankingLoss, SoftplusLoss, SoftMarginRankingLoss
from pykeen.regularizers import NoRegularizer
from pykeen.training import SLCWATrainingLoop
from pykeen.utils import set_random_seed
from pykeen.sampling import (
    BasicNegativeSampler,
    PseudoTypedNegativeSampler,
)

from .data import load_publication_factories
from .evaluation import evaluate_full_ranking


REPO_ROOT = Path(__file__).resolve().parents[3]


MODEL_CLASSES = {
    "distmult": DistMult,
    "transe": TransE,
    "complex": ComplEx,
    "rgcn": RGCN,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
        help="PyKEEN experiment YAML.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing run directory.",
    )

    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError("Config must contain a YAML mapping.")

    return config


def git_value(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=REPO_ROOT,
            text=True,
        ).strip()
    except Exception:
        return None


def cpu_state_dict(model) -> dict[str, Any]:
    state: dict[str, Any] = {}

    for key, value in model.state_dict().items():
        if torch.is_tensor(value):
            state[key] = value.detach().cpu().clone()
        else:
            state[key] = copy.deepcopy(value)

    return state


def save_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def build_model(
    model_cfg: dict[str, Any],
    training_factory,
    seed: int,
):
    model_name = str(model_cfg["name"])
    model_key = model_name.lower()

    if model_key not in MODEL_CLASSES:
        raise ValueError(
            f"Unsupported PyKEEN model: {model_name}. "
            f"Available: {sorted(MODEL_CLASSES)}"
        )

    model_class = MODEL_CLASSES[model_key]

    loss_name = str(
        model_cfg.get("loss", "margin_ranking")
    ).lower()

    loss_classes = {
        "margin_ranking": MarginRankingLoss,
        "bce": BCEWithLogitsLoss,
        "softplus": SoftplusLoss,
        "soft_margin_ranking": SoftMarginRankingLoss,
    }

    if loss_name not in loss_classes:
        raise ValueError(
            f"Unsupported loss: {loss_name}"
        )

    kwargs: dict[str, Any] = {
        "triples_factory": training_factory,
        "embedding_dim": int(
            model_cfg.get("embedding_dim", 128)
        ),
        "random_seed": seed,
        "loss": loss_classes[loss_name],
    }

    if bool(
        model_cfg.get(
            "disable_regularizer",
            False,
        )
    ):
        kwargs["regularizer"] = NoRegularizer

    if model_key == "rgcn":
        kwargs.update(
            {
                "num_layers": int(
                    model_cfg.get(
                        "num_layers",
                        2,
                    )
                ),
                "interaction": str(
                    model_cfg.get(
                        "interaction",
                        "distmult",
                    )
                ),
                "use_bias": bool(
                    model_cfg.get(
                        "use_bias",
                        True,
                    )
                ),
                "edge_dropout": float(
                    model_cfg.get(
                        "edge_dropout",
                        0.4,
                    )
                ),
                "self_loop_dropout": float(
                    model_cfg.get(
                        "self_loop_dropout",
                        0.2,
                    )
                ),
            }
        )

        activation = model_cfg.get("activation")
        if activation is not None:
            kwargs["activation"] = activation

        decomposition = model_cfg.get("decomposition")
        if decomposition is not None:
            kwargs["decomposition"] = decomposition

            decomposition_kwargs = model_cfg.get(
                "decomposition_kwargs"
            )

            if decomposition_kwargs is not None:
                kwargs["decomposition_kwargs"] = (
                    decomposition_kwargs
                )

    return model_class(**kwargs)


def main() -> None:
    args = parse_args()

    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)

    seed = int(cfg.get("seed", 42))
    set_random_seed(seed)

    model_cfg = cfg["model"]
    training_cfg = cfg["training"]
    runs_cfg = cfg["runs"]

    model_name = str(model_cfg["name"])
    model_key = model_name.lower()

    experiment_name = str(
        runs_cfg["experiment_name"]
    )

    run_dir = (
        REPO_ROOT
        / str(runs_cfg["out_dir"])
        / model_key
        / experiment_name
        / f"seed_{seed}"
    )

    if run_dir.exists():
        if not args.force:
            raise SystemExit(
                f"Run directory already exists: {run_dir}\n"
                "Use --force only if you intentionally want "
                "to overwrite it."
            )

        shutil.rmtree(run_dir)

    checkpoints_dir = run_dir / "checkpoints"

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoints_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        config_path,
        run_dir / "config.yaml",
    )

    data = load_publication_factories()

    training = data["training"]

    print("=" * 72)
    print("PyKEEN benchmark training")
    print("=" * 72)
    print(f"Model:              {model_name}")
    print(f"Seed:               {seed}")
    print(f"Training triples:   {training.num_triples}")
    print(f"Entities:           {training.num_entities}")
    print(f"Relations:          {training.num_relations}")
    print(
        f"Validation triples: "
        f"{data['validation'].num_triples}"
    )
    print(
        f"Drug candidates:    "
        f"{len(data['candidate_drugs'])}"
    )
    print(f"Run:                {run_dir}")
    print("=" * 72)

    model = build_model(
        model_cfg=model_cfg,
        training_factory=training,
        seed=seed,
    )

    optimizer_name = str(
        training_cfg.get("optimizer", "Adam")
    )

    training_loop = SLCWATrainingLoop(
        model=model,
        triples_factory=training,
        optimizer=optimizer_name,
        optimizer_kwargs={
            "lr": float(
                training_cfg.get("lr", 1e-4)
            ),
            "weight_decay": float(
                training_cfg.get(
                    "weight_decay",
                    0.0,
                )
            ),
        },
        negative_sampler=str(
            training_cfg.get(
                "negative_sampler",
                "basic",
            )
        ),
        negative_sampler_kwargs={
            "num_negs_per_pos": int(
                training_cfg.get(
                    "num_negs_per_pos",
                    1,
                )
            ),
            "filtered": bool(
                training_cfg.get(
                    "filtered_negative_sampling",
                    False,
                )
            ),
        },
    )

    max_epochs = int(
        training_cfg["max_epochs"]
    )

    eval_every = int(
        training_cfg["eval_every"]
    )

    patience_evals = int(
        training_cfg.get(
            "patience_evals",
            0,
        )
    )

    batch_size = int(
        training_cfg.get(
            "batch_size",
            1024,
        )
    )

    pin_memory = bool(
        training_cfg.get(
            "pin_memory",
            False,
        )
    )

    if max_epochs <= 0:
        raise ValueError(
            "max_epochs must be positive."
        )

    if eval_every <= 0:
        raise ValueError(
            "eval_every must be positive."
        )

    best_mrr = float("-inf")
    best_epoch = -1
    best_metrics: dict[str, float] | None = None

    evaluations_without_improvement = 0

    training_log: list[dict[str, Any]] = []

    current_epoch = 0
    first_training_call = True

    while current_epoch < max_epochs:
        target_epoch = min(
            current_epoch + eval_every,
            max_epochs,
        )

        losses = training_loop.train(
            triples_factory=training,
            num_epochs=target_epoch,
            batch_size=batch_size,
            continue_training=not first_training_call,
            use_tqdm=True,
            use_tqdm_batch=False,
            pin_memory=pin_memory,
        )

        first_training_call = False
        current_epoch = target_epoch

        model.eval()

        valid_metrics = evaluate_full_ranking(
            model=model,
            data=data,
            split="valid",
        )

        current_mrr = float(
            valid_metrics["mrr"]
        )

        current_loss = (
            float(losses[-1])
            if losses
            else None
        )

        entry = {
            "epoch": current_epoch,
            "loss": current_loss,
            "valid_mrr": current_mrr,
            "valid_hits@1": float(
                valid_metrics["hits@1"]
            ),
            "valid_hits@3": float(
                valid_metrics["hits@3"]
            ),
            "valid_hits@10": float(
                valid_metrics["hits@10"]
            ),
        }

        training_log.append(entry)

        print(
            f"Epoch {current_epoch:04d} | "
            f"Loss: "
            f"{current_loss if current_loss is not None else 'n/a'} | "
            f"Valid MRR: {current_mrr:.4f}"
        )

        if current_mrr > best_mrr:
            best_mrr = current_mrr
            best_epoch = current_epoch

            best_metrics = {
                key: float(value)
                if isinstance(
                    value,
                    (float, int),
                )
                else value
                for key, value
                in valid_metrics.items()
            }

            torch.save(
                {
                    "model_name": model_name,
                    "model_state_dict": (
                        cpu_state_dict(model)
                    ),
                    "config": cfg,
                    "best_epoch": best_epoch,
                    "best_valid_mrr": best_mrr,
                    "best_valid_metrics": (
                        best_metrics
                    ),
                },
                checkpoints_dir / "best.pt",
            )

            save_json(
                run_dir / "best_metrics.json",
                {
                    "best_epoch": best_epoch,
                    "best_valid_mrr": best_mrr,
                    **{
                        f"best_valid_{key}": value
                        for key, value
                        in best_metrics.items()
                        if key != "mrr"
                    },
                },
            )

            evaluations_without_improvement = 0

        else:
            evaluations_without_improvement += 1

        save_json(
            run_dir / "training_log.json",
            training_log,
        )

        if (
            patience_evals > 0
            and evaluations_without_improvement
            >= patience_evals
        ):
            print(
                "Early stopping: no validation MRR "
                f"improvement for {patience_evals} "
                "evaluation steps."
            )
            break

    torch.save(
        {
            "model_name": model_name,
            "model_state_dict": (
                cpu_state_dict(model)
            ),
            "config": cfg,
            "final_epoch": current_epoch,
            "best_epoch": best_epoch,
            "best_valid_mrr": best_mrr,
        },
        checkpoints_dir / "last.pt",
    )

    metadata = {
        "model": model_name,
        "seed": seed,
        "pykeen_version": pykeen.get_version(),
        "torch_version": torch.__version__,
        "git_commit": git_value(
            "rev-parse",
            "HEAD",
        ),
        "git_branch": git_value(
            "branch",
            "--show-current",
        ),
        "training_triples": (
            training.num_triples
        ),
        "validation_triples": (
            data["validation"].num_triples
        ),
        "test_triples": (
            data["testing"].num_triples
        ),
        "candidate_pool_size": len(
            data["candidate_drugs"]
        ),
        "loss": (
            model.loss.__class__.__name__
        ),
        "best_epoch": best_epoch,
        "best_valid_mrr": best_mrr,
        "final_epoch": current_epoch,
    }

    save_json(
        run_dir / "run_metadata.json",
        metadata,
    )

    print()
    print("=" * 72)
    print("Training complete")
    print(f"Best epoch:          {best_epoch}")
    print(
        f"Best validation MRR: {best_mrr:.4f}"
    )
    print(
        f"Loss:                "
        f"{model.loss.__class__.__name__}"
    )
    print(
        f"Best checkpoint:     "
        f"{checkpoints_dir / 'best.pt'}"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
