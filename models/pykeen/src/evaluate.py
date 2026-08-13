from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from .data import load_publication_factories
from .evaluation import (
    evaluate_fixed_1v50,
    evaluate_full_ranking,
)
from .train import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run_dir",
        required=True,
        help="PyKEEN run directory containing checkpoints/best.pt",
    )

    parser.add_argument(
        "--split",
        choices=("valid", "test"),
        default="test",
    )

    parser.add_argument(
        "--protocol",
        choices=("full", "1v50", "both"),
        default="both",
    )

    return parser.parse_args()


def save_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()

    run_dir = Path(args.run_dir)
    checkpoint_path = (
        run_dir / "checkpoints" / "best.pt"
    )

    if not checkpoint_path.is_file():
        raise SystemExit(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    cfg = checkpoint["config"]
    seed = int(cfg.get("seed", 42))

    data_cfg = cfg.get("data", {})
    processed_dir = data_cfg.get("processed_dir")

    data = load_publication_factories(
        processed_dir=processed_dir,
    )

    model = build_model(
        model_cfg=cfg["model"],
        training_factory=data["training"],
        seed=seed,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    output_dir = run_dir / "evaluation"
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print("PyKEEN benchmark evaluation")
    print("=" * 72)
    print(
        f"Model:       "
        f"{checkpoint['model_name']}"
    )
    print(
        f"Best epoch:  "
        f"{checkpoint['best_epoch']}"
    )
    print(
        f"Best val MRR:"
        f" {checkpoint['best_valid_mrr']:.6f}"
    )
    print(f"Split:       {args.split}")
    print(f"Protocol:    {args.protocol}")
    print("=" * 72)

    if args.protocol in {"full", "both"}:
        full_metrics = evaluate_full_ranking(
            model=model,
            data=data,
            split=args.split,
        )

        payload = {
            "model": checkpoint["model_name"],
            "split": args.split,
            "protocol": "full_filtered",
            "best_epoch": int(
                checkpoint["best_epoch"]
            ),
            "best_valid_mrr": float(
                checkpoint["best_valid_mrr"]
            ),
            **full_metrics,
        }

        output_path = (
            output_dir
            / f"metrics_{args.split}_full.json"
        )

        save_json(
            output_path,
            payload,
        )

        print()
        print("--- FULL FILTERED RANKING ---")
        print(
            f"MRR:       "
            f"{full_metrics['mrr']:.4f}"
        )
        print(
            f"Hits@1:    "
            f"{full_metrics['hits@1']:.4f}"
        )
        print(
            f"Hits@3:    "
            f"{full_metrics['hits@3']:.4f}"
        )
        print(
            f"Hits@10:   "
            f"{full_metrics['hits@10']:.4f}"
        )
        print(
            f"Examples:  "
            f"{full_metrics['num_examples']}"
        )
        print(
            f"Candidates:"
            f" {full_metrics['candidate_pool_size']}"
        )
        print(f"Saved:     {output_path}")

    if args.protocol in {"1v50", "both"}:
        if args.split not in {"valid", "test"}:
            raise SystemExit(
                "1v50 supports only valid/test."
            )

        fixed_metrics = evaluate_fixed_1v50(
            model=model,
            data=data,
            split=args.split,
        )

        payload = {
            "model": checkpoint["model_name"],
            "split": args.split,
            "protocol": "fixed_1v50",
            "best_epoch": int(
                checkpoint["best_epoch"]
            ),
            "best_valid_mrr": float(
                checkpoint["best_valid_mrr"]
            ),
            **fixed_metrics,
        }

        output_path = (
            output_dir
            / f"metrics_{args.split}_1v50.json"
        )

        save_json(
            output_path,
            payload,
        )

        print()
        print("--- FIXED 1-vs-50 ---")
        print(
            f"MRR:       "
            f"{fixed_metrics['mrr']:.4f}"
        )
        print(
            f"Hits@1:    "
            f"{fixed_metrics['hits@1']:.4f}"
        )
        print(
            f"Hits@3:    "
            f"{fixed_metrics['hits@3']:.4f}"
        )
        print(
            f"Hits@10:   "
            f"{fixed_metrics['hits@10']:.4f}"
        )
        print(
            f"AUC-ROC:   "
            f"{fixed_metrics['auc_roc']:.4f}"
        )
        print(
            f"Avg. Prec.:"
            f" {fixed_metrics['average_precision']:.4f}"
        )
        print(
            f"Examples:  "
            f"{fixed_metrics['num_examples']}"
        )
        print(f"Saved:     {output_path}")


if __name__ == "__main__":
    main()
