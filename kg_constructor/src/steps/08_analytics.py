from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import dgl
import matplotlib.pyplot as plt
import numpy as np

from kg_constructor.src.utils.log import get_logger

logger = get_logger(__name__)


def run(cfg: Dict[str, Any], paths, force: bool = False) -> None:
    """
    Step 08_analytics: generate basic analytics over the training graph.

    Inputs:
      - data/processed/train_graph.bin
      - data/processed/graph_metadata.json
      - data/processed/entity2id.json
      - data/processed/rel2id.json

    Outputs:
      - data/output/plots/*.png
      - data/output/reports/analytics_report.json
    """
    output_dir = Path(paths.output_dir)
    processed_dir = Path(paths.processed_dir)

    plots_dir = output_dir / "plots"
    reports_dir = output_dir / "reports"
    plots_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    graph_path = processed_dir / "train_graph.bin"
    meta_path = processed_dir / "graph_metadata.json"
    entity2id_path = processed_dir / "entity2id.json"
    rel2id_path = processed_dir / "rel2id.json"

    report_path = reports_dir / "analytics_report.json"

    # Plot paths
    rel_dist_path = plots_dir / "relation_distribution.png"
    degree_log_path = plots_dir / "degree_distribution_log.png"
    top_nodes_path = plots_dir / "top_nodes_by_degree.png"

    if not force and report_path.exists() and rel_dist_path.exists() and degree_log_path.exists() and top_nodes_path.exists():
        logger.info("Analytics outputs already exist. Skipping (use --force to re-run).")
        return

    if not graph_path.exists():
        raise FileNotFoundError("Missing train_graph.bin. Run 07_index_dgl first.")

    g_list, _ = dgl.load_graphs(str(graph_path))
    g = g_list[0]

    metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    entity2id = json.loads(entity2id_path.read_text(encoding="utf-8")) if entity2id_path.exists() else {}
    rel2id = json.loads(rel2id_path.read_text(encoding="utf-8")) if rel2id_path.exists() else {}

    # Build inverse maps for readability in plots/reports
    id2entity = {v: k for k, v in entity2id.items()}
    id2rel = {v: k for k, v in rel2id.items()}

    # --- Relation distribution ---
    rel_types = g.edata["rel_type"].cpu().numpy()
    rel_counts = Counter(rel_types.tolist())
    rel_labels = [id2rel.get(rid, str(rid)) for rid in sorted(rel_counts.keys())]
    rel_values = [rel_counts[rid] for rid in sorted(rel_counts.keys())]

    plt.figure()
    plt.bar(rel_labels, rel_values)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(rel_dist_path)
    plt.close()

    # --- Degree distribution (log-log) ---
    deg = (g.in_degrees() + g.out_degrees()).cpu().numpy()
    deg_sorted = np.sort(deg)

    # Use histogram bins
    hist, bin_edges = np.histogram(deg_sorted, bins=50)
    x = bin_edges[1:]
    y = hist

    plt.figure()
    plt.xscale("log")
    plt.yscale("log")
    plt.plot(x, y)
    plt.xlabel("Degree (log)")
    plt.ylabel("Count (log)")
    plt.tight_layout()
    plt.savefig(degree_log_path)
    plt.close()

    # --- Top nodes by degree ---
    topk = 20
    top_idx = np.argsort(-deg)[:topk]
    top_labels = [id2entity.get(int(i), str(int(i))) for i in top_idx]
    top_values = [int(deg[int(i)]) for i in top_idx]

    plt.figure()
    plt.barh(list(reversed(top_labels)), list(reversed(top_values)))
    plt.tight_layout()
    plt.savefig(top_nodes_path)
    plt.close()

    # --- Report ---
    report = {
        "nodes": int(g.num_nodes()),
        "edges": int(g.num_edges()),
        "relations": {id2rel.get(int(k), str(k)): int(v) for k, v in rel_counts.items()},
        "degree": {
            "min": int(deg.min()) if len(deg) else 0,
            "max": int(deg.max()) if len(deg) else 0,
            "mean": float(deg.mean()) if len(deg) else 0.0,
            "median": float(np.median(deg)) if len(deg) else 0.0,
        },
        "metadata": metadata,
        "plots": {
            "relation_distribution": str(rel_dist_path),
            "degree_distribution_log": str(degree_log_path),
            "top_nodes_by_degree": str(top_nodes_path),
        },
    }

    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Saved plots in: {plots_dir}")
    logger.info(f"Saved analytics report: {report_path}")
