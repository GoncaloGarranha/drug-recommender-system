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


def generate_analytics(cfg: Dict[str, Any], paths, force: bool = False) -> None:
    """
    Step 08: Generate analytics over the training graph.
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

    graph_bin = processed_dir / "train_graph.bin"
    metadata_json = processed_dir / "graph_metadata.json"
    entity_to_id_json = processed_dir / "entity2id.json"
    relation_to_id_json = processed_dir / "rel2id.json"
    report_json = reports_dir / "analytics_report.json"

    relation_dist_plot = plots_dir / "relation_distribution.png"
    degree_log_plot = plots_dir / "degree_distribution_log.png"
    top_nodes_plot = plots_dir / "top_nodes_by_degree.png"

    if not force and report_json.exists() and relation_dist_plot.exists() and degree_log_plot.exists() and top_nodes_plot.exists():
        logger.info("Analytics outputs already exist. Skipping (use --force to re-run).")
        return

    if not graph_bin.exists():
        raise FileNotFoundError("Missing train_graph.bin. Run 07_index_dgl first.")

    graphs_list, _ = dgl.load_graphs(str(graph_bin))
    graph = graphs_list[0]

    metadata = json.loads(metadata_json.read_text(encoding="utf-8")) if metadata_json.exists() else {}
    entity_to_id = json.loads(entity_to_id_json.read_text(encoding="utf-8")) if entity_to_id_json.exists() else {}
    relation_to_id = json.loads(relation_to_id_json.read_text(encoding="utf-8")) if relation_to_id_json.exists() else {}

    id_to_entity = {v: k for k, v in entity_to_id.items()}
    id_to_relation = {v: k for k, v in relation_to_id.items()}

    rel_types = graph.edata["etype"].cpu().numpy()
    rel_counts = Counter(rel_types.tolist())
    rel_labels = [id_to_relation.get(rid, str(rid)) for rid in sorted(rel_counts.keys())]
    rel_values = [rel_counts[rid] for rid in sorted(rel_counts.keys())]

    plt.figure()
    plt.bar(rel_labels, rel_values)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(relation_dist_plot)
    plt.close()

    degrees = (graph.in_degrees() + graph.out_degrees()).cpu().numpy()
    degrees_sorted = np.sort(degrees)

    hist, bin_edges = np.histogram(degrees_sorted, bins=50)
    x = bin_edges[1:]
    y = hist

    plt.figure()
    plt.xscale("log")
    plt.yscale("log")
    plt.plot(x, y)
    plt.xlabel("Degree (log)")
    plt.ylabel("Count (log)")
    plt.tight_layout()
    plt.savefig(degree_log_plot)
    plt.close()

    topk = 20
    top_indices = np.argsort(-degrees)[:topk]
    top_labels = [id_to_entity.get(int(i), str(int(i))) for i in top_indices]
    top_values = [int(degrees[int(i)]) for i in top_indices]

    plt.figure()
    plt.barh(list(reversed(top_labels)), list(reversed(top_values)))
    plt.tight_layout()
    plt.savefig(top_nodes_plot)
    plt.close()

    report = {
        "nodes": int(graph.num_nodes()),
        "edges": int(graph.num_edges()),
        "relations": {id_to_relation.get(int(k), str(k)): int(v) for k, v in rel_counts.items()},
        "degree": {
            "min": int(degrees.min()) if len(degrees) else 0,
            "max": int(degrees.max()) if len(degrees) else 0,
            "mean": float(degrees.mean()) if len(degrees) else 0.0,
            "median": float(np.median(degrees)) if len(degrees) else 0.0,
        },
        "metadata": metadata,
        "plots": {
            "relation_distribution": str(relation_dist_plot),
            "degree_distribution_log": str(degree_log_plot),
            "top_nodes_by_degree": str(top_nodes_plot),
        },
    }

    report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Saved plots in: {plots_dir}")
    logger.info(f"Saved analytics report: {report_json}")


run = generate_analytics


if __name__ == "__main__":
    from kg_constructor.src.utils.config import load_config

    class Paths:
        def __init__(self, root: Path):
            self.root = root
            self.output_dir = root / "kg_constructor" / "data" / "output"
            self.processed_dir = root / "kg_constructor" / "data" / "processed"

    root = Path(__file__).parent.parent.parent.parent
    config = load_config(root / "kg_constructor" / "configs" / "config.yaml")
    paths = Paths(root)
    
    generate_analytics(config, paths, force=True)

