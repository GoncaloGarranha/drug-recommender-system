from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np


def compute_ranking_metrics(ranks: Sequence[float], hits_k: Tuple[int, ...] = (1, 3, 10)) -> Dict[str, float]:
	"""Compute MRR and Hits@K from a sequence of ranks."""
	if not ranks:
		raise ValueError("No ranks provided for metric computation.")

	ranks_array = np.array(ranks, dtype=np.float64)
	return {
		"mrr": float(np.mean(1.0 / ranks_array)),
		**{f"hits@{k}": float(np.mean(ranks_array <= k)) for k in hits_k},
	}
