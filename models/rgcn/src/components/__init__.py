"""Reusable R-GCN model components."""

from .decoder import DistMultDecoder
from .encoder import RGCNEncoder

__all__ = ["DistMultDecoder", "RGCNEncoder"]
