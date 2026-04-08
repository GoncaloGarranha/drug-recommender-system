import os
import random
import numpy as np
import torch

def set_seed(seed: int) -> None:
    """Set random seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # CPU-first deterministic settings
    torch.use_deterministic_algorithms(False)
