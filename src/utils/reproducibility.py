# src/utils/reproducibility.py
import random
import os
import numpy as np
import torch

def set_seed(seed: int = 42):
    """
    Freezes all random variance generators across Python, NumPy, and PyTorch
    in order to ensure mathematical reproducibility of model experiments.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Enforce deterministic behaviors in CuDNN algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False