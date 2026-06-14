
import random

import torch
import numpy as np


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# The set of links between keypoints that make up the skeleton in the COCO pose model.
SKELETON = [
    [15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11],
    [6, 12], [5, 6], [5, 7], [6, 8], [7, 9], [8, 10], [1, 2],
    [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]
]


def set_seed(seed=42):
    """
    Set the seed for builtin, numpy, and torch random modules. To be called
    before any operation involving randomness to ensure repeatability.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def remove_idx(input: torch.Tensor, idx: torch.Tensor, dim=0) -> torch.Tensor:
    """
    Remove all indices in `idx` from the input tensor at dimension `dim`.
    """
    mask = torch.ones(input.shape[dim], dtype=torch.bool, device=input.device)
    mask[idx] = False
    return torch.index_select(input, dim, torch.nonzero(mask).squeeze())


def count_model_params(model):
    """
    This is a simple function that just computes the number of trainable
    parameters of the given model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
