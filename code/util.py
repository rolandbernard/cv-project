
import json
import random
from typing import Any

import torch
import numpy as np

# Slightly lower precision for performance.
torch.set_float32_matmul_precision('high')

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# The set of links between keypoints that make up the skeleton in the COCO pose model.
SKELETON = [
    [15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11],
    [6, 12], [5, 6], [5, 7], [6, 8], [7, 9], [8, 10], [1, 2],
    [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]
]
COLORS = [
    "#FF5733", "#33FF57", "#3357FF", "#FF33A8",
    "#33FFF5", "#F5FF33", "#A833FF", "#F3AAEE"
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
    """ Remove all indices in `idx` from the input tensor at dimension `dim`. """
    mask = torch.ones(input.shape[dim], dtype=torch.bool, device=input.device)
    mask[idx] = False
    return torch.index_select(input, dim, torch.nonzero(mask).squeeze())


def count_model_params(model):
    """
    This is a simple function that just computes the number of trainable
    parameters of the given model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def to_list(something) -> list:
    if isinstance(something, np.ndarray) or isinstance(something, torch.Tensor):
        return something.tolist()
    if isinstance(something, list):
        return [to_list(s) for s in something]
    return something


def save_tracks(
    file: str, cams: list[Any], frames: list[list[Any]], fps: float,
    center: tuple[float, float, float] = (0, 0, 0), up: tuple[float, float, float] = (0, 1, 0)
):
    """ Save recorded tracking data to the given file. """
    data = {
        "cameras": [{k: to_list(c[k]) for k in ["R", "t", "K"]} for c in cams],
        "frames": [[{k: to_list(t[k]) for k in ["id", "kpts"]} for t in f] for f in frames],
        "fps": fps, "center": list(center), "up": list(up)
    }
    with open(file, "w") as f:
        json.dump(data, f)


def load_tracks(file: str):
    """ Load recorded tracking data from the given file. """
    with open(file, "r") as f:
        data = json.load(f)
    return data["cameras"], data["frames"], data["fps"], data["center"], data["up"]
