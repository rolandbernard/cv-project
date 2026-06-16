
import os
import json
import random

import torch
import numpy as np
import scipy.optimize

# Slightly lower precision for performance.
torch.set_float32_matmul_precision('high')

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# The set of links between keypoints that make up the skeleton in the COCO pose model.
SKELETON = [
    [15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11],
    [6, 12], [5, 6], [5, 7], [6, 8], [7, 9], [8, 10], [1, 2],
    [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]
]
# Links that should have a constant physical length over time.
RIGID_SKELETON = [
    [15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11],
    [6, 12], [5, 6], [5, 7], [6, 8], [7, 9], [8, 10], [1, 2],
    [0, 1], [0, 2], [1, 3], [2, 4]
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


def save_json(file: str, data):
    """ Save a JSON file and create the directory if necessary. """
    os.makedirs(os.path.dirname(file), exist_ok=True)
    with open(file, "w") as f:
        json.dump(data, f)


def load_json(file: str):
    """ Load some JSON file into a python object and return it. """
    with open(file, "r") as f:
        return json.load(f)


def save_tracks(
    file: str, cams: list, frames: list[list], fps: float,
    center: tuple[float, float, float] = (0, 0, 0), up: tuple[float, float, float] = (0, -1, 0)
):
    """ Save recorded tracking data to the given file. """
    save_json(file, {
        "cameras": [{k: to_list(c[k]) for k in ["R", "t", "K"]} for c in cams],
        "frames": [[{k: to_list(t[k]) for k in ["id", "kpts"]} for t in f] for f in frames],
        "fps": fps, "center": list(center), "up": list(up)
    })


def load_tracks(file: str):
    """ Load recorded tracking data from the given file. """
    data = load_json(file)
    return data["cameras"], data["frames"], data["fps"], data["center"], data["up"]


def evaluate_mot_metrics(gt_frames, pred_frames, dist_threshold=15.0):
    """ Computes MPJPE, MOTA, MOTP, Precision, and Recall from data. """
    total_gt_kpts, total_pred_kpts = 0, 0
    total_tp, total_fp, total_fn = 0, 0, 0
    total_id_switches = 0
    mpjpe_errors, mpjpe_count = 0, 0
    motp_errors = 0
    prev_gt_to_pred_map = {}
    # Only go until the minium of gt and prediction since some videos are cut
    # short before ground truth values end.
    for idx in range(min(len(gt_frames), len(pred_frames))):
        frame_gt = gt_frames[idx] if idx < len(gt_frames) else {}
        frame_pred = pred_frames[idx] if idx < len(pred_frames) else {}
        total_gt_kpts += len(frame_gt)
        total_pred_kpts += len(frame_pred)
        # Build cost matrix based on MPJPE distance.
        cost_matrix = np.zeros((len(frame_gt), len(frame_pred)))
        for i, gt in enumerate(frame_gt):
            for j, pred in enumerate(frame_pred):
                gt_joints = np.array(gt["kpts"])
                valid = np.sum(np.abs(gt_joints), axis=1) != 0.0
                pred_joints = np.array(pred["kpts"])
                cost_matrix[i, j] = np.mean(
                    np.linalg.norm(gt_joints - pred_joints, axis=1),
                    where=valid
                )
        gt_ind, pred_ind = scipy.optimize.linear_sum_assignment(cost_matrix)
        current_gt_to_pred_map = prev_gt_to_pred_map.copy()
        matches = 0
        for g, p in zip(gt_ind, pred_ind):
            mpjpe_errors += cost_matrix[g, p].item()
            mpjpe_count += 1
            # Apply gating threshold to only allow those with mean below threshold.
            if cost_matrix[g, p] <= dist_threshold:
                gt_id, pred_id = frame_gt[g]["id"], frame_pred[p]["id"]
                current_gt_to_pred_map[gt_id] = pred_id
                motp_errors += cost_matrix[g, p].item()
                matches += 1
                # Check for identity switch
                if gt_id in prev_gt_to_pred_map and prev_gt_to_pred_map[gt_id] != pred_id:
                    total_id_switches += 1
        total_tp += matches
        # Unmatched ground truth items are false negatives.
        total_fn += len(frame_gt) - matches
        # Unmatched predictions items are false positives.
        total_fp += len(frame_pred) - matches
        # Save assignment map for next iteration.
        prev_gt_to_pred_map = current_gt_to_pred_map
    # Compute metrics.
    mota = 1.0 - ((total_fn + total_fp + total_id_switches) / total_gt_kpts) \
        if total_gt_kpts > 0 else 0.0
    motp = motp_errors / total_tp if total_tp > 0 else 0.0
    mpjpe = mpjpe_errors / mpjpe_count if mpjpe_count > 0 else 0.0
    precision = total_tp / (total_tp + total_fp) \
        if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) \
        if (total_tp + total_fn) > 0 else 0.0
    # Output results as dictionary.
    return {
        "MPJPE": mpjpe,
        "MOTA": mota,
        "MOTP (Avg Miss Distance)": motp,
        "Precision": precision,
        "Recall": recall,
        "Counts": {
            "GT Objects": total_gt_kpts,
            "Pred Objects": total_pred_kpts,
            "True Positives": total_tp,
            "False Positives": total_fp,
            "False Negatives": total_fn,
            "ID Switches": total_id_switches
        }
    }


def evaluate_from_files(gt_file: str, pred_file: str, dist_threshold=15.0):
    """ Load results and ground truth from the given files and compute metrics. """
    _, gt_frames, _, _, _ = load_tracks(gt_file)
    _, pred_frames, _, _, _ = load_tracks(pred_file)
    return evaluate_mot_metrics(gt_frames, pred_frames, dist_threshold)
