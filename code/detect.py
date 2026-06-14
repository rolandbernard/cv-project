
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ultralytics import YOLO
from ultralytics.nn.tasks import PoseModel
from ultralytics.nn.modules import Detect, Pose26, Conv
import scipy.optimize

import util
import dataset
from camera import Camera
from util import NetStorage


class PoseDetector:
    """
    This is an wrapper class around the YOLO based pose estimation models. Given
    an image it produces as output a tensor of shape (P, K*3) where P is the
    number of detected persons an K = 17 is the number of keypoints. It also
    gives an estimated covariance matrix based on keypoint visibility.
    """

    def __init__(
        self, model_name: str = "yolo26n-pose", threshold: float = 0.5, kpt_threshold: float = 0.25,
        min_keypoint: int = 3, var_min: float = 16.0, var_vis: float = 0.005, var_inv: float = 5.0,
        path: str = "./nets", compile: bool = True
    ):
        model: PoseModel = YOLO(
            f"{path}/{model_name}.pt").model  # type: ignore
        model.eval()
        if compile:
            model.compile()
        self.model = model
        self.threshold = threshold
        self.kpt_threshold = kpt_threshold
        self.min_keypoint = min_keypoint
        self.var_min = var_min
        self.var_vis = var_vis
        self.var_inv = var_inv
        self.num_keypoint = 17

    def detect_base(self, images: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """
        Basic version of the detection loop that expects the images to be already
        batched in the expected format.
        """
        pred, _ = self.model(images)
        results = []
        for img_res in pred:
            valid = img_res[(img_res[:, 4] > self.threshold) &
                            ((img_res[:, 8::3] > self.kpt_threshold).sum() >= self.min_keypoint)]
            valid_points = valid[:, 6:].view(-1, self.num_keypoint, 3)
            # Compute variance based on bounding box size and kpt visibility.
            bb_size = torch.linalg.vector_norm(
                valid[:, 2:4] - valid[:, 0:2], dim=1, keepdim=True)
            vis = torch.clamp(
                (valid_points[:, :, 2] - self.kpt_threshold) / (1.0 - self.kpt_threshold), min=0)
            var = self.var_min + bb_size * bb_size * \
                (self.var_vis / (vis + self.var_vis / self.var_inv))
            results.append((
                valid_points[:, :, 0:2].reshape(-1, self.num_keypoint*2),
                torch.kron(torch.diag_embed(var),
                           torch.eye(2, device=var.device))
            ))
        return results

    def detect_simple(self, images: torch.Tensor | list[torch.Tensor]) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """
        Run the detection algorithm and return discovered keypoints. We expect
        a batch of images and return a batch of results. For the results we return
        for P persons two tensors of shape (P, 17*2) for the positions of all 17
        keypoints and and a tensor of shape (P, 17*2, 17*2) for the covariance.
        """
        with torch.inference_mode():
            if isinstance(images, list):
                images = torch.stack(images)
            if images.shape[-1] == 3:
                images = images.permute(0, 3, 1, 2)
            if images.dtype == torch.uint8:
                images = images.to(torch.float32) / 255.0
            return self.detect_base(images)

    def detect(self, cams: list[Camera], images: torch.Tensor | list[torch.Tensor]) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """
        This is like `detect_simple`, but the returned keypoints are in normalized
        camera coordinates rather than in pixel space, effectively removing the
        camera intrinsics.
        """
        return [
            (cam.undistort_points(pos), cam.undistort_covars(cov))
            for cam, (pos, cov) in zip(cams, self.detect_simple(images))
        ]

    def to(self, *args, **kargs):
        """
        Apply the PyTorch `.to` method to the contained model.
        """
        self.model = self.model.to(*args, **kargs)
        return self

