
import os
import time
import argparse

import cv2
import torch
import numpy as np
import scipy.interpolate

import util
from camera import Camera, triangulate
from detect import PoseDetector
from source import OfflineVideoSource, OnlineVideoSource
from tracker import (
    CrossViewFirstTracker, Tracker,
    build_constrained_physics, build_physics,
)
from visualize import LiveSkeletonPlayer, show_cv2_images

try:
    from moge.model.v2 import MoGeModel
    MOGE_AVAILABLE = True
except ImportError:
    MOGE_AVAILABLE = False

try:
    from kornia.feature import LoFTR
    LOFTR_AVAILABLE = True
except ImportError:
    LOFTR_AVAILABLE = False


def rootsift(des: np.ndarray) -> np.ndarray:
    """ Compute Root-SIFT from SIFT features. """
    des /= (des.sum(axis=1, keepdims=True) + 1e-7)
    return np.sqrt(des)


def match_images_sift(img1: torch.Tensor, img2: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """ Create a set of point correspondences between two images using SIFT. """
    # Extract features from the images
    sift = cv2.SIFT_create(  # type: ignore
        nfeatures=10000, contrastThreshold=0.01, edgeThreshold=15)
    kp1, des1 = sift.detectAndCompute(
        cv2.cvtColor(img1.numpy(), cv2.COLOR_RGB2GRAY), None)
    kp2, des2 = sift.detectAndCompute(
        cv2.cvtColor(img2.numpy(), cv2.COLOR_RGB2GRAY), None)
    if des1 is None or des2 is None:
        print("No features found.")
        exit(1)
    des1, des2 = rootsift(des1), rootsift(des2)
    # Perform matching using features
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=100))
    ms = flann.knnMatch(des1, des2, k=2)  # type: ignore
    gms = sorted(ms, key=lambda pt: pt[0].distance / pt[1].distance)[:50]
    pts1 = np.array([kp1[m.queryIdx].pt for m, _ in gms], dtype=np.float32)
    pts2 = np.array([kp2[m.trainIdx].pt for m, _ in gms], dtype=np.float32)
    return pts1, pts2


def match_images_loftr(model, img1: torch.Tensor, img2: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """ Create a set of point correspondences between two images using LoFTR. """
    img1_g = torch.from_numpy(cv2.cvtColor(img1.numpy(), cv2.COLOR_RGB2GRAY))
    img2_g = torch.from_numpy(cv2.cvtColor(img2.numpy(), cv2.COLOR_RGB2GRAY))
    with torch.inference_mode():
        correspondences = model({
            "image0": img1_g.unsqueeze(0).unsqueeze(0).float() / 255.0,
            "image1": img2_g.unsqueeze(0).unsqueeze(0).float() / 255.0
        })
    pts1 = correspondences["keypoints0"].cpu().numpy()
    pts2 = correspondences["keypoints1"].cpu().numpy()
    return pts1, pts2


def filter_matches(
    pts1: np.ndarray, pts2: np.ndarray, cam1: None | Camera = None, cam2: None | Camera = None
) -> tuple[np.ndarray, np.ndarray]:
    """ Filter out outliers in matched points between images. """
    if cam1 is not None and cam2 is not None:
        _, mask = cv2.findEssentialMat(
            pts1, pts2,
            cam1.intrinsic.numpy(), cam1.distortion.numpy(),
            cam2.intrinsic.numpy(), cam2.distortion.numpy(),
            method=cv2.RANSAC, threshold=1.0, prob=0.99
        )
    else:
        _, mask = cv2.findFundamentalMat(
            pts1, pts2, cv2.USAC_MAGSAC, 1.0, 0.99, 5000)
    return pts1[mask.ravel() == 1], pts2[mask.ravel() == 1]


def triangulate_matches(pts1: np.ndarray, pts2: np.ndarray, cam1: Camera, cam2: Camera) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Triangulate matches between cameras and filter out outliers. """
    pts3d = triangulate(
        [cam1, cam2], [torch.from_numpy(pts1), torch.from_numpy(pts2)]
    )
    rep1, rep2 = cam1.project(pts3d).numpy(), cam2.project(pts3d).numpy()
    valid1 = np.linalg.norm(rep1 - pts1, axis=-1) < 1.0
    valid2 = np.linalg.norm(rep2 - pts2, axis=-1) < 1.0
    mask = valid1 & valid2
    pts1, pts2 = pts1[mask].astype(np.int64), pts2[mask].astype(np.int64)
    return pts1, pts2, pts3d.numpy()[mask]


def points_from_depth(cam: Camera, depth: torch.Tensor, scale: float | torch.Tensor) -> torch.Tensor:
    """ Unproject points using depth values and a scale. """
    height, width = depth.shape
    v, u = torch.meshgrid(
        torch.arange(height), torch.arange(width), indexing="ij")
    uv = cam.undistort_points(torch.stack([u, v], dim=-1))
    depth = depth.unsqueeze(-1)
    cam3d = torch.concat([uv * depth, depth], dim=-1) * scale
    return cam.camera_to_world(cam3d)


def evaluate_moge(model, cam: Camera, img: torch.Tensor) -> torch.Tensor:
    """ Evaluate the MoGe-2 model on the given image. """
    height, width = img.shape[:2]
    p_img = (img.to(torch.float32) / 255.0).permute(2, 0, 1).to(util.DEVICE)
    fov_x = torch.rad2deg(2 * torch.atan(width / (2 * cam.intrinsic[0, 0]))) \
        if cam.has_intrinsics() else None
    with torch.inference_mode():
        out = model.infer(p_img, fov_x=fov_x)
    if not cam.has_intrinsics():
        K: torch.Tensor = out["intrinsics"].cpu()
        K[0, :] *= width
        K[1, :] *= height
        cam.intrinsic = K
    return out["depth"].cpu()


def estimate_params(cams: list[Camera], imgs: list[torch.Tensor], use_loftr: bool) -> list[np.ndarray]:
    """ Estimate camera extrinsics using point correspondences. """
    # Create matches between pixels in different images.
    matches: list[list[tuple]] = [[([], []) for _ in cams] for _ in cams]
    if use_loftr:
        model = LoFTR("indoor_new")
    for i in range(len(cams)):
        for j in range(i):
            if cams[i].forward() @ cams[j].forward() > 0.7:
                if use_loftr:
                    pts1, pts2 = match_images_loftr(model, imgs[i], imgs[j])
                else:
                    pts1, pts2 = match_images_sift(imgs[i], imgs[j])
                pts1, pts2 = filter_matches(pts1, pts2, cams[i], cams[j])
                matches[i][j] = (pts1, pts2)
                matches[j][i] = (pts2, pts1)
    points3d = [np.zeros(img.shape, dtype=np.float32) for img in imgs]
    if any(cam.has_extrinsics() for cam in cams):
        # Assume all extrinsics are already correct.
        for i, ms in enumerate(matches):
            for j, (pts1, pts2) in enumerate(ms):
                if len(pts1) >= 1:
                    pts1, pts2, pts3d \
                        = triangulate_matches(pts1, pts2, cams[i], cams[j])
                    points3d[i][pts1[:, 1], pts1[:, 0]] = pts3d
                    points3d[j][pts2[:, 1], pts2[:, 0]] = pts3d
    elif len(cams) >= 2:
        # Start building with two cameras with most matches to first camera.
        mi = max(range(len(cams)), key=lambda i: len(matches[0][i][0]))
        pts1, pts2 = matches[0][mi]
        _, _, R, t, _ = cv2.recoverPose(
            pts1, pts2,
            cams[0].intrinsic.numpy(), cams[0].distortion.numpy(),
            cams[mi].intrinsic.numpy(), cams[mi].distortion.numpy(),
            method=cv2.RANSAC, threshold=1.0, prob=0.99
        )
        cams[mi].rotation = torch.from_numpy(R.astype(np.float32))
        cams[mi].translation = torch.from_numpy(t.astype(np.float32).flatten())
        pts1, pts2, pts3d = triangulate_matches(pts1, pts2, cams[0], cams[mi])
        points3d[0][pts1[:, 1], pts1[:, 0]] = pts3d
        points3d[mi][pts2[:, 1], pts2[:, 0]] = pts3d
        # Add all other cameras
        added = {0, mi}
        missing = {i for i in range(len(cams)) if i != 0 and i != mi}
        while len(missing) > 0:
            mi = max(
                missing,
                key=lambda i: sum(len(matches[i][j][0]) for j in added)
            )
            pts2d, pts3d = [], []
            for j, (pts1, pts2) in enumerate(matches[mi]):
                if j in added and len(pts1) >= 1:
                    pts2 = pts2.astype(np.int64)
                    pt3d = points3d[j][pts2[:, 1], pts2[:, 0]]
                    mask = np.any(pt3d != 0, axis=1)
                    pts2d.append(pts1[mask])
                    pts3d.append(pt3d[mask])
            pts2d, pts3d = np.concat(pts2d), np.concat(pts3d)
            _, rvec, t, _ = cv2.solvePnPRansac(
                pts3d, pts2d,
                cams[mi].intrinsic.numpy(), cams[mi].distortion.numpy(),
                reprojectionError=8.0, confidence=0.99, iterationsCount=5000,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            R, _ = cv2.Rodrigues(rvec)
            cams[mi].rotation = torch.from_numpy(R.astype(np.float32))
            cams[mi].translation \
                = torch.from_numpy(t.astype(np.float32).flatten())
            for j, (pts1, pts2) in enumerate(matches[mi]):
                if j in added and len(pts1) >= 1 and cams[mi].forward() @ cams[j].forward() > 0.7:
                    pts1, pts2, pts3d \
                        = triangulate_matches(pts1, pts2, cams[mi], cams[j])
                    points3d[mi][pts1[:, 1], pts1[:, 0]] = pts3d
                    points3d[j][pts2[:, 1], pts2[:, 0]] = pts3d
            added.add(mi)
            missing.remove(mi)
    return points3d


def estimate_params_simple(cams: list[Camera], imgs: list[torch.Tensor], use_loftr: bool) -> list[tuple]:
    """ Estimate camera parameters and generate point clouds. """
    for cam, img in zip(cams, imgs):
        if not cam.has_intrinsics():
            # Best guess for intrinsics.
            f = 1.2 * max(*img.shape)
            cam.intrinsic = torch.tensor([
                [f, 0.0, img.shape[1] / 2.0],
                [0.0, f, img.shape[0] / 2.0],
                [0.0, 0.0, 1.0]
            ])
    points3d = estimate_params(cams, imgs, use_loftr)
    pts, clrs = [], []
    for pts3d, img in zip(points3d, imgs):
        mask = np.any(pts3d != 0, axis=2)
        pts.append(pts3d[mask])
        clrs.append(img[mask])
    return list(zip(pts, clrs))


def depth_refinement(dense: np.ndarray, mask: np.ndarray, sparse: np.ndarray) -> np.ndarray:
    """ Refines a dense depth map using sparse anchors. """
    rbf = scipy.interpolate.RBFInterpolator(
        np.argwhere(mask) / np.array(dense.shape),
        sparse[mask] / dense[mask],
        kernel='thin_plate_spline', smoothing=0.5
    )
    yy, xx = np.mgrid[0:dense.shape[0], 0:dense.shape[1]]
    coords = np.stack([yy.ravel(), xx.ravel()], axis=-1)
    scale_field = rbf(coords / np.array(dense.shape)).reshape(dense.shape)
    return dense * scale_field


def estimate_params_moge(
    cams: list[Camera], imgs: list[torch.Tensor], use_loftr: bool, center: None | tuple = None, up: None | tuple = None
) -> list[tuple]:
    """ Estimate camera parameters and generate point clouds. """
    known_extr = any(cam.has_extrinsics() for cam in cams)
    moge_model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal")
    moge_model.to(util.DEVICE)
    moge_model.eval()
    depths = []
    for cam, img in zip(cams, imgs):
        depths.append(evaluate_moge(moge_model, cam, img).numpy())
    points3d = estimate_params(cams, imgs, use_loftr)
    scales = []
    for cam, pts3d, depth in zip(cams, points3d, depths):
        mask = np.any(pts3d != 0, axis=2)
        pts3d = cam.world_to_camera(torch.from_numpy(pts3d)).numpy()
        scale = np.median(depth[mask] / pts3d[mask, 2])
        depth /= scale
        scales.append(scale)
        depth[:, :] = depth_refinement(depth, mask, pts3d[..., 2])
    pts_scale = 1 if known_extr else np.median(scales)
    pts, clrs = [], []
    for cam, depth, img in zip(cams, depths, imgs):
        cam.scale(pts_scale)
        depth = torch.from_numpy(depth)
        if center is not None and up is not None:
            # If the floor is known, force depth to be above it.
            height, width = depth.shape
            v, u = torch.meshgrid(
                torch.arange(height), torch.arange(width), indexing="ij")
            uv = cam.undistort_points(torch.stack([u, v], dim=-1))
            dirs = torch.concat(
                [uv, torch.ones(height, width, 1)], dim=-1) @ cam.rotation
            c = torch.tensor(center, dtype=torch.float32)
            n = torch.tensor(up, dtype=torch.float32)
            denom = (dirs @ n)
            plane, mask = ((c - cam.center()) @ n) / denom, denom < -1e-5
            depth[mask] = torch.minimum(depth[mask], plane[mask])
        pts3d = points_from_depth(cam, depth, pts_scale)
        pts.append(pts3d.numpy())
        clrs.append(img.numpy())
    return list(zip(pts, clrs))


def augment_tracks(
    file: str, cams: list[Camera], streams: list[str], use_loftr: bool = False,
    use_moge: bool = False, center: None | tuple = None, up: None | tuple = None
):
    """
    Augments a recorded tracking data JSON file with video stream URLs,
    point cloud 3D coordinates, and their colors.
    """
    data = util.load_json(file)
    backgrounds = []
    for video_path in streams:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video source {video_path}")
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError(f"Could not read video source {video_path}")
        bg_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        backgrounds.append(torch.from_numpy(bg_img))
    if use_loftr:
        if not LOFTR_AVAILABLE:
            raise RuntimeError("LoFTR not found.")
    if use_moge:
        if not MOGE_AVAILABLE:
            raise RuntimeError("MoGe-2 not found.")
        clouds = estimate_params_moge(cams, backgrounds, use_loftr, center, up)
    else:
        clouds = estimate_params_simple(cams, backgrounds, use_loftr)
    all_pts, all_clrs = [], []
    for pts, clrs in clouds:
        all_pts.append(pts)
        all_clrs.append(clrs)
    data["stream"] = [os.path.abspath(p) for p in streams]
    data["points"] = [util.to_list(pts) for pts in all_pts]
    data["colors"] = [util.to_list(clrs) for clrs in all_clrs]
    data["cameras"] = [
        {k: util.to_list(c[k]) for k in ["R", "t", "K", "distCoef"]} for c in cams]
    util.save_json(file, data)


if __name__ == "__main__":
    """ Main execution function for the demo. """
    parser = argparse.ArgumentParser(
        description="Cross-view 3D skeleton tracking demo.")
    parser.add_argument("urls", nargs="+",
                        help="URLs or paths to video sources")
    parser.add_argument("--cams", nargs="+",
                        help="Calibration files for cameras")
    parser.add_argument("--distance", type=float,
                        help="Known distance between cameras for scaling")
    parser.add_argument("--scale", type=float,
                        help="Known unit scale of the extrinsics in meters")
    parser.add_argument("--resize", type=int, nargs=2,
                        help="Resize frames to (width, height)")
    parser.add_argument("--moge", action="store_true",
                        help="Use MoGe-2 for camera estimation")
    parser.add_argument("--loftr", action="store_true",
                        help="Use LoFTR for camera estimation")
    parser.add_argument("--no-constraint", action="store_true",
                        help="Do not use rigid body constraints")
    parser.add_argument("--cross-first", action="store_true",
                        help="Match using cross-view association first")
    parser.add_argument("--no-cloud", action="store_true",
                        help="Do not add point clouds to the visualization")
    parser.add_argument("--intrinsics-only", action="store_true",
                        help="Only load intrinsics from files")
    parser.add_argument("--cams-only", action="store_true",
                        help="Only show camera positions")
    args = parser.parse_args()
    cameras = [Camera() for _ in args.urls]
    if args.cams is not None:
        for cam, file in zip(cameras, args.cams):
            cam.load_file(file)
            if args.intrinsics_only:
                cam.rotation = torch.eye(3)
                cam.translation = torch.zeros(3)
    clouds: list[tuple] = []
    if not args.cams_only:
        if all(os.path.isfile(u) for u in args.urls):
            source = OfflineVideoSource(args.urls)
        else:
            source = OnlineVideoSource(
                [int(s) if s.isdigit() else s for s in args.urls])
        source.start()
        # Wait some time to make sure all cameras are connected and get frames.
        time.sleep(1)
        ts, frames, _ = source.next_frames()
        if ts is None or frames is None:
            print("Failed to get initial frames.")
            exit(1)
        if args.loftr:
            if not LOFTR_AVAILABLE:
                print("LoFTR not found.")
                exit(1)
        if args.moge:
            if not MOGE_AVAILABLE:
                print("MoGe-2 not found.")
                exit(1)
            clouds = estimate_params_moge(cameras, frames, args.loftr)
        else:
            clouds = estimate_params_simple(cameras, frames, args.loftr)
        # Scale distance if ground truth is provided.
        scale = 1
        if args.distance is not None and len(cameras) >= 2:
            dist = torch.linalg.vector_norm(
                cameras[0].center() - cameras[1].center()).item()
            scale = args.distance / dist
        elif args.scale is not None:
            scale = args.scale
        if scale != 1:
            for cam in cameras:
                cam.scale(scale)
            for pts, _ in clouds:
                pts *= scale
    # Setup the player and tracker.
    player = LiveSkeletonPlayer(cameras)
    if not args.no_cloud:
        for pts, clrs in clouds:
            player.add_point_cloud(pts, clrs)
    if not args.cams_only:
        source.cameras = cameras
        source.resize = tuple(args.resize) if args.resize is not None else None
        source.to(util.DEVICE)
        detector = PoseDetector()
        detector.to(util.DEVICE)
        physics = build_physics(1.0) \
            if args.no_constraint else build_constrained_physics(1.0)
        physics.to(util.DEVICE)
        tracker_cls = CrossViewFirstTracker if args.cross_first else Tracker
        tracker = tracker_cls(detector, physics)
        # Run the tracking loop and update the visualization.
        last_ts = ts
        while True:
            ts, frames, _ = source.next_frames()
            if ts is None or frames is None:
                break
            dt = ts - last_ts
            if dt <= 0:
                # We don"t have any new frames available.
                time.sleep(0.01)
                continue
            tracker.predict(dt)
            tracker.update(cameras, frames)
            last_ts = ts
            player.update(tracker.get_prediction())
            show_cv2_images(
                cameras,
                [cv2.cvtColor(f.cpu().numpy(), cv2.COLOR_RGB2BGR)
                 for f in frames],
                tracker.get_prediction()
            )
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    else:
        while True:
            player.update([])
            time.sleep(0.01)
