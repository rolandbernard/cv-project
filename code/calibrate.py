
import os
import json
import time
import shutil
import argparse
from itertools import count

import cv2
import torch
import numpy as np
import networkx as nx
import scipy.optimize

from camera import Camera
from source import OfflineVideoSource, OnlineVideoSource


def multi_camera_calibrate(
    cameras: list[Camera], all_pts: list[list[np.ndarray]], objp: np.ndarray,
    img_size: tuple[int, int], keep_intrinsics=False
):
    """ Calibrate multiple cameras from checkerboard target detections. """
    # Compute camera intrinsics for each camera individually.
    if not keep_intrinsics:
        for i, cam in enumerate(cameras):
            imgpts = [pts[i] for pts in all_pts if pts[i] is not None]
            _, cameraMatrix, distCoeffs, _, _ = cv2.calibrateCamera(
                [objp]*len(imgpts), imgpts,
                img_size, None, None  # type: ignore
            )
            cam.intrinsic = torch.from_numpy(cameraMatrix.astype(np.float32))
            cam.distortion = torch.from_numpy(
                distCoeffs.astype(np.float32).flatten())
    # If there is more than one camera, also determine camera extrinsics.
    if len(cameras) >= 2:
        # Generate order in which we add cameras. Try to reduce cumulative error
        # by using the shortest-path tree on inverse of shared views.
        G = nx.Graph()
        for i in range(len(cameras)):
            for j in range(i):
                shared = sum(1 for pts in all_pts
                             if pts[i] is not None and pts[j] is not None)
                G.add_edge(i, j, weight=1 / shared)
        preds, _ = nx.dijkstra_predecessor_and_distance(G, source=0)
        tree = nx.Graph()
        for node, preds in preds.items():
            for pred in preds:
                tree.add_edge(pred, node)
        calib_order = list(nx.bfs_edges(tree, source=0))
        for i, j in calib_order:
            imgpts = [pts for pts in all_pts
                      if pts[i] is not None and pts[j] is not None]
            ptsi = [pts[i] for pts in imgpts]
            ptsj = [pts[j] for pts in imgpts]
            # Fix intrinsics since we optimized them already above.
            _, _, _, _, _, R, t, _, _ = cv2.stereoCalibrate(
                [objp] * len(imgpts), ptsi, ptsj,
                cameras[i].intrinsic.numpy(), cameras[i].distortion.numpy(),
                cameras[j].intrinsic.numpy(), cameras[j].distortion.numpy(),
                img_size, flags=cv2.CALIB_FIX_INTRINSIC
            )
            R = torch.from_numpy(R.astype(np.float32))
            t = torch.from_numpy(t.astype(np.float32).flatten())
            cameras[j].rotation = cameras[j].rotation @ R
            cameras[j].translation = R @ cameras[j].translation + t


def bundle_adjustment(
    cameras: list[Camera], all_pts: list[list[np.ndarray]], objp: np.ndarray,
    keep_intrinsics=False
):
    """ Refines camera extrinsics and frame-by-frame calibration board poses. """
    # Compute initial extrinsics and board pose estimates.
    camera_intr = []
    if not keep_intrinsics:
        for cam in cameras:
            Ki = cam.intrinsic.numpy()
            intr = np.array([Ki[0, 0], Ki[1, 1], Ki[0, 2], Ki[1, 2]])
            dist = cam.distortion.numpy()
            camera_intr.append(np.concat([intr, dist]))
    camera_extr = []
    for cam in cameras[1:]:
        rvec, _ = cv2.Rodrigues(cam.rotation.numpy())
        tvec = cam.translation.numpy()
        camera_extr.append(np.concat([rvec.flatten(), tvec]))
    board_params = []
    if keep_intrinsics:
        all_pts = [pts for pts in all_pts
                   if sum(1 for p in pts if p is not None) >= 2]
    for pts in all_pts:
        for i, cam in enumerate(cameras):
            if pts[i] is not None:
                Ki = cam.intrinsic.numpy()
                disti = cam.distortion.numpy()
                _, rvec, tvec = cv2.solvePnP(objp, pts[i], Ki, disti)
                R_l, _ = cv2.Rodrigues(rvec)
                Ri = cam.rotation.numpy()
                ti = cam.translation.numpy()
                R_w, t_w = Ri.T @ R_l, Ri.T @ (tvec.flatten() - ti)
                rvec_w, _ = cv2.Rodrigues(R_w)
                board_params.append(np.concat((rvec_w.flatten(), t_w)))
                break
    init_params = np.concat(
        camera_intr + camera_extr + board_params + [objp.flatten()])
    f_idx = [np.array([
        f for f, pts in enumerate(all_pts) if pts[i] is not None
    ]) for i in range(len(cameras))]
    pts2d_t = [np.concat([
        all_pts[f][i].reshape(objp.shape[0], 2) for f in f_idx[i]
    ]) for i in range(len(cameras))]

    def residual_fn(params):
        if keep_intrinsics:
            cam_extr_idx = 0
            cam_K = np.zeros((len(cameras), 3, 3))
            cam_dist = np.zeros((len(cameras), 5))
            for i, cam in enumerate(cameras):
                cam_K[i] = cam.intrinsic.numpy()
                cam_dist[i] = cam.distortion.numpy()
        else:
            cam_extr_idx = len(cameras) * 9
            cam_intr = params[:cam_extr_idx].reshape(-1, 9)
            cam_K = np.zeros((len(cameras), 3, 3))
            cam_K[:, 0, 0], cam_K[:, 1, 1] = cam_intr[:, 0], cam_intr[:, 1]
            cam_K[:, 0, 2], cam_K[:, 1, 2] = cam_intr[:, 2], cam_intr[:, 3]
            cam_K[:, 2, 2] = 1
            cam_dist = cam_intr[:, 4:]
        cam_end_idx = cam_extr_idx + (len(cameras) - 1) * 6
        cam_extr = np.zeros((len(cameras), 6))
        cam_extr[1:] = params[cam_extr_idx:cam_end_idx].reshape(-1, 6)
        board_p = params[cam_end_idx:-objp.size].reshape(-1, 6)
        obj_p = params[-objp.size:].reshape(objp.shape)
        obj_p[0], obj_p[-1] = objp[0], objp[-1]  # Fix two points for scale.
        R_b = np.stack([cv2.Rodrigues(r)[0] for r in board_p[:, :3]])
        pts3d = (obj_p @ R_b.mT) + board_p[:, None, 3:]
        residuals = []
        for i in range(len(cameras)):
            if len(f_idx[i]) == 0:
                continue
            cam3d = pts3d[f_idx[i]].reshape(-1, 3)
            pts2d_p, _ = cv2.projectPoints(
                cam3d, cam_extr[i, :3], cam_extr[i, 3:], cam_K[i], cam_dist[i])
            residuals.append((pts2d_p.reshape(-1, 2) - pts2d_t[i]).flatten())
        return np.concat(residuals)

    res = scipy.optimize.least_squares(residual_fn, init_params)
    rms = np.sqrt(np.mean(residual_fn(res.x)**2))
    print(f"Final RMS reprojection error: {rms:0.3f}.")
    if keep_intrinsics:
        cam_extr_idx = 0
    else:
        cam_extr_idx = len(cameras) * 9
        opt_cam_intr = res.x[:cam_extr_idx].reshape(-1, 9)
        for params, cam in zip(opt_cam_intr, cameras):
            K = np.zeros((3, 3), dtype=np.float32)
            K[0, 0], K[1, 1] = params[0], params[1]
            K[0, 2], K[1, 2] = params[2], params[3]
            K[2, 2] = 1
            dist = params[4:]
            cam.intrinsic = torch.from_numpy(K.astype(np.float32))
            cam.distortion = torch.from_numpy(dist.astype(np.float32))
    cam_end_idx = cam_extr_idx + (len(cameras) - 1) * 6
    opt_cam_extr = res.x[cam_extr_idx:cam_end_idx].reshape(-1, 6)
    for params, cam in zip(opt_cam_extr, cameras[1:]):
        rvec, tvec = params[:3], params[3:]
        R_mat, _ = cv2.Rodrigues(rvec)
        cam.rotation = torch.from_numpy(R_mat.astype(np.float32))
        cam.translation = torch.from_numpy(tvec.astype(np.float32).flatten())
    return cameras


if __name__ == "__main__":
    """
    Main execution function for the calibration script. This calibration script
    works using known fixed checkerboard patterns to accurately calibrate cameras.
    """
    parser = argparse.ArgumentParser(
        description="Camera calibration program.")
    parser.add_argument("urls", nargs="+",
                        help="URLs or paths to video sources")
    parser.add_argument("--cams", nargs="+",
                        help="Calibration files for cameras")
    parser.add_argument("--nrows", default=5, help="Rows of calibration grid")
    parser.add_argument("--ncols", default=7,
                        help="Columns of calibration grid")
    parser.add_argument("--csize", default=0.035,
                        help="Size of calibration grid cell in meters")
    parser.add_argument("--save-imgs",
                        help="Save all recorded images to this folder")
    parser.add_argument("--load-imgs",
                        help="Load recorded images to this folder")
    parser.add_argument("--keep-intrinsics", action="store_true",
                        help="Load intrinsics from files and use them")
    parser.add_argument("--no-flip", action="store_true",
                        help="Do not try to flip detections to align them")
    args = parser.parse_args()
    all_pts = []
    if args.load_imgs is not None:
        # Load existing frames from disk.
        iters = sorted(set(
            int(file[:file.index("_")]) for file in os.listdir(args.load_imgs)))
        for iter in iters:
            fst_frames = []
            img_pts = []
            for i in range(len(args.urls)):
                bgr_img = cv2.imread(f"{args.load_imgs}/{iter}_{i}.jpg")
                assert bgr_img is not None
                gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCornersSB(
                    gray_img, (args.nrows, args.ncols), None,
                    flags=cv2.CALIB_CB_ACCURACY + cv2.CALIB_CB_EXHAUSTIVE)
                if ret:
                    img_pts.append(corners)
                else:
                    img_pts.append(None)
                fst_frames.append(bgr_img)
            all_pts.append(img_pts)
    else:
        # Live record new frames.
        if all(os.path.isfile(u) for u in args.urls):
            source = OfflineVideoSource(args.urls)
        else:
            source = OnlineVideoSource(
                [int(s) if s.isdigit() else s for s in args.urls])
        source.start()
        # Wait some time to make sure all cameras are connected and get frames.
        time.sleep(1)
        ts, fst_frames, _ = source.next_frames()
        if ts is None or fst_frames is None:
            print("Failed to get initial frames.")
            exit(1)
        if args.save_imgs is not None:
            if os.path.exists(args.save_imgs):
                shutil.rmtree(args.save_imgs)
            os.makedirs(args.save_imgs, exist_ok=True)
        # Collect frames for calibration.
        record_next = False
        for iter in count():
            _, frames, _ = source.next_frames()
            if frames is None:
                break
            img_pts = []
            bgr_frames = []
            for frame in frames:
                bgr_img = cv2.cvtColor(frame.cpu().numpy(), cv2.COLOR_RGB2BGR)
                gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
                flags = cv2.CALIB_CB_ACCURACY + cv2.CALIB_CB_EXHAUSTIVE \
                    if record_next else 0
                ret, corners = cv2.findChessboardCornersSB(
                    gray_img, (args.nrows, args.ncols), None, flags=flags)
                if ret:
                    img_pts.append(corners)
                    cv2.drawChessboardCorners(
                        bgr_img, (args.nrows, args.ncols), corners, ret)
                else:
                    img_pts.append(None)
                bgr_frames.append(bgr_img)
            if record_next:
                all_pts.append(img_pts)
                if args.save_imgs is not None:
                    for i, frame in enumerate(frames):
                        frame = cv2.cvtColor(
                            frame.cpu().numpy(), cv2.COLOR_RGB2BGR)
                        cv2.imwrite(f"{args.save_imgs}/{iter}_{i}.jpg", frame)
            vis_frame = np.concat(bgr_frames)
            if vis_frame.shape[0] > 1000:
                width = round(vis_frame.shape[1] * 1000 / vis_frame.shape[0])
                vis_frame = cv2.resize(vis_frame, (width, 1000))
            cv2.imshow("Streams", vis_frame)
            last_ts = ts
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                record_next = True
            else:
                record_next = False
    all_pts = [pts for pts in all_pts if any(p is not None for p in pts)]
    objp = np.zeros((args.nrows * args.ncols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:args.nrows, 0:args.ncols].T.reshape(-1, 2)
    objp *= args.csize
    if not args.no_flip:
        # Try to align the order of points to be the same accross images.
        for pts in all_pts:
            avg_dir = np.zeros(2)
            for i, p in enumerate(pts):
                if p is not None:
                    dir = p[-1, 0] - p[0, 0]
                    if np.dot(dir, avg_dir) < 0:
                        pts[i] = np.flipud(p)
                        dir = -dir
                    avg_dir += dir
    cameras = [Camera() for _ in args.urls]
    height, width, _ = fst_frames[0].shape
    if args.keep_intrinsics:
        for file, cam in zip(args.cams or [], cameras):
            cam.load_file(file)
        multi_camera_calibrate(cameras, all_pts, objp, (width, height), True)
    else:
        multi_camera_calibrate(cameras, all_pts, objp, (width, height))
    bundle_adjustment(cameras, all_pts, objp, args.keep_intrinsics)
    # Write out the calibration parameters.
    for file, cam in zip(args.cams or [], cameras):
        with open(file, "w") as f:
            json.dump({
                "K": cam["K"], "distCoef": cam["distCoef"],
                "R": cam["R"], "t": cam["t"],
            }, f)
