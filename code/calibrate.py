
import os
import json
import time
import argparse
from itertools import count

import cv2
import numpy as np
import torch
import torch.nn as nn

import util
from camera import Camera
from source import OfflineVideoSource, OnlineVideoSource


if __name__ == "__main__":
    """
    Main execution function for the calibration script. This calibration script
    works using known fixed checkerboard patterns to accurately calibrate cameras.
    """
    parser = argparse.ArgumentParser(
        description="Camera calibration program.")
    parser.add_argument("urls", nargs="+",
                        help="URLs or paths to video sources")
    parser.add_argument("--cams", nargs="+", required=True,
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
    args = parser.parse_args()
    urls = args.urls
    all_pts = []
    if args.load_imgs is not None:
        # Load existing frames from disk.
        iters = set(
            int(file[:file.index("_")]) for file in os.listdir(args.load_imgs))
        for iter in iters:
            fst_frames = []
            img_pts = []
            for i in range(len(urls)):
                bgr_img = cv2.imread(f"{args.load_imgs}/{iter}_{i}.jpg")
                assert bgr_img is not None
                gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(
                    gray_img, (args.nrows, args.ncols), None)
                if ret:
                    corners = cv2.cornerSubPix(
                        gray_img, corners, (11, 11), (-1, -1),
                        (cv2.TERM_CRITERIA_EPS +
                         cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                    )
                    img_pts.append(corners)
                    cv2.drawChessboardCorners(
                        bgr_img, (args.nrows, args.ncols), corners, ret)
                else:
                    img_pts.append(None)
                fst_frames.append(bgr_img)
            all_pts.append(img_pts)
    else:
        # Live record new frames.
        is_offline = all(os.path.isfile(u) for u in urls)
        if is_offline:
            source = OfflineVideoSource(urls)
        else:
            source = OnlineVideoSource(urls)
        source.start()
        # Wait some time to make sure all cameras are connected and get frames.
        print("Waiting for frames...")
        time.sleep(1)
        ts, fst_frames, _ = source.next_frames()
        if ts is None or fst_frames is None:
            print("Failed to get initial frames.")
            exit(1)
        if args.save_imgs is not None:
            if os.path.exists(args.save_imgs):
                os.removedirs(args.save_imgs)
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
                ret, corners = cv2.findChessboardCorners(
                    gray_img, (args.nrows, args.ncols), None)
                if ret:
                    corners = cv2.cornerSubPix(
                        gray_img, corners, (11, 11), (-1, -1),
                        (cv2.TERM_CRITERIA_EPS +
                         cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                    )
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
            cv2.imshow("Streams", np.concat(bgr_frames))
            last_ts = ts
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                record_next = True
            else:
                record_next = False
    objp = np.zeros((args.nrows * args.ncols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:args.nrows, 0:args.ncols].T.reshape(-1, 2)
    objp *= args.csize
    # Compute camera intrinsics for each camera individually.
    cameras: list[Camera] = []
    for i in range(len(urls)):
        height, width, _ = fst_frames[i].shape
        imgpts = [pts[i] for pts in all_pts if pts[i] is not None]
        ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(
            [objp]*len(imgpts), imgpts,
            (width, height), None, None  # type: ignore
        )
        cameras.append(Camera.from_dict({
            "K": cameraMatrix.astype(np.float32),
            "distCoef": distCoeffs.astype(np.float32).flatten()
        }))
    # If there is more than one camera, also determine camera extrinsics.
    if len(cameras) > 1:
        for i in range(1, len(cameras)):
            imgpts = [pts for pts in all_pts
                      if pts[0] is not None and pts[i] is not None]
            pts0 = np.array([pts[0] for pts in imgpts]).reshape(-1, 2)
            ptsi = np.array([pts[i] for pts in imgpts]).reshape(-1, 2)
            _, _, R, t, _ = cv2.recoverPose(
                pts0, ptsi,
                cameras[0].intrinsic.numpy(), cameras[0].distortion.numpy(),
                cameras[i].intrinsic.numpy(), cameras[i].distortion.numpy(),
            )
            cameras[i].rotation = torch.from_numpy(R.astype(np.float32))
            cameras[i].translation = torch.from_numpy(
                t.astype(np.float32).flatten())
            # Scale translation based on triangulated point distances.
            proj0 = cameras[0].proj_matrix().numpy()
            proji = cameras[i].proj_matrix().numpy()
            pts4d = cv2.triangulatePoints(proj0, proji, pts0.T, ptsi.T).T
            pts3d = (pts4d[:, :3] / pts4d[:, 3, None]) \
                .reshape(-1, args.nrows, args.ncols, 3)
            dist_h = np.mean(np.linalg.norm(
                pts3d[:, :-1, :] - pts3d[:, 1:, :], axis=-1))
            dist_v = np.mean(np.linalg.norm(
                pts3d[:, :, :-1] - pts3d[:, :, 1:], axis=-1))
            scale = (2 * args.csize) / (dist_h + dist_v)
            cameras[i].translation *= scale
    # Write out the calibration parameters.
    for file, cam in zip(args.cams, cameras):
        with open(file, "w") as f:
            json.dump({
                "K": cam["K"], "distCoef": cam["distCoef"],
                "R": cam["R"], "t": cam["t"],
            }, f)
