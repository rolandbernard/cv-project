
import os
import json
import time
import argparse

import cv2
import numpy as np
import scipy.optimize as opt
import torch

import util
from camera import Camera
from detect import PoseDetector
from source import OfflineVideoSource, OnlineVideoSource
from tracker import (
    CrossViewFirstTracker,
    Tracker,
    build_constrained_physics,
    build_physics,
)


if __name__ == "__main__":
    """ Main execution function for the calibration script. """
    parser = argparse.ArgumentParser(
        description="Camera calibration program.")
    parser.add_argument("urls", nargs="+",
                        help="URLs or paths to video sources")
    parser.add_argument("--cams", nargs="+", required=True,
                        help="Calibration files for cameras")
    parser.add_argument("--nrows", default=10, help="Rows of calibration grid")
    parser.add_argument("--ncols", default=7,
                        help="Columns of calibration grid")
    parser.add_argument("--csize", default=0.015,
                        help="Size of calibration grid cell in meters")
    args = parser.parse_args()
    urls = args.urls
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
    record_next = False
    all_pts = []
    while True:
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
                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                )
                img_pts.append(corners)
                cv2.drawChessboardCorners(
                    bgr_img, (args.nrows, args.ncols), corners, ret)
            else:
                img_pts.append(None)
            bgr_frames.append(bgr_img)
        if record_next:
            all_pts.append(img_pts)
        cv2.imshow("Streams", np.concatenate(bgr_frames))
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
    cameras = []
    for i in range(len(urls)):
        height, width, _ = fst_frames[i].shape
        imgpts = [pts[i] for pts in all_pts if pts[i] is not None]
        ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(
            [objp]*len(imgpts), imgpts,
            (width, height), None, None  # type: ignore
        )
        print("RMS reprojection error:", ret)
        print("\nCamera Matrix:\n", cameraMatrix)
        print("\nDistortion Coefficients:\n", distCoeffs.ravel())
        cameras.append(Camera.from_dict({
            "K": cameraMatrix, "distCoef": distCoeffs
        }))
    if len(cameras) > 1:
        # If there is more than one camera, also determine relative pose.
        pass
    for file, cam in zip(args.cams, cameras):
        with open(file, "w") as f:
            json.dump({
                "K": cam["K"], "distCoef": cam["distCoef"],
                "R": cam["R"], "t": cam["t"],
            }, f)
