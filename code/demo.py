
import argparse
import os
import time

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
from visualize import LiveSkeletonPlayer

try:
    from moge.model.v2 import MoGeModel
    MOGE_AVAILABLE = True
except ImportError:
    MOGE_AVAILABLE = False


def rootsift(des):
    if des is None:
        return None
    des /= (des.sum(axis=1, keepdims=True) + 1e-7)
    return np.sqrt(des)


def estimate_camera_params(img1, img2, distance=None):
    """ Estimate intrinsic and extrinsic camera parameters from two images. """
    if distance is None:
        distance = 1.0
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    cx1, cy1 = w1 / 2.0, h1 / 2.0
    cx2, cy2 = w2 / 2.0, h2 / 2.0
    # Extract features from the images
    sift = cv2.SIFT_create(  # type: ignore
        nfeatures=10000, contrastThreshold=0.01, edgeThreshold=15)
    kp1, des1 = sift.detectAndCompute(
        cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY), None)
    kp2, des2 = sift.detectAndCompute(
        cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY), None)
    if des1 is None or des2 is None:
        print("No features found.")
        exit(1)
    des1, des2 = rootsift(des1), rootsift(des2)
    # Perform matching using features
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=100))
    ms = flann.knnMatch(des1, des2, k=2)  # type: ignore
    gms = [m for m, n in ms if m.distance < 0.75 * n.distance]
    if len(gms) < 15:
        gms = [m for m, n in ms if m.distance < 0.85 * n.distance]
    if len(gms) < 8:
        print("Not enough feature matches.")
        exit(1)
    pts1 = np.array([kp1[m.queryIdx].pt for m in gms])
    pts2 = np.array([kp2[m.trainIdx].pt for m in gms])
    # Filter out outliers using fundamental matrix
    _, mask = cv2.findFundamentalMat(
        pts1, pts2, cv2.USAC_MAGSAC, 1.0, 0.9999, 50000)
    if mask is None:
        print("Fundamental matrix estimation failed.")
        exit(1)
    inliers1, inliers2 = pts1[mask.ravel() == 1], pts2[mask.ravel() == 1]

    # Optimize focal lengths and center points
    def epipolar_loss(params):
        f1_c, cx1_c, cy1_c, f2_c, cx2_c, cy2_c = params
        K1_c = np.array([[f1_c, 0, cx1_c], [0, f1_c, cy1_c], [0, 0, 1]])
        K2_c = np.array([[f2_c, 0, cx2_c], [0, f2_c, cy2_c], [0, 0, 1]])
        p1_n = cv2.undistortPoints(np.expand_dims(inliers1, 1), K1_c, None)
        p2_n = cv2.undistortPoints(np.expand_dims(inliers2, 1), K2_c, None)
        E_c, _ = cv2.findEssentialMat(
            p1_n, p2_n, np.eye(3), method=cv2.FM_8POINT)
        if E_c is None or E_c.shape != (3, 3):
            return 1e10
        F_px = np.linalg.inv(K2_c).T @ E_c @ np.linalg.inv(K1_c)
        pts1_h = np.column_stack([inliers1, np.ones(len(inliers1))])
        pts2_h = np.column_stack([inliers2, np.ones(len(inliers2))])
        l2 = (F_px @ pts1_h.T).T
        l1 = (F_px.T @ pts2_h.T).T
        alg_err = np.sum(pts2_h * l2, axis=1)
        dist_err = (alg_err**2) / (l2[:, 0]**2 + l2[:, 1]**2 + 1e-8) + \
                   (alg_err**2) / (l1[:, 0]**2 + l1[:, 1]**2 + 1e-8)
        return np.mean(dist_err)

    f1 = 1.25 * max(h1, w1)
    f2 = 1.25 * max(h2, w2)
    print("Optimizing intrinsics...")
    res = opt.minimize(
        epipolar_loss,
        x0=[f1, cx1, cy1, f2, cx2, cy2],
        bounds=[
            (0.5 * w1, 3 * w1), (w1 * 0.4, w1 * 0.6), (h1 * 0.4, h1 * 0.6),
            (0.5 * w2, 3 * w2), (w2 * 0.4, w2 * 0.6), (h2 * 0.4, h2 * 0.6)
        ],
        method='Nelder-Mead'
    )
    f1, cx1, cy1, f2, cx2, cy2 = res.x
    K1 = np.array([[f1, 0, cx1], [0, f1, cy1], [0, 0, 1]])
    K2 = np.array([[f2, 0, cx2], [0, f2, cy2], [0, 0, 1]])
    p1_n = cv2.undistortPoints(np.expand_dims(inliers1, 1), K1, None)
    p2_n = cv2.undistortPoints(np.expand_dims(inliers2, 1), K2, None)
    E, final_mask = cv2.findEssentialMat(
        p1_n, p2_n, np.eye(3), method=cv2.FM_8POINT)
    _, R, t, _ = cv2.recoverPose(E, p1_n, p2_n, np.eye(3), mask=final_mask)

    # Perform Bundle Adjustment to optimize further
    def get_ba_errors(params, pts1, pts2):
        f1_ba, cx1_ba, cy1_ba, f2_ba, cx2_ba, cy2_ba = params[0:6]
        rvec, curr_t = params[6:9], params[9:12]
        K1_ba = np.array([[f1_ba, 0, cx1_ba], [0, f1_ba, cy1_ba], [0, 0, 1]])
        K2_ba = np.array([[f2_ba, 0, cx2_ba], [0, f2_ba, cy2_ba], [0, 0, 1]])
        curr_R, _ = cv2.Rodrigues(rvec)
        P1 = K1_ba @ np.hstack((np.eye(3), np.zeros((3, 1))))
        P2 = K2_ba @ np.hstack((curr_R, curr_t.reshape(3, 1)))
        pts4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        pts3d = (pts4d[:3] / (pts4d[3] + 1e-7)).T
        imgpts1, _ = cv2.projectPoints(
            pts3d, np.zeros(3), np.zeros(3), K1_ba, None)
        imgpts2, _ = cv2.projectPoints(pts3d, rvec, curr_t, K2_ba, None)
        err1 = (pts1 - imgpts1.reshape(-1, 2)).flatten()
        err2 = (pts2 - imgpts2.reshape(-1, 2)).flatten()
        return np.concatenate([err1, err2])

    print("Final Bundle Adjustment...")
    rvec_init, _ = cv2.Rodrigues(R)
    res_ba = opt.least_squares(
        get_ba_errors,
        np.concatenate([[
            f1, cx1, cy1, f2, cx2, cy2], rvec_init.flatten(), t.flatten()]),
        args=(inliers1, inliers2), ftol=1e-4
    )
    f1, cx1, cy1, f2, cx2, cy2 = res_ba.x[0:6]
    R, _ = cv2.Rodrigues(res_ba.x[6:9])
    t = res_ba.x[9:12]
    t = (t / (np.linalg.norm(t) + 1e-7)) * distance
    K1 = np.array([[f1, 0, cx1], [0, f1, cy1], [0, 0, 1]], dtype=np.float32)
    K2 = np.array([[f2, 0, cx2], [0, f2, cy2], [0, 0, 1]], dtype=np.float32)
    return K1, K2, R, t


def estimate_camera_params_moge(img1, img2, distance=None):
    """ Estimate camera parameters using MoGe-2 model. """
    if not MOGE_AVAILABLE:
        print("MoGe-2 not found.")
        exit(1)
    # Generate intrinsic estimates and point cloud using MoGe-2
    print("Loading MoGe-2...")
    model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal")
    model.to(util.DEVICE)
    model.eval()
    p_img1 = (torch.from_numpy(img1).to(torch.float32) / 255.0) \
        .permute(2, 0, 1).to(util.DEVICE)
    p_img2 = (torch.from_numpy(img2).to(torch.float32) / 255.0) \
        .permute(2, 0, 1).to(util.DEVICE)
    with torch.inference_mode():
        out1, out2 = model.infer(p_img1), model.infer(p_img2)
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    K1 = out1["intrinsics"].cpu().numpy()
    K1[0, :] *= w1
    K1[1, :] *= h1
    K2 = out2["intrinsics"].cpu().numpy()
    K2[0, :] *= w2
    K2[1, :] *= h2
    # Find matching features in the two images
    print("Computing initial camera parameters...")
    sift = cv2.SIFT_create(  # type: ignore
        nfeatures=10000, contrastThreshold=0.01, edgeThreshold=15)
    kp1, des1 = sift.detectAndCompute(
        cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY), None)
    kp2, des2 = sift.detectAndCompute(
        cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY), None)
    if des1 is None or des2 is None:
        print("No features found.")
        exit(1)
    des1, des2 = rootsift(des1), rootsift(des2)
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=100))
    ms = flann.knnMatch(des1, des2, k=2)  # type: ignore
    gms = [m for m, n in ms if m.distance < 0.75 * n.distance]
    if len(gms) < 15:
        gms = [m for m, n in ms if m.distance < 0.85 * n.distance]
    if len(gms) < 8:
        print("Not enough feature matches.")
        exit(1)
    # Collect predicted 3D location for all matches if seen as valid
    p1_3d, p2_3d = [], []
    pm1, pm2 = out1["points"].cpu().numpy(), out2["points"].cpu().numpy()
    mk1, mk2 = out1["mask"].cpu().numpy(), out2["mask"].cpu().numpy()
    for m in gms:
        u1, v1 = map(int, kp1[m.queryIdx].pt)
        u2, v2 = map(int, kp2[m.trainIdx].pt)
        if mk1[v1, u1] > 0.5 and mk2[v2, u2] > 0.5:
            p1_3d.append(pm1[v1, u1])
            p2_3d.append(pm2[v2, u2])
    p1_3d, p2_3d = np.array(p1_3d), np.array(p2_3d)
    print(f"Found {p1_3d.shape[0]} matches.")
    # Randomized algorithm to find inliers
    best_R, best_t, bi = np.eye(3), np.zeros(3), []
    for _ in range(500):
        idx = np.random.choice(len(p1_3d), 3, replace=False)
        s, d = p2_3d[idx], p1_3d[idx]
        cs, cd = np.mean(s, 0), np.mean(d, 0)
        H = (s - cs).T @ (d - cd)
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = Vt.T @ U.T
        t_vec = cd - R @ cs
        df = p1_3d - (p2_3d @ R.T + t_vec)
        inliers = np.where(np.sum(df**2, 1) < 0.1)[0]
        if len(inliers) > len(bi):
            bi, best_R, best_t = inliers, R, t_vec
    print(f"Found {len(bi)} inliners.")
    # Compute final rotation and translation using inliers
    s, d = p2_3d[bi], p1_3d[bi]
    cs, cd = np.mean(s, 0), np.mean(d, 0)
    H = (s - cs).T @ (d - cd)
    U, _, Vt = np.linalg.svd(H)
    best_R = Vt.T @ U.T
    if np.linalg.det(best_R) < 0:
        Vt[2, :] *= -1
        best_R = Vt.T @ U.T
    best_t = cd - best_R @ cs
    cb = np.linalg.norm(best_t)
    print(f"MoGe metric baseline: {cb:.2f}m.")
    if distance is not None and cb > 1e-6:
        # Scale distance between cameras to used defined value
        scl = distance / cb
        best_t *= scl
        pm1 *= scl
        pm2 *= scl
    pts1_v, clrs1_v = pm1.reshape(-1, 3), img1.reshape(-1, 3)
    pts2_v, clrs2_v = pm2.reshape(-1, 3), img2.reshape(-1, 3)
    return K1, K2, best_R, best_t, (pts1_v, clrs1_v), (pts2_v, clrs2_v)


if __name__ == "__main__":
    """ Main execution function for the demo. """
    parser = argparse.ArgumentParser(
        description="Cross-view 3D skeleton tracking demo.")
    parser.add_argument("url1", help="URL or path to the first video source")
    parser.add_argument("url2", help="URL or path to the second video source")
    parser.add_argument("--distance", type=float,
                        help="Known distance between cameras for scaling")
    parser.add_argument("--resize", type=int, nargs=2,
                        help="Resize frames to (width, height)")
    parser.add_argument("--moge", action="store_true",
                        help="Use MoGe-2 for camera estimation")
    parser.add_argument("--no-constraint", action="store_true",
                        help="Do not use rigid body constraints")
    parser.add_argument("--cross-first", action="store_true",
                        help="Match using cross-view association first")
    args = parser.parse_args()
    urls = [args.url1, args.url2]
    is_offline = all(os.path.isfile(u) for u in urls)
    resize = tuple(args.resize) if args.resize else None
    if is_offline:
        source = OfflineVideoSource(urls, resize=resize)
    else:
        source = OnlineVideoSource(urls, resize=resize)
    source.start()
    print("Waiting for frames...")
    ts, frames = None, None
    for _ in range(10):
        ts, frames, _ = source.next_frames()
        if ts is not None:
            break
        time.sleep(0.1)
    if ts is None or frames is None:
        print("Failed to get initial frames.")
        exit(1)
    img1, img2 = frames[0].cpu().numpy(), frames[1].cpu().numpy()
    print("Estimating parameters...")
    clouds = None
    if args.moge:
        K1, K2, R, t, cloud1, cloud2 = estimate_camera_params_moge(
            img1, img2, args.distance)
        pts2_world = (R @ cloud2[0].T + t.reshape(3, 1)).T
        clouds = [(cloud1[0], cloud1[1]), (pts2_world, cloud2[1])]
    else:
        K1, K2, R, t = estimate_camera_params(img1, img2, args.distance)
    print(f"K1:\n{K1}\nK2:\n{K2}\nR:\n{R}\nt:\n{t}")
    cam1 = Camera(
        rotation=torch.eye(3),
        translation=torch.zeros(3),
        intrinsic=torch.from_numpy(K1).float()
    )
    cam2 = Camera(
        rotation=torch.from_numpy(R.T).float(),
        translation=torch.from_numpy(-R.T @ t.flatten()).float(),
        intrinsic=torch.from_numpy(K2).float()
    )
    cameras = [cam1, cam2]
    source.cameras = cameras
    source.to(util.DEVICE)
    detector = PoseDetector()
    detector.to(util.DEVICE)
    physics = build_physics(1.0) \
        if args.no_constraint else build_constrained_physics(1.0)
    physics.to(util.DEVICE)
    tracker_cls = CrossViewFirstTracker if args.cross_first else Tracker
    tracker = tracker_cls(detector, physics)
    player = LiveSkeletonPlayer(cameras)
    if clouds is not None:
        for pts, clrs in clouds:
            player.add_point_cloud(pts, clrs)
    last_ts = ts
    while True:
        ts, frames, _ = source.next_frames()
        if ts is None or frames is None:
            break
        dt = ts - last_ts
        if dt <= 0:
            # We don't have any new frames available.
            time.sleep(0.01)
            continue
        tracker.predict(dt)
        tracker.update(cameras, frames)
        player.update(tracker.get_prediction())
        stacked = np.vstack(
            [cv2.cvtColor(f.cpu().numpy(), cv2.COLOR_RGB2BGR) for f in frames])
        if stacked.shape[0] > 800:
            # Cap height to 800 pixels.
            h_new, w_new = 800, int(stacked.shape[1] * 800 / stacked.shape[0])
            stacked = cv2.resize(stacked, (w_new, h_new))
        cv2.imshow("Streams", stacked)
        last_ts = ts
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
