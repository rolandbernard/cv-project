import argparse
import os
import sys
import time

import cv2
import torch
import numpy as np
import pyvista as pv
import scipy.optimize as opt

from source import OnlineVideoSource, OfflineVideoSource
from camera import Camera
from tracker import Tracker, build_physics, build_constrained_physics
from detect import PoseDetector
import util

try:
    from moge.model.v2 import MoGeModel
    MOGE_AVAILABLE = True
except ImportError:
    MOGE_AVAILABLE = False

class LiveSkeletonPlayer:
    def __init__(self, cameras: list[Camera], center=(0, 0, 0), up=(0, -1, 0)):
        cam_dicts = []
        for cam in cameras:
            cam_dicts.append({"R": cam.rotation.tolist(), "t": cam.translation.tolist(), "K": cam.intrinsic.tolist()})
        self.scale = self.approx_scale(cam_dicts)
        self.track_meshes, self.track_actors = {}, {}
        self.pl = pv.Plotter(); self.pl.add_axes(); self.pl.set_background("white")
        lines = []
        for p1, p2 in util.SKELETON: lines.extend([2, p1, p2])
        self.skeleton = np.array(lines)
        self.setup_ground(center, up); self.setup_cameras(cam_dicts)
        self.pl.camera_position = [(center[0]+(2 if up[0]>=0 else -2)*self.scale, center[1]+(2 if up[1]>=0 else -2)*self.scale, center[2]+(2 if up[2]>=0 else -2)*self.scale), center, up]
        self.pl.show(interactive_update=True)

    def add_point_cloud(self, points, colors, point_size=2):
        poly = pv.PolyData(points)
        poly["colors"] = colors
        self.pl.add_mesh(poly, scalars="colors", rgb=True, point_size=point_size, render_points_as_spheres=True, opacity=0.6)

    def approx_scale(self, cameras):
        max_dist = 1.0
        for cam0 in cameras:
            for cam1 in cameras:
                center0 = -np.array(cam0["R"]).T @ np.array(cam0["t"]).flatten()
                center1 = -np.array(cam1["R"]).T @ np.array(cam1["t"]).flatten()
                dist = np.linalg.norm(center0 - center1).item()
                if dist > max_dist: max_dist = dist
        return max_dist

    def setup_ground(self, center, up):
        ground = pv.Plane(center=center, direction=up, i_size=self.scale * 2, j_size=self.scale * 2, i_resolution=20, j_resolution=20)
        self.pl.add_mesh(ground, style="wireframe", color="lightgray")

    def setup_cameras(self, cameras, scale=0.1):
        for cam in cameras:
            rotation, translate = np.array(cam["R"]), np.array(cam["t"]).flatten()
            camera_center = -rotation.T @ translate
            intrinsics = np.array(cam["K"])
            fx, fy = intrinsics[0, 0], intrinsics[1, 1]; cx, cy = intrinsics[0, 2], intrinsics[1, 2]
            width, height = cx * 2, cy * 2; z_cam = self.scale * scale
            x0 = (0 - cx) * z_cam / fx; x1 = (width - cx) * z_cam / fx
            y0 = (0 - cy) * z_cam / fy; y1 = (height - cy) * z_cam / fy
            corners_cam = np.array([[x0, y0, z_cam], [x1, y0, z_cam], [x1, y1, z_cam], [x0, y1, z_cam]])
            corners_world = (corners_cam @ rotation) + camera_center
            vertices = np.vstack([camera_center, corners_world])
            lines = np.array([[2, 0, 1], [2, 0, 2], [2, 0, 3], [2, 0, 4], [2, 1, 2], [2, 2, 3], [2, 3, 4], [2, 4, 1]]).flatten()
            self.pl.add_mesh(pv.PolyData(vertices, lines=lines), color="black", line_width=2)
            self.pl.add_mesh(pv.Sphere(radius=z_cam * 0.1, center=camera_center), color="red")

    def get_or_create_track(self, track_id):
        if track_id not in self.track_meshes:
            mesh = pv.PolyData(); color = util.COLORS[track_id % len(util.COLORS)]
            actor = self.pl.add_mesh(mesh, color=color, render_lines_as_tubes=True, line_width=8, render_points_as_spheres=True, point_size=15, smooth_shading=True)
            self.track_meshes[track_id], self.track_actors[track_id] = mesh, actor
        return self.track_meshes[track_id], self.track_actors[track_id]

    def update(self, tracks):
        for actor in self.track_actors.values(): actor.SetVisibility(False)
        for track in tracks:
            kpts = np.array(track["kpts"]); new_mesh = pv.PolyData(kpts); new_mesh.lines = self.skeleton
            mesh, actor = self.get_or_create_track(track["id"]); mesh.copy_from(new_mesh); actor.SetVisibility(True)
        self.pl.update()

def estimate_camera_params(img1, img2, distance=None, fov1=None, fov2=None):
    if distance is None: distance = 1.0 
    h1, w1 = img1.shape[:2]; h2, w2 = img2.shape[:2]
    cx1, cy1 = w1 / 2.0, h1 / 2.0; cx2, cy2 = w2 / 2.0, h2 / 2.0
    sift = cv2.SIFT_create(nfeatures=10000, contrastThreshold=0.01, edgeThreshold=15)
    kp1, des1 = sift.detectAndCompute(cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY), None)
    kp2, des2 = sift.detectAndCompute(cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY), None)
    if des1 is None or des2 is None: raise RuntimeError("No features found.")
    def rootsift(des):
        if des is None: return None
        des /= (des.sum(axis=1, keepdims=True) + 1e-7); return np.sqrt(des)
    des1, des2 = rootsift(des1), rootsift(des2)
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=100))
    raw_matches = flann.knnMatch(des1, des2, k=2)
    matches = [m for m, n in raw_matches if m.distance < 0.75 * n.distance]
    if len(matches) < 15: matches = [m for m, n in raw_matches if m.distance < 0.85 * n.distance]
    if len(matches) < 8: raise RuntimeError("Not enough feature matches.")
    pts1, pts2 = np.float32([kp1[m.queryIdx].pt for m in matches]), np.float32([kp2[m.trainIdx].pt for m in matches])
    fm_method = getattr(cv2, "USAC_MAGSAC", cv2.FM_RANSAC)
    _, mask = cv2.findFundamentalMat(pts1, pts2, fm_method, 1.0, 0.9999, 50000)
    if mask is None: raise RuntimeError("Fundamental matrix estimation failed.")
    inliers1, inliers2 = pts1[mask.ravel() == 1], pts2[mask.ravel() == 1]
    def epipolar_loss(params):
        f1, cx1, cy1, f2, cx2, cy2 = params
        K1_c = np.array([[f1, 0, cx1], [0, f1, cy1], [0, 0, 1]], dtype=np.float32)
        K2_c = np.array([[f2, 0, cx2], [0, f2, cy2], [0, 0, 1]], dtype=np.float32)
        p1_n = cv2.undistortPoints(np.expand_dims(inliers1, 1), K1_c, None)
        p2_n = cv2.undistortPoints(np.expand_dims(inliers2, 1), K2_c, None)
        E_c, _ = cv2.findEssentialMat(p1_n, p2_n, np.eye(3), method=cv2.FM_8POINT)
        if E_c is None or E_c.shape != (3, 3): return 1e10
        F_px = np.linalg.inv(K2_c).T @ E_c @ np.linalg.inv(K1_c)
        pts1_h = np.column_stack([inliers1, np.ones(len(inliers1))])
        pts2_h = np.column_stack([inliers2, np.ones(len(inliers2))])
        l2 = (F_px @ pts1_h.T).T; l1 = (F_px.T @ pts2_h.T).T
        alg_err = np.sum(pts2_h * l2, axis=1)
        dist_err = (alg_err**2) / (l2[:, 0]**2 + l2[:, 1]**2 + 1e-8) + (alg_err**2) / (l1[:, 0]**2 + l1[:, 1]**2 + 1e-8)
        return np.mean(dist_err)
    f1 = (np.sqrt(w1**2+h1**2)/(2*np.tan(np.radians(fov1)/2))) if fov1 else 1.17 * max(h1, w1)
    f2 = (np.sqrt(w2**2+h2**2)/(2*np.tan(np.radians(fov2)/2))) if fov2 else 1.17 * max(h2, w2)
    cx1, cy1 = w1 / 2.0, h1 / 2.0; cx2, cy2 = w2 / 2.0, h2 / 2.0
    if fov1 is None or fov2 is None:
        print("Optimizing independent intrinsics...")
        res = opt.minimize(epipolar_loss, x0=[f1, cx1, cy1, f2, cx2, cy2], bounds=[(0.5*w1, 3*w1), (w1*0.4, w1*0.6), (h1*0.4, h1*0.6), (0.5*w2, 3*w2), (w2*0.4, w2*0.6), (h2*0.4, h2*0.6)], method='Nelder-Mead')
        f1, cx1, cy1, f2, cx2, cy2 = res.x
    K1, K2 = np.array([[f1, 0, cx1], [0, f1, cy1], [0, 0, 1]], dtype=np.float32), np.array([[f2, 0, cx2], [0, f2, cy2], [0, 0, 1]], dtype=np.float32)
    p1_n = cv2.undistortPoints(np.expand_dims(inliers1, 1), K1, None); p2_n = cv2.undistortPoints(np.expand_dims(inliers2, 1), K2, None)
    E, final_mask = cv2.findEssentialMat(p1_n, p2_n, np.eye(3), method=cv2.FM_8POINT)
    _, R, t, _ = cv2.recoverPose(E, p1_n, p2_n, np.eye(3), mask=final_mask)
    def get_ba_errors(params, pts1, pts2):
        f1, cx1, cy1, f2, cx2, cy2, rvec, curr_t = params[0], params[1], params[2], params[3], params[4], params[5], params[6:9], params[9:12]
        K1 = np.array([[f1, 0, cx1], [0, f1, cy1], [0, 0, 1]], dtype=np.float32); K2 = np.array([[f2, 0, cx2], [0, f2, cy2], [0, 0, 1]], dtype=np.float32)
        curr_R, _ = cv2.Rodrigues(rvec); P1 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1)))); P2 = K2 @ np.hstack((curr_R, curr_t.reshape(3, 1)))
        pts4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T); pts3d = (pts4d[:3] / (pts4d[3] + 1e-7)).T
        imgpts1, _ = cv2.projectPoints(pts3d, np.zeros(3), np.zeros(3), K1, None); imgpts2, _ = cv2.projectPoints(pts3d, rvec, curr_t, K2, None)
        return np.concatenate([(pts1 - imgpts1.reshape(-1, 2)).flatten(), (pts2 - imgpts2.reshape(-1, 2)).flatten()])
    rvec_init, _ = cv2.Rodrigues(R)
    res_ba = opt.least_squares(get_ba_errors, np.concatenate([[f1, cx1, cy1, f2, cx2, cy2], rvec_init.flatten(), t.flatten()]), args=(inliers1, inliers2), ftol=1e-4)
    f1, cx1, cy1, f2, cx2, cy2 = res_ba.x[0:6]; R, _ = cv2.Rodrigues(res_ba.x[6:9]); t = res_ba.x[9:12]
    t = (t / (np.linalg.norm(t) + 1e-7)) * distance
    K1, K2 = np.array([[f1, 0, cx1], [0, f1, cy1], [0, 0, 1]], dtype=np.float32), np.array([[f2, 0, cx2], [0, f2, cy2], [0, 0, 1]], dtype=np.float32)
    return K1, K2, R, t

def estimate_camera_params_moge(img1, img2, distance=None):
    if not MOGE_AVAILABLE: raise RuntimeError("MoGe-2 not found.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading MoGe-2..."); model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(device); model.eval()
    def infer(img):
        t = torch.tensor(img / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)
        with torch.no_grad(): return model.infer(t)
    out1, out2 = infer(img1), infer(img2); h1, w1 = img1.shape[:2]; h2, w2 = img2.shape[:2]
    K1 = out1["intrinsics"].cpu().numpy(); K1[0,:] *= w1; K1[1,:] *= h1
    K2 = out2["intrinsics"].cpu().numpy(); K2[0,:] *= w2; K2[1,:] *= h2
    sift = cv2.SIFT_create(nfeatures=5000); kp1, des1 = sift.detectAndCompute(cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY), None); kp2, des2 = sift.detectAndCompute(cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY), None)
    bf = cv2.BFMatcher(); ms = bf.knnMatch(des1, des2, k=2); gms = [m for m, n in ms if m.distance < 0.7 * n.distance]
    p1_3d, p2_3d, pm1, pm2 = [], [], out1["points"].cpu().numpy(), out2["points"].cpu().numpy()
    mk1, mk2 = out1["mask"].cpu().numpy(), out2["mask"].cpu().numpy()
    for m in gms:
        u1, v1 = map(int, kp1[m.queryIdx].pt); u2, v2 = map(int, kp2[m.trainIdx].pt)
        if mk1[v1, u1] > 0.5 and mk2[v2, u2] > 0.5: p1_3d.append(pm1[v1, u1]); p2_3d.append(pm2[v2, u2])
    p1_3d, p2_3d = np.array(p1_3d), np.array(p2_3d); best_R, best_t, bi = np.eye(3), np.zeros(3), []
    for _ in range(500):
        idx = np.random.choice(len(p1_3d), 3, replace=False); s, d = p2_3d[idx], p1_3d[idx]; cs, cd = np.mean(s, 0), np.mean(d, 0)
        H = (s - cs).T @ (d - cd); U, S, Vt = np.linalg.svd(H); R = Vt.T @ U.T
        if np.linalg.det(R) < 0: Vt[2,: ]*=-1; R = Vt.T @ U.T
        t = cd - R @ cs; df = p1_3d - (p2_3d @ R.T + t); i = np.where(np.sum(df**2, 1) < 0.01)[0]
        if len(i) > len(bi): bi, best_R, best_t = i, R, t
    if len(bi) >= 3:
        s, d = p2_3d[bi], p1_3d[bi]; cs, cd = np.mean(s, 0), np.mean(d, 0); H = (s - cs).T @ (d - cd); U, S, Vt = np.linalg.svd(H); best_R = Vt.T @ U.T
        if np.linalg.det(best_R) < 0: Vt[2,:]*=-1; best_R = Vt.T @ U.T
        best_t = cd - best_R @ cs
    cb = np.linalg.norm(best_t)
    if distance is not None and cb > 1e-6: scl = distance / cb; best_t *= scl; pm1 *= scl; pm2 *= scl; print(f"MoGe Baseline: {cb:.2f}m. Scaled to user distance: {distance:.2f}m.")
    else: print(f"Using MoGe metric baseline: {cb:.2f}m.")
    step = 4; pts1_v, clrs1_v = pm1[::step, ::step].reshape(-1, 3), img1[::step, ::step].reshape(-1, 3)
    pts2_v, clrs2_v = pm2[::step, ::step].reshape(-1, 3), img2[::step, ::step].reshape(-1, 3)
    return K1, K2, best_R, best_t, (pts1_v, clrs1_v), (pts2_v, clrs2_v)

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("url1"); parser.add_argument("url2"); parser.add_argument("--distance", type=float); parser.add_argument("--fov1", type=float); parser.add_argument("--fov2", type=float); parser.add_argument("--resize", type=int, nargs=2); parser.add_argument("--moge", action="store_true"); args = parser.parse_args()
    urls = [args.url1, args.url2]; is_offline = all(os.path.isfile(u) for u in urls); resize = tuple(args.resize) if args.resize else None; source = OfflineVideoSource(urls, resize=resize) if is_offline else OnlineVideoSource(urls, resize=resize)
    source.start(); print("Waiting for frames..."); ts, frames, _ = None, None, None
    for _ in range(100):
        ts, frames, _ = source.next_frames()
        if ts is not None: break
        time.sleep(0.1)
    if ts is None: return
    img1, img2 = frames[0].cpu().numpy(), frames[1].cpu().numpy(); print("Estimating parameters..."); clouds = None
    try:
        if args.moge: 
            K1, K2, R, t, cloud1, cloud2 = estimate_camera_params_moge(img1, img2, args.distance)
            pts2_world = (R @ cloud2[0].T + t.reshape(3, 1)).T
            clouds = [(cloud1[0], cloud1[1]), (pts2_world, cloud2[1])]
        else: K1, K2, R, t = estimate_camera_params(img1, img2, args.distance, args.fov1, args.fov2)
    except Exception as e: print(f"Error: {e}"); source.release(); return
    print(f"K1:\n{K1}\nK2:\n{K2}\nR:\n{R}\nt:\n{t}"); cam1 = Camera(rotation=torch.eye(3), translation=torch.zeros(3), intrinsic=torch.from_numpy(K1).float()); cam2 = Camera(rotation=torch.from_numpy(R.T).float(), translation=torch.from_numpy(-R.T @ t.flatten()).float(), intrinsic=torch.from_numpy(K2).float())
    cameras = [cam1, cam2]; source.cameras = cameras; detector = PoseDetector(path="./nets"); physics = build_constrained_physics(scale=1.0)
    if torch.cuda.is_available(): detector.to("cuda"); cam1.to("cuda"); cam2.to("cuda"); physics.to("cuda")
    tracker = Tracker(detector, physics); player = LiveSkeletonPlayer(cameras)
    if clouds:
        for pts, clrs in clouds: player.add_point_cloud(pts, clrs)
    last_ts = ts
    try:
        while True:
            ts, frames, _ = source.next_frames()
            if ts is None: break
            dt = ts - last_ts
            if dt <= 0: continue
            tracker.predict(dt); tracker.update(cameras, frames); player.update([{"id": t.id, "kpts": t.get_keypoints().tolist()} for t in tracker.get_prediction()])
            stacked = np.vstack([cv2.cvtColor(f.cpu().numpy(), cv2.COLOR_RGB2BGR) for f in frames])
            if stacked.shape[0] > 800: stacked = cv2.resize(stacked, (int(stacked.shape[1]*800/stacked.shape[0]), 800))
            cv2.imshow("Streams", stacked); last_ts = ts
            if cv2.waitKey(1) & 0xFF == ord('q'): break
    finally: source.release(); cv2.destroyAllWindows()

if __name__ == "__main__": main()
