
import time
import copy

import torch
import scipy.optimize
from torch import Tensor

import util
import camera
import kalman
import source
from camera import Camera
from detect import PoseDetector
from kalman import LinearPhysics, ConstrainedPhysics


class Track:
    """
    This is a simple implementation of the track in which the state is made up
    of a vector in which the first elements form the coordinates of keypoints and
    a matrix corresponding to the complete covariance matrix. This  type of track
    does not include history.
    """

    def __init__(self, id: int, init_mean: torch.Tensor, init_cov: torch.Tensor, num_keypoint=17, num_dim=3):
        self.id = id
        self.num_detection = 0
        self.last_detection = 0
        self.num_keypoint = num_keypoint
        self.num_dim = num_dim
        self.moved(init_mean, init_cov)

    def moved(self, mean: torch.Tensor, cov: torch.Tensor):
        """ Update the track to the new state (from a prediction). """
        self.mean = mean
        self.cov = cov

    def update(self, mean: torch.Tensor, cov: torch.Tensor):
        """ Update the track to the new state (from a detection). """
        self.moved(mean, cov)
        self.last_detection = 0
        self.num_detection += 1

    def no_update(self):
        """ Record that there was no update for this track for one update cycle. """
        self.last_detection += 1

    def get_keypoints(self) -> torch.Tensor:
        """
        Get the keypoints for this track. The keypoints should be derived from
        the internal state of the track in some implementation defined way.
        """
        tot = self.num_keypoint*self.num_dim
        return self.mean[:tot].view(-1, self.num_dim)

    def get_full_covariances(self) -> torch.Tensor:
        """
        Get a full covariance matrix for this track, including covariances between
        different keypoints. This is used internally, but for visualization we
        use `get_covariances`.
        """
        tot = self.num_keypoint*self.num_dim
        return self.cov[:tot, :tot]

    def get_covariances(self) -> torch.Tensor:
        """
        Get a covariance matrix for each of the keypoints. The covariances are
        marginalized for each keypoint even if there are inter-keypoint variances.
        May return a small diagonal matrix if not implemented.
        """
        return per_point_cov(self.get_full_covariances())

    def __getitem__(self, key: str):
        if key == "id":
            return self.id
        if key == "kpts":
            return self.get_keypoints().tolist()
        raise KeyError


def per_point_cov(covar: torch.Tensor, num_dim: int = 3) -> torch.Tensor:
    """ Extract the block diagonal part of the given covariance matrix. """
    *Bs, N, N = covar.shape
    by_point = covar.view(-1, N // num_dim, num_dim, N // num_dim, num_dim)
    blocks = torch.diagonal(by_point, dim1=1, dim2=3)
    return blocks.permute(0, 3, 1, 2).view(*Bs, -1, num_dim, num_dim)


class Tracker:
    """
    This is the main class that performs multi-camera pose tracking. It uses
    a PoseDetector to get 2d detections from each camera view and then associates
    and tracks them in 3d using a Kalman filter. Tracks become active if they are
    observed a sufficient number of times, and removed once they are not observed
    for some time.
    """

    def __init__(
        self, detector: PoseDetector, physics: LinearPhysics, min_age: int = 3, max_inv: int = 10,
        mo_threshold: float = 0.0, mn_threshold: float = 0.0, num_keypoint: int = 17
    ):
        self.tracks: list[Track] = []
        self.last_id = 0
        self.detector = detector
        self.physics = physics
        self.min_age = min_age
        self.max_inv = max_inv
        self.mo_threshold = mo_threshold
        self.mn_threshold = mn_threshold
        self.num_keypoint = num_keypoint
        self.timing_stats = {}

    def add_time_stat(self, stage: str, start: float):
        """ Add a call to the timing stats that ended now and started at the given time. """
        if stage not in self.timing_stats:
            self.timing_stats[stage] = [0.0, 0]
        self.timing_stats[stage][0] += time.perf_counter() - start
        self.timing_stats[stage][1] += 1

    def get_prediction(self) -> list[Track]:
        """
        Get the internal state prediction for the current step. This will only
        return tracks that have a detection in the latest frame.
        """
        return [copy.copy(track) for track in self.tracks if track.last_detection == 0]

    def predict(self, dt: float):
        """
        Process a forward step in time to the given amount of seconds and update
        the internal state of all tracks accordingly, but without using any new
        external information.
        """
        t_start = time.perf_counter()
        if len(self.tracks) != 0:
            means = torch.stack([track.mean for track in self.tracks])
            covs = torch.stack([track.cov for track in self.tracks])
            means, covs = self.physics.predict(dt, means, covs)
            for track, mean, cov in zip(self.tracks, means, covs):
                track.moved(mean, cov)
        self.add_time_stat("prediction", t_start)

    def associate_pred_to_detection(self, cam: Camera, kpts: torch.Tensor, covs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Associate the given detections to one of the existing tracks. At most
        one detection in associated to each track. Returns two tensors of shape
        where the first has index of the tracks.
        """
        num_detect, num_dim = kpts.shape
        num_track = len(self.tracks)
        if num_detect == 0 or num_track == 0:
            return (
                torch.tensor([], dtype=torch.long, device=kpts.device),
                torch.tensor([], dtype=torch.long, device=kpts.device)
            )
        # Project track to 2d camera plane. Also project covariances.
        pred_means = torch.stack(
            [track.mean[:self.num_keypoint*3] for track in self.tracks])
        pred_jacs, pred_kpts = kalman.batched_jacobian(
            lambda x: cam.project_pinhole(x.view(-1, 3)).view(-1, self.num_keypoint*2), pred_means)
        pred_covar = torch.stack([track.cov[:self.num_keypoint*3, :self.num_keypoint*3]
                                 for track in self.tracks])
        pred_covar = pred_jacs @ pred_covar @ pred_jacs.mT
        # Build cost matrix.
        cost_matrix = torch.zeros(
            (num_track + num_detect, num_detect), device=kpts.device)
        for j in range(num_detect):
            dist = (kpts[j] - pred_kpts).unsqueeze(-1)
            total_cov = covs[j] + pred_covar
            dist = (dist.mT @ torch.linalg.solve(total_cov, dist)).flatten()
            _, logdet = torch.linalg.slogdet(total_cov)
            cost_matrix[:num_track, j] = dist + logdet \
                - num_dim*self.mo_threshold
        # Run Hungarian matching.
        cost_np = cost_matrix.cpu().numpy()
        row_idx, col_idx = scipy.optimize.linear_sum_assignment(cost_np)
        # Filter out matches that matched with dummies.
        tr_idx = torch.tensor(row_idx, dtype=torch.long, device=kpts.device)
        det_idx = torch.tensor(col_idx, dtype=torch.long, device=kpts.device)
        valid_mask = tr_idx < num_track
        return tr_idx[valid_mask], det_idx[valid_mask]

    def associate_detections(self, cams: list[Camera], detections: list[tuple[torch.Tensor, torch.Tensor]]) -> list[torch.Tensor]:
        """
        Associate the given detections between cameras. This will not compare
        against existing tracks, instead only matching between different camera
        views. Uses iterative greedy matching.
        """
        num_cams = len(cams)
        active_tracks = []
        for c_idx, (kpts, covs) in enumerate(detections):
            num_detect, num_dim = kpts.shape
            num_track = len(active_tracks)
            if num_track == 0 or num_detect == 0:
                # Initialize since we don't have any active tracks to match yet.
                for i in range(num_detect):
                    active_tracks.append({c_idx: i})
                continue
            # Build cost matrix.
            cost_matrix = torch.zeros(
                (num_track + num_detect, num_detect), device=kpts.device)
            cam_2 = cams[c_idx]
            for t_idx, track in enumerate(active_tracks):
                # Collect all camera views and points for this track.
                m_cams, m_kpts, m_covs = [cam_2], [kpts], [covs]
                for past_c, past_d in track.items():
                    kpt_1 = detections[past_c][0][past_d]
                    cov_1 = detections[past_c][1][past_d]
                    m_cams.append(cams[past_c])
                    m_kpts.append(kpt_1.expand(num_detect, num_dim))
                    m_covs.append(cov_1.expand(num_detect, num_dim, num_dim))
                # Triangulate using all camera views at once.
                mean3d = camera.triangulate_undistorted(
                    m_cams,
                    [m.view(num_detect, -1, 2) for m in m_kpts],
                    [per_point_cov(c, 2) for c in m_covs]
                )
                # Compute cost for reprojection into new camera view.
                pred_2 = cam_2.project_pinhole(mean3d) \
                    .view(num_detect, num_dim)
                diff2 = (kpts - pred_2).unsqueeze(-1)
                dist2 = (diff2.mT @ torch.linalg.solve(covs, diff2)).flatten()
                _, logdet_2 = torch.linalg.slogdet(covs)
                cost_sum = dist2 + logdet_2
                # Compute cost for reprojection into all prev camera view.
                for past_c, past_d in track.items():
                    kpt_1 = detections[past_c][0][past_d]
                    cov_1 = detections[past_c][1][past_d]
                    pred_1 = cams[past_c].project_pinhole(mean3d)\
                        .view(num_detect, num_dim)
                    diff1 = (kpt_1.expand(num_detect, num_dim) - pred_1) \
                        .unsqueeze(-1)
                    dist1 = (diff1.mT @ torch.linalg.solve(cov_1, diff1)) \
                        .flatten()
                    _, logdet_1 = torch.linalg.slogdet(cov_1)
                    cost_sum += dist1 + logdet_1
                # Average cost over all reprojections.
                avg_cost = (cost_sum / (len(track) + 1)) \
                    - (num_dim * self.mn_threshold)
                cost_matrix[t_idx, :] = avg_cost
            # Run Hungarian matching.
            cost_np = cost_matrix.cpu().numpy()
            tr_idx, det_idx = scipy.optimize.linear_sum_assignment(cost_np)
            # Assign detections to active tracks and create new tracks.
            for t, d in zip(tr_idx, det_idx):
                if t < num_track:
                    active_tracks[t][c_idx] = d
                else:
                    active_tracks.append({c_idx: d})
        # Format results as expected.
        matched_results = []
        for c_idx in range(num_cams):
            matched_results.append(torch.tensor([
                track.get(c_idx, -1) for track in active_tracks
            ], dtype=torch.long, device=detections[0][0].device))
        return matched_results

    def new_track(self, mean: torch.Tensor) -> Track:
        """ Create a new track with the given mean. """
        self.last_id += 1
        full_mean = self.physics.init_mean.clone()
        full_mean[:self.num_keypoint*3] = mean
        if isinstance(self.physics, ConstrainedPhysics):
            full_mean[self.physics.constraints[2]] \
                = self.physics.compute_distances(mean)
        full_cov = self.physics.init_cov.clone()
        return Track(self.last_id, full_mean, full_cov, self.num_keypoint)

    def update_track(self, track: Track, cams: list[Camera], kpts: list[torch.Tensor], covs: list[torch.Tensor]):
        """ Update the given track with detections in multiple camera views. """
        ob_fs = [lambda x, cam=cam: cam.project_pinhole(x[:self.num_keypoint*3].view(-1, 3)).flatten()
                 for cam in cams]
        ob_ms = [mean for mean in kpts]
        ob_vs = [cov for cov in covs]
        # Add pseudo-observation for limb length constraints.
        if isinstance(self.physics, ConstrainedPhysics):
            ob_fs.append(lambda x: self.physics.pseudo_obs(x))  # type: ignore
            # Constraints are supposed to have zero difference.
            ob_ms.append(torch.zeros(
                self.physics.num_constr, device=track.mean.device))
            ob_vs.append(self.physics.constr_cov)
        ob_f, ob_m, ob_v = kalman.emerge_obs(ob_fs, ob_ms, ob_vs)
        mean, cov = kalman.eupdate(track.mean, track.cov, ob_m, ob_v, ob_f)
        track.update(mean, cov)

    def update(self, cams: list[Camera], imgs: list[torch.Tensor]):
        """
        Update the internal track states based on new incoming images, but don't
        perform any internal time step updates.
        """
        # Perform 2d detection on each image.
        t_start = time.perf_counter()
        detections = self.detector.detect(cams, imgs)
        self.add_time_stat("detection", t_start)
        # Match each images detections to tracks.
        t_start = time.perf_counter()
        obs: list[tuple[list[Camera], list[torch.Tensor], list[torch.Tensor]]] \
            = [([], [], []) for _ in self.tracks]
        nomatch = []
        for cam, (kpts, covs) in zip(cams, detections):
            tr_idx, det_idx = self.associate_pred_to_detection(cam, kpts, covs)
            for ti, di in zip(tr_idx, det_idx):
                obs[ti][0].append(cam)
                obs[ti][1].append(kpts[di])
                obs[ti][2].append(covs[di])
            nomatch.append((
                util.remove_idx(kpts, det_idx), util.remove_idx(covs, det_idx)
            ))
        self.add_time_stat("matching_old", t_start)
        # Update matched tracks using new detections.
        t_start = time.perf_counter()
        new_tracks = []
        for track, (ob_cams, ob_kpts, ob_covs) in zip(self.tracks, obs):
            if len(ob_cams) != 0:
                self.update_track(track, ob_cams, ob_kpts, ob_covs)
                new_tracks.append(track)
            else:
                # Check if we want to delete the track.
                track.no_update()
                if track.num_detection >= self.min_age and track.last_detection <= self.max_inv:
                    new_tracks.append(track)
        self.add_time_stat("update", t_start)
        # Match unassigned detections between camera views.
        t_start = time.perf_counter()
        matched = self.associate_detections(cams, nomatch)
        self.add_time_stat("matching_new", t_start)
        t_start = time.perf_counter()
        # Create new tracks.
        for match in zip(*matched):
            m_cams, m_kpts, m_covs = [], [], []
            for cam, m, det in zip(cams, match, nomatch):
                if m.item() != -1:
                    m_cams.append(cam)
                    kpts, covs = det[0][m], det[1][m]
                    m_kpts.append(kpts)
                    m_covs.append(covs)
            if len(m_cams) >= 2:
                # Create new track if we have more than two views.
                mean = camera.triangulate_undistorted(
                    m_cams,
                    [m.view(-1, 2) for m in m_kpts],
                    [per_point_cov(c, 2) for c in m_covs]
                ).flatten()
                track = self.new_track(mean)
                self.update_track(track, m_cams, m_kpts, m_covs)
                new_tracks.append(track)
        self.tracks = new_tracks
        self.add_time_stat("create_new", t_start)

    def evaluate(self, source: source.VideoSource, progress=100, limit=1000000) -> tuple[list[Camera], list[list[Track]], float]:
        """
        Run the complete evaluation loop using the given video source and return
        as a result the camera positions, tracks in each frame, and fps. This will
        run until the video source is exhausted.
        """
        last_cams, frames = [], []
        last_ts = 0
        try:
            with torch.inference_mode():
                source.start()
                for i in range(1, limit + 1):
                    ts, imgs, cams = source.next_frames()
                    if ts is None or cams is None or imgs is None:
                        break
                    dt = ts - last_ts
                    self.predict(dt)
                    self.update(cams, imgs)
                    frames.append(self.get_prediction())
                    last_ts, last_cams = ts, cams
                    if progress != 0 and i % progress == 0:
                        print(f"finished frame {i}")
        except KeyboardInterrupt:
            # We want to stop, but let's still return the results so the time
            # was not wasted. (Allows for early stop.)
            pass
        finally:
            source.release()
        return last_cams, frames, 1/dt


class CrossViewFirstTracker(Tracker):
    """
    Tracker that performs cross-view association first.
    """

    def update(self, cams: list[Camera], imgs: list[torch.Tensor]):
        # Perform 2d detection on each image.
        t_start = time.perf_counter()
        detections = self.detector.detect(cams, imgs)
        self.add_time_stat("detection", t_start)
        # Match detections between camera views.
        t_start = time.perf_counter()
        matched = self.associate_detections(cams, detections)
        self.add_time_stat("matching_new", t_start)
        # Match matched detections against tracks.
        t_start = time.perf_counter()
        # Build cost matrix.
        num_detect, num_dim = matched[0].shape[0], detections[0][0].shape[1]
        num_track = len(self.tracks)
        cost_matrix = torch.zeros(
            (num_track + num_detect, num_detect), device=matched[0].device)
        if num_detect > 0 and num_track > 0:
            a_pred_kpts, a_pred_covar = [], []
            for cam in cams:
                # Project track to 2d camera plane. Also project covariances.
                pred_means = torch.stack(
                    [track.mean[:self.num_keypoint*3] for track in self.tracks])
                pred_jacs, pred_kpts = kalman.batched_jacobian(
                    lambda x: cam.project_pinhole(x.view(-1, 3)).view(-1, self.num_keypoint*2), pred_means)
                pred_covar = torch.stack([track.cov[:self.num_keypoint*3, :self.num_keypoint*3]
                                         for track in self.tracks])
                pred_covar = pred_jacs @ pred_covar @ pred_jacs.mT
                a_pred_kpts.append(pred_kpts)
                a_pred_covar.append(pred_covar)
            for j, match in enumerate(zip(*matched)):
                count = 0
                for i, (cam, m, det) in enumerate(zip(cams, match, detections)):
                    if m.item() != -1:
                        kpts, covs = det[0][m], det[1][m]
                        dist = (kpts - a_pred_kpts[i]).unsqueeze(-1)
                        total_cov = covs + a_pred_covar[i]
                        dist = (
                            dist.mT @ torch.linalg.solve(total_cov, dist)).flatten()
                        _, logdet = torch.linalg.slogdet(total_cov)
                        cost_matrix[:num_track, j] += dist + logdet \
                            - num_dim*self.mo_threshold
                        count += 1
                cost_matrix[:num_track, j] /= count
        # Run Hungarian matching.
        cost_np = cost_matrix.cpu().numpy()
        row_idx, col_idx = scipy.optimize.linear_sum_assignment(cost_np)
        self.add_time_stat("matching_old", t_start)
        # Update matched tracks or create a new one.
        t_start = time.perf_counter()
        new_tracks = []
        for i, j in zip(row_idx, col_idx):
            match = [m[j] for m in matched]
            m_cams, m_kpts, m_covs = [], [], []
            for cam, m, det in zip(cams, match, detections):
                if m.item() != -1:
                    m_cams.append(cam)
                    kpts, covs = det[0][m], det[1][m]
                    m_kpts.append(kpts)
                    m_covs.append(covs)
            if i < num_track:
                # Update the existing matched track.
                track = self.tracks[i]
            else:
                # Create a new track if we have sufficient detections.
                if len(m_cams) >= 2:
                    # Create new track if we have more than two views.
                    mean = camera.triangulate_undistorted(
                        m_cams,
                        [m.view(-1, 2) for m in m_kpts],
                        [per_point_cov(c, 2) for c in m_covs]
                    ).flatten()
                    track = self.new_track(mean)
                else:
                    continue
            self.update_track(track, m_cams, m_kpts, m_covs)
            new_tracks.append(track)
        for i in set(range(num_track)) - set(row_idx):
            track = self.tracks[i]
            # Check if we want to delete the track.
            track.no_update()
            if track.num_detection >= self.min_age and track.last_detection <= self.max_inv:
                new_tracks.append(track)
        self.add_time_stat("update", t_start)
        self.tracks = new_tracks


def build_physics(scale=100.0) -> LinearPhysics:
    """
    Build some standard physics based on the given scale. The scale must be
    relative to meters, i.e., `scale=100` means centimeter units.
    """
    nk = 17*3
    dyn_mat = torch.concat([
        torch.concat([torch.zeros(nk, nk), torch.eye(nk)], dim=1),
        torch.concat([torch.zeros(nk, nk), torch.eye(nk)*-0.2], dim=1)
    ], dim=0)
    dyn_cov = torch.diag(torch.concat([
        torch.full((nk,), (0.05 * scale)**2),
        torch.full((nk,), (3.0 * scale)**2),
    ]))
    init_mean = torch.zeros(nk + nk)
    init_cov = torch.diag(torch.concat([
        torch.full((nk,), (1.0 * scale)**2),
        torch.full((nk,), (5.0 * scale)**2),
    ]))
    return LinearPhysics(dyn_mat, dyn_cov, init_mean, init_cov)


def build_constrained_physics(scale=100.0, sym=True) -> kalman.ConstrainedPhysics:
    """ Build physics that includes limb length estimation and rigid body constraints. """
    links = util.RIGID_SKELETON
    nk, num_links = 17*3, len(links)
    num_len = max(util.RIGID_SKELETON_SYM) + 1 if sym else num_links
    dyn_mat = torch.concat([
        torch.concat([
            torch.zeros(nk, nk), torch.eye(nk), torch.zeros(nk, num_len)], dim=1),
        torch.concat([
            torch.zeros(nk, nk), torch.eye(nk)*-0.2, torch.zeros(nk, num_len)], dim=1),
        torch.zeros(num_len, 2*nk + num_len)
    ], dim=0)
    dyn_cov = torch.diag(torch.concat([
        torch.full((nk,), (0.01 * scale)**2),
        torch.full((nk,), (3.0 * scale)**2),
        torch.full((num_len,), (0.01 * scale)**2),
    ]))
    init_mean = torch.zeros(nk + nk + num_len)
    init_cov = torch.diag(torch.concat([
        torch.full((nk,), (1.0 * scale)**2),
        torch.full((nk,), (5.0 * scale)**2),
        torch.full((num_len,), (1.0 * scale)**2),
    ]))
    constraints = torch.tensor([
        [i, j, 2*nk + (util.RIGID_SKELETON_SYM[k] if sym else k)]
        for k, (i, j) in enumerate(links)
    ], dtype=torch.long).T
    point_mix = torch.tensor([[5, 6], [11, 12]], dtype=torch.long).T
    constr_cov = torch.eye(num_links) * (0.1 * scale)**2
    return kalman.ConstrainedPhysics(
        dyn_mat, dyn_cov, init_mean, init_cov, constraints, point_mix, constr_cov)


def build_walled_physics(scale=100.0, sym=True, center=(0, 0, 0), up=(0, -1, 0)) -> kalman.ConstrainedPhysics:
    """ Build physics that includes limb length and wall/floor constraints. """
    links = util.RIGID_SKELETON
    nkp, nk, num_links = 17, 17*3, len(links)
    num_len = max(util.RIGID_SKELETON_SYM) + 1 if sym else num_links
    dyn_mat = torch.concat([
        torch.concat([
            torch.zeros(nk, nk), torch.eye(nk), torch.zeros(nk, num_len)], dim=1),
        torch.concat([
            torch.zeros(nk, nk), torch.eye(nk)*-0.2, torch.zeros(nk, num_len)], dim=1),
        torch.zeros(num_len, 2*nk + num_len)
    ], dim=0)
    dyn_cov = torch.diag(torch.concat([
        torch.full((nk,), (0.01 * scale)**2),
        torch.full((nk,), (3.0 * scale)**2),
        torch.full((num_len,), (0.01 * scale)**2),
    ]))
    init_mean = torch.zeros(nk + nk + num_len)
    init_cov = torch.diag(torch.concat([
        torch.full((nk,), (1.0 * scale)**2),
        torch.full((nk,), (5.0 * scale)**2),
        torch.full((num_len,), (1.0 * scale)**2),
    ]))
    constraints = torch.tensor([
        [i, j, 2*nk + (util.RIGID_SKELETON_SYM[k] if sym else k)]
        for k, (i, j) in enumerate(links)
    ], dtype=torch.long).T
    point_mix = torch.tensor([[5, 6], [11, 12]], dtype=torch.long).T
    constr_cov = torch.diag(torch.concat([
        torch.full((num_links,), (0.1 * scale)**2),
        torch.full((nkp,), (1.0 * scale)**2),
        torch.full((2,), (2.0 * scale)**2),
    ]))
    wall_centers = torch.tensor([center], dtype=torch.float32)
    wall_norm = torch.tensor([up], dtype=torch.float32)
    feet_kpts = torch.tensor([[15, 0], [16, 0]], dtype=torch.long).T
    return kalman.WalledPhysics(
        dyn_mat, dyn_cov, init_mean, init_cov, constraints, point_mix,
        constr_cov, wall_centers, wall_norm, feet_kpts, 0.05 * scale)
