
import copy
from itertools import count

import torch
import scipy.optimize

import util
import camera
import kalman
import source
from camera import Camera
from detect import PoseDetector
from kalman import LinearPhysics


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
    *Bs, N, N = covar.shape
    by_point = covar.view(-1, N // num_dim, num_dim, N // num_dim, num_dim)
    blocks = torch.diagonal(by_point, dim1=1, dim2=3)
    return blocks.permute(0, 3, 1, 2).view(*Bs, -1, num_dim, num_dim)


class Tracker:
    """
    A tracker that only tracks objects in individual 2d images. Detections are
    matched to tracks using the hungarian algorithm. Tracks become active if they
    are observed a sufficient number of times, and removed once they are not
    observed for some time.
    """

    def __init__(
        self, detector: PoseDetector, physics: LinearPhysics, min_age: int = 3, max_inv: int = 10,
        mo_threshold: float = 0.0, mn_threshold: float = 0.0, min_var=1e-5, num_keypoint: int = 17
    ):
        self.tracks: list[Track] = []
        self.last_id = 0
        self.detector = detector
        self.physics = physics
        self.min_age = min_age
        self.max_inv = max_inv
        self.mo_threshold = mo_threshold
        self.mn_threshold = mn_threshold
        self.min_var = min_var
        self.num_keypoint = num_keypoint

    def get_prediction(self) -> list[Track]:
        """ Get the internal state prediction for the current step. """
        return [copy.copy(track) for track in self.tracks if track.num_detection >= self.min_age]

    def predict(self, dt: float):
        """
        Process a forward step in time to the given amount of seconds and update
        the internal state of all tracks accordingly, but without using any new
        external information.
        """
        if len(self.tracks) != 0:
            means = torch.stack([track.mean for track in self.tracks])
            covs = torch.stack([track.cov for track in self.tracks])
            means, covs = self.physics.predict(dt, means, covs)
            for track, mean, cov in zip(self.tracks, means, covs):
                track.moved(mean, cov)

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
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_np)
        # Filter out matches that matched with dummies.
        tr_idx = torch.tensor(row_ind, dtype=torch.long, device=kpts.device)
        det_idx = torch.tensor(col_ind, dtype=torch.long, device=kpts.device)
        valid_mask = tr_idx < num_track
        return tr_idx[valid_mask], det_idx[valid_mask]

    def associate_detections(self, cams: list[Camera], detections: list[tuple[torch.Tensor, torch.Tensor]]) -> list[torch.Tensor]:
        """
        Associate the given detections to one of the existing tracks. At most
        one detection in associated to each track. Returns tensors with index
        into the given detections array.
        """
        num_cams = len(cams)
        active_tracks = []
        for c_idx, (kpts, covs) in enumerate(detections):
            num_detect = kpts.shape[0]
            num_track = len(active_tracks)
            if num_track == 0 or num_detect == 0:
                # If no tracks exists yet, populate with first cameras detection.
                for i in range(num_detect):
                    active_tracks.append({c_idx: i})
                continue
            # Build cost matrix.
            cost_matrix = torch.zeros((num_track + num_detect, num_detect))
            for t_idx, track in enumerate(active_tracks):
                for d_idx in range(num_detect):
                    # Distance between a track and a detection is the mean
                    # reprojection error to the track's detections.
                    total_dist = 0.0
                    for past_c, past_d in track.items():
                        m_cams = [cams[past_c], cams[c_idx]]
                        m_kpts = [detections[past_c][0][past_d], kpts[d_idx]]
                        m_covs = [detections[past_c][1][past_d], covs[d_idx]]
                        mean3d = camera.triangulate_undistorted(
                            m_cams,
                            [m.view(-1, 2) for m in m_kpts],
                            [per_point_cov(c, 2) for c in m_covs]
                        )
                        diff1 = m_kpts[0] \
                            - m_cams[0].project_pinhole(mean3d).flatten()
                        diff2 = m_kpts[1] \
                            - m_cams[1].project_pinhole(mean3d).flatten()
                        dist1 = torch.dot(
                            diff1, torch.linalg.solve(m_covs[0], diff1))
                        dist2 = torch.dot(
                            diff2, torch.linalg.solve(m_covs[1], diff2))
                        total_dist += (dist1 + dist2).item()
                    cost_matrix[t_idx, d_idx] = total_dist / len(track) - 24
            # Run Hungarian matching.
            cost_np = cost_matrix.cpu().numpy()
            row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_np)
            # Filter out matches that matched with dummies.
            valid_mask = row_ind < num_track
            tr_idx = row_ind[valid_mask]
            det_idx = col_ind[valid_mask]
            # Assign each matched detection to a track.
            for t_idx, d_det in zip(tr_idx, det_idx):
                active_tracks[t_idx][c_idx] = d_det
            # Assign unmatched detections to new tracks.
            matched_d = set(det_idx)
            for d_idx in range(num_detect):
                if d_idx not in matched_d:
                    active_tracks.append({c_idx: d_idx})
        # Format the dictionaries into the tuple structure expected by your update() loop
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
        full_cov = self.physics.init_cov.clone()
        return Track(self.last_id, full_mean, full_cov, self.num_keypoint)

    def constrain_covars(self):
        """ Constrain the covariances to avoid them becoming too small. """
        for track in self.tracks:
            track.cov.diagonal().clamp(min=self.min_var)

    def update_track(self, track: Track, cams: list[Camera], kpts: list[torch.Tensor], covs: list[torch.Tensor]):
        obf, ob_m, ob_v = kalman.emerge_obs(
            [lambda x, c=cam: c.project_pinhole(x[:self.num_keypoint*3].view(-1, 3)).flatten()
             for cam in cams],
            [mean for mean in kpts],
            [cov for cov in covs]
        )
        mean, cov = kalman.eupdate(
            track.mean, track.cov, ob_m, ob_v, obf)
        track.update(mean, cov)

    def update(self, cams: list[Camera], imgs: list[torch.Tensor]):
        """
        Update the internal track states based on new incoming images, but don't
        perform any internal time step updates.
        """
        # Perform 2d detection on each image.
        detections = self.detector.detect(cams, imgs)
        # Match each images detections to tracks.
        obs: list[tuple[list[Camera], list[torch.Tensor], list[torch.Tensor]]] \
            = [([], [], []) for _ in range(len(self.tracks))]
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
        new_tracks = []
        # Update matched tracks using new detections.
        for track, (ob_cams, ob_kpts, ob_covs) in zip(self.tracks, obs):
            if len(ob_cams) != 0:
                self.update_track(track, ob_cams, ob_kpts, ob_covs)
                new_tracks.append(track)
            else:
                # Check if we want to delete the track.
                track.no_update()
                if track.num_detection >= self.min_age and track.last_detection <= self.max_inv:
                    new_tracks.append(track)
        # Match unassigned detections to create new tracks.
        matched = self.associate_detections(cams, nomatch)
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

    def evaluate(self, source: source.VideoSource, progress=100) -> tuple[list[Camera], list[list[Track]], float]:
        """
        Run the complete evaluation loop using the given video source and return
        as a result the camera positions, tracks in each frame, and fps. This will
        run until the video source is exhausted.
        """
        last_cams, frames = [], []
        last_ts = 0
        try:
            source.start()
            for i in count(1):
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
