
import json

import pyvista as pv
import numpy as np

from camera import Camera

SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (11, 12), (5, 11), (6, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]

COLORS = ["#FF5733", "#33FF57", "#3357FF",
          "#FF33A8", "#33FFF5", "#F5FF33", "#A833FF"]


class Track:
    def __init__(self, track_id, kpts, covs):
        self.id = track_id
        self.kpts = np.array(kpts)
        self.covs = np.array(covs)

    def get_keypoints(self):
        return self.kpts

    def get_covariances(self):
        return self.covs


class SkeletonPlayer:
    def __init__(self, cameras, frames, cov_threshold=0.0):
        self.cameras = cameras
        self.frames = frames
        self.num_frames = len(frames)
        self.current_frame = 0
        self.is_playing = False

        self.cov_threshold = cov_threshold

        self.track_meshes = {}
        self.track_actors = {}

        self.pl = pv.Plotter()
        self.pl.add_axes()
        self.pl.set_background("white")

        self.setup_ground()
        self.setup_widgets()
        self.update_scene(0)

    def setup_ground(self):
        ground = pv.Plane(
            center=(0, 0, 0), direction=(0, 1, 0), i_size=10,
            j_size=10, i_resolution=20, j_resolution=20
        )
        self.pl.add_mesh(ground, style="wireframe", color="lightgray")

    def get_or_create_track(self, track_id):
        if track_id not in self.track_meshes:
            mesh = pv.PolyData()
            color = COLORS[track_id % len(COLORS)]
            actor = self.pl.add_mesh(
                mesh, color=color,
                render_lines_as_tubes=True, line_width=8,
                render_points_as_spheres=True, point_size=15,
                smooth_shading=True
            )
            self.track_meshes[track_id] = mesh
            self.track_actors[track_id] = actor
        return self.track_meshes[track_id], self.track_actors[track_id]

    def update_scene(self, value):
        frame_idx = int(np.round(value))
        if frame_idx >= self.num_frames:
            frame_idx = self.num_frames - 1
        self.current_frame = frame_idx
        for actor in self.track_actors.values():
            actor.SetVisibility(False)
        for track in self.frames[self.current_frame]:
            kpts = track.get_keypoints()
            covs = track.get_covariances()
            uncertainties = np.trace(covs, axis1=1, axis2=2)
            valid_mask = uncertainties < self.cov_threshold
            valid_indices = np.where(valid_mask)[0]
            if len(valid_indices) == 0:
                continue
            mapping = {old_idx: new_idx for new_idx,
                       old_idx in enumerate(valid_indices)}
            valid_kpts = kpts[valid_indices]
            lines = []
            for p1, p2 in SKELETON:
                if p1 in mapping and p2 in mapping:
                    lines.extend([2, mapping[p1], mapping[p2]])
            new_mesh = pv.PolyData(valid_kpts)
            if lines:
                new_mesh.lines = np.array(lines)
            mesh, actor = self.get_or_create_track(track.id)
            mesh.copy_from(new_mesh)
            actor.SetVisibility(True)

    def toggle_play(self):
        self.is_playing = not self.is_playing

    def timer_callback(self, step):
        if self.is_playing:
            next_frame = (self.current_frame + 1) % self.num_frames
            self.slider.GetRepresentation().SetValue(next_frame)
            self.update_scene(next_frame)
            self.pl.render()

    def next_frame(self):
        if not self.is_playing:
            next_frame = (self.current_frame + 1) % self.num_frames
            self.slider.GetRepresentation().SetValue(next_frame)
            self.update_scene(next_frame)
            self.pl.render()

    def prev_frame(self):
        if not self.is_playing:
            prev_frame = (self.current_frame - 1) % self.num_frames
            self.slider.GetRepresentation().SetValue(prev_frame)
            self.update_scene(prev_frame)
            self.pl.render()

    def setup_widgets(self):
        self.slider = self.pl.add_slider_widget(
            callback=self.update_scene, rng=[0, self.num_frames - 1], value=0,
            pointa=(0.1, 0.03), pointb=(0.9, 0.03), style="modern",
            interaction_event="always"
        )
        self.pl.add_timer_event(
            max_steps=1000000, duration=1000//25, callback=self.timer_callback)
        self.pl.add_key_event("space", self.toggle_play)
        self.pl.add_key_event("Right", self.next_frame)
        self.pl.add_key_event("Left", self.prev_frame)

    def show(self):
        self.pl.show()


if __name__ == "__main__":
    frames = []
    for i in range(1, 1500):
        with open(f"code/data/demo2/{i}.json", "r") as f:
            tracks = json.load(f)
        frames.append([
            Track(track["id"], track["kpts"], track["covs"])
            for track in tracks
        ])
    cameras = []
    for cid in range(4):
        cam = Camera()
        cam.load_ini(f"code/data/salsa/cam{cid}.ini")
        cameras.append(cam)
    player = SkeletonPlayer(cameras, frames, cov_threshold=1.0)
    player.show()
