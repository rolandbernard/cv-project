
import os
import sys

from typing import Any

import pyvista as pv
import numpy as np

import util


class BaseSkeletonPlayer:
    """
    Base class for 3D skeleton visualization using PyVista.
    """

    def __init__(self, cameras: list, center=(0, 0, 0), up=(0, -1, 0)):
        self.scale = self.approx_scale(cameras)
        self.track_meshes = {}
        self.track_actors = {}

        # Setup the plotter. (Launder the type though Any to avoid wrong errors.)
        self.pl: Any = pv.Plotter()
        self.pl.add_axes()
        self.pl.set_background("white")
        # Precompute skeleton topology.
        lines = []
        for p1, p2 in util.RIGID_SKELETON:
            lines.extend([2, p1, p2])
        self.skeleton = np.array(lines)
        # Setup the scene.
        self.setup_cameras(cameras)
        # Set initial camera position.
        dist = 2 * self.scale
        cam_pos = (
            center[0] + (dist if up[0] >= 0 else -dist),
            center[1] + (dist if up[1] >= 0 else -dist),
            center[2] + (dist if up[2] >= 0 else -dist)
        )
        self.pl.camera_position = [cam_pos, center, up]

    def approx_scale(self, cameras: list):
        """ Estimate the scale of the scene based on camera positions. """
        max_dist = 1e-5
        centers = []
        for cam in cameras:
            center = -np.array(cam["R"]).T @ np.array(cam["t"]).flatten()
            centers.append(center)

        for i, c0 in enumerate(centers):
            for j in range(i + 1, len(centers)):
                dist = np.linalg.norm(c0 - centers[j]).item()
                if dist > max_dist:
                    max_dist = dist
        return max_dist

    def setup_ground(self, center: tuple[float, float, float], up: tuple[float, float, float]):
        """ Add a ground plane to the scene. """
        ground = pv.Plane(
            center=center, direction=up, i_size=self.scale,
            j_size=self.scale, i_resolution=20, j_resolution=20
        )
        self.pl.add_mesh(ground, style="wireframe", color="lightgray")

    def setup_cameras(self, cameras: list, scale: float = 0.05):
        """ Add camera frustums to the scene. """
        for cam in cameras:
            rotation = np.array(cam["R"])
            translate = np.array(cam["t"]).flatten()
            camera_center = -rotation.T @ translate
            intrinsics = np.array(cam["K"])
            fx, fy = intrinsics[0, 0], intrinsics[1, 1]
            cx, cy = intrinsics[0, 2], intrinsics[1, 2]
            width, height = cx * 2, cy * 2
            z_cam = self.scale * scale
            x0 = (0 - cx) * z_cam / fx
            x1 = (width - cx) * z_cam / fx
            y0 = (0 - cy) * z_cam / fy
            y1 = (height - cy) * z_cam / fy
            corners_cam = np.array([
                [x0, y0, z_cam], [x1, y0, z_cam],
                [x1, y1, z_cam], [x0, y1, z_cam]
            ])
            corners_world = (corners_cam @ rotation) + camera_center
            vertices = np.vstack([camera_center, corners_world])
            lines = np.array([
                [2, 0, 1], [2, 0, 2], [2, 0, 3], [2, 0, 4],
                [2, 1, 2], [2, 2, 3], [2, 3, 4], [2, 4, 1]
            ]).flatten()
            camera_wireframe = pv.PolyData(vertices, lines=lines)
            self.pl.add_mesh(camera_wireframe, color="black", line_width=2)
            self.pl.add_mesh(
                pv.Sphere(radius=z_cam * 0.1, center=camera_center), color="red")

    def get_or_create_track(self, track_id, gt: bool = False):
        """ Retrieve an existing track mesh/actor or create a new one. """
        track_key = (gt, track_id)
        if track_key not in self.track_meshes:
            mesh = pv.PolyData()
            if gt:
                actor = self.pl.add_mesh(
                    mesh, color="red",
                    render_lines_as_tubes=True, line_width=2,
                    render_points_as_spheres=True, point_size=2,
                    smooth_shading=True
                )
            else:
                color = util.COLORS[track_id % len(util.COLORS)]
                actor = self.pl.add_mesh(
                    mesh, color=color,
                    render_lines_as_tubes=True, line_width=8,
                    render_points_as_spheres=True, point_size=15,
                    smooth_shading=True
                )
            self.track_meshes[track_key] = mesh
            self.track_actors[track_key] = actor
        return self.track_meshes[track_key], self.track_actors[track_key]

    def augmented_kpts(self, track):
        """ Augment keypoints if necessary. """
        kpts = np.array(track["kpts"])
        if len(kpts) == 17:
            return np.concatenate([
                kpts, kpts[5:7].mean(axis=0, keepdims=True)
            ])
        return kpts

    def add_point_cloud(self, points, colors, point_size=0.001):
        """ Add a point cloud to the 3D visualization. """
        poly = pv.PolyData(points)
        poly["colors"] = colors
        self.pl.add_mesh(
            poly, scalars="colors", rgb=True,
            point_size=point_size * self.scale,
            render_points_as_spheres=True,
            opacity=0.6
        )

    def set_frame(self, tracks, gt_tracks=None):
        """ Update the scene with a new set of tracks. """
        # Hide all tracks.
        for actor in self.track_actors.values():
            actor.SetVisibility(False)
        # Update and show all tracks visible at this time step.
        for track in tracks:
            new_mesh = pv.PolyData(self.augmented_kpts(track))
            new_mesh.lines = self.skeleton
            mesh, actor = self.get_or_create_track(track["id"])
            mesh.copy_from(new_mesh)
            actor.SetVisibility(True)
        # Update ground truth tracks if available.
        if gt_tracks:
            for track in gt_tracks:
                new_mesh = pv.PolyData(self.augmented_kpts(track))
                new_mesh.lines = self.skeleton
                mesh, actor = self.get_or_create_track(track["id"], True)
                mesh.copy_from(new_mesh)
                actor.SetVisibility(True)


class SkeletonPlayer(BaseSkeletonPlayer):
    """
    Offline skeleton player for pre-recorded tracks.
    """

    def __init__(
        self, cameras: list, frames: list, fps: float, center=(0, 0, 0),
        up=(0, -1, 0), gt_frames: None | list = None
    ):
        super().__init__(cameras, center, up)
        self.fps = fps
        self.frames = frames
        self.gt_frames = gt_frames
        self.current_frame = 0
        self.is_playing = False
        self.setup_ground(center, up)
        self.setup_widgets()
        self.update_scene(0)

    def update_scene(self, value):
        """ Callback for slider widget. """
        frame_idx = int(np.round(value))
        if frame_idx >= len(self.frames):
            frame_idx = len(self.frames) - 1
        self.current_frame = frame_idx
        gt_tracks = None
        if self.gt_frames and self.current_frame < len(self.gt_frames):
            gt_tracks = self.gt_frames[self.current_frame]
        self.set_frame(self.frames[self.current_frame], gt_tracks)

    def toggle_play(self):
        """ Toggle playback on/off. """
        self.is_playing = not self.is_playing

    def timer_callback(self, step):
        """ Timer callback for automatic playback. """
        if self.is_playing:
            self.next_frame()

    def next_frame(self):
        """ Advance to the next frame. """
        next_frame = (self.current_frame + 1) % len(self.frames)
        self.slider.GetRepresentation().SetValue(next_frame)
        self.update_scene(next_frame)
        self.pl.render()

    def prev_frame(self):
        """ Go back to the previous frame. """
        prev_frame = (self.current_frame - 1) % len(self.frames)
        self.slider.GetRepresentation().SetValue(prev_frame)
        self.update_scene(prev_frame)
        self.pl.render()

    def setup_widgets(self):
        """ Setup UI widgets for offline playback. """
        self.slider = self.pl.add_slider_widget(
            callback=self.update_scene, rng=[0, len(self.frames) - 1], value=0,
            pointa=(0.1, 0.03), pointb=(0.9, 0.03), style="modern",
            interaction_event="always"
        )
        self.pl.add_timer_event(
            max_steps=1000000, duration=round(1000/self.fps), callback=self.timer_callback)
        self.pl.add_key_event("space", self.toggle_play)
        self.pl.add_key_event("Right", self.next_frame)
        self.pl.add_key_event("Left", self.prev_frame)

    def show(self):
        """ Show the plotter by opening the window. """
        pv.set_jupyter_backend('client')
        self.pl.show()


class LiveSkeletonPlayer(BaseSkeletonPlayer):
    """
    Real-time skeleton player for live tracking.
    """

    def __init__(self, cameras: list, center=(0, 0, 0), up=(0, -1, 0)):
        super().__init__(cameras, center, up)
        self.pl.show(interactive_update=True)

    def update(self, tracks):
        """ Update the visualization with new tracks. """
        self.set_frame(tracks)
        self.pl.render()


def load_from_files(main_file: str, gt_file: None | str = None) -> SkeletonPlayer:
    """ Load results and optional ground truth from the given files. """
    cams, frames, fps, center, up = util.load_tracks(main_file)
    gt_frames = None
    if gt_file is not None:
        _, gt_frames, _, _, _ = util.load_tracks(gt_file)
    return SkeletonPlayer(cams, frames, fps, center, up, gt_frames)


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        print(f"usage: python {sys.argv[0]} FILENAME [GT_FILENAME]")
        exit(1)
    if not os.path.isfile(sys.argv[1]):
        print(f"unable to open tracking file '{sys.argv[1]}'")
        exit(1)
    cams, frames, fps, center, up = util.load_tracks(sys.argv[1])
    gt_frames = None
    if len(sys.argv) > 2:
        if not os.path.isfile(sys.argv[2]):
            print(f"unable to open ground truth file '{sys.argv[2]}'")
            exit(1)
        _, gt_frames, _, _, _ = util.load_tracks(sys.argv[2])
    player = SkeletonPlayer(cams, frames, fps, center, up, gt_frames)
    player.show()
