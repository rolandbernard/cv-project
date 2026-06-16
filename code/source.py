
import time
import copy
import threading

import cv2
import torch
import numpy as np

from camera import Camera


class VideoSource:
    """
    This is an abstract class to implement for all video input sources used in
    this project. It must take care not only of the individual frames, but also
    synchronizing them and providing camera intrinsics and extrinsics.
    """

    def start(self):
        """
        Some input methods may need to explicitly start the capturing of new
        frames to open external connections or start background processing.
        """
        pass

    def release(self):
        """
        Some methods may need to dispose of some external resources like active
        connections when the user is finished with them.
        """
        pass

    def next_frames(self) -> tuple[None, None, None] | tuple[float, list[torch.Tensor], list[Camera]]:
        """
        This function must be implemented by all video input methods. It should
        return `None` if there are no new further frames available, and there
        will not be any in the future. Otherwise, it should return a tuple
        containing respectively the real time of the observations in seconds
        from some arbitrary point in the past, the list of camera parameters, and
        the list of images from each camera.
        """
        raise NotImplementedError


class OfflineVideoSource(VideoSource):
    """
    This class implements input from video files using OpenCV. For timing, it
    uses the framerate of the streams. Note that the frame rate of all videos
    must be the same or this source will fail to start.
    """

    def __init__(self, streams: list[str], cameras: None | list[Camera] = None, resize: None | tuple[int, int] = None):
        """
        Create a new offline video source. A list of streams pointing to video
        files must be given together with associated camera parameters.
        """
        self.streams = streams
        self.cameras = cameras or [Camera() for _ in streams]
        self.resize = resize

    def start(self):
        self.caps = [cv2.VideoCapture(s) for s in self.streams]
        fps = self.caps[0].get(cv2.CAP_PROP_FPS)
        for cap, path in zip(self.caps, self.streams):
            if not cap.isOpened():
                raise RuntimeError(f"could not open video source {path}")
            if fps != cap.get(cv2.CAP_PROP_FPS):
                raise RuntimeError(f"videos do not have the same frame rate")

    def release(self):
        for cap in self.caps:
            cap.release()

    def next_frames(self) -> tuple[None, None, None] | tuple[float, list[torch.Tensor], list[Camera]]:
        device = self.cameras[0].intrinsic.device
        cameras = []
        frames = []
        for cam, cap in zip(self.cameras, self.caps):
            if not cap.isOpened():
                return None, None, None
            success, frame = cap.read()
            if not success:
                return None, None, None
            if self.resize is not None:
                cam = copy.copy(cam)
                cam.resize((frame.shape[1], frame.shape[0]), self.resize)
                frame = cv2.resize(frame, self.resize)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(torch.tensor(frame, device=device))
            cameras.append(cam)
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        return timestamp, frames, cameras

    def to(self, *args, **kargs):
        """ Apply the PyTorch `.to` method to all contained cameras. """
        self.cameras = [cam.to(*args, **kargs) for cam in self.cameras]
        return self


class ThreadedVideoStream:
    """
    A class that encapsulates a single video stream in the `OnlineVideoSource`
    video input source. This spawns a thread that pulls the latest frame from
    a camera as fast as possible, and then returns the latest one.
    """

    def __init__(self, stream: int | str):
        self.stream = stream
        self.lock = threading.Lock()

    def update(self):
        while self.running and self.cap.isOpened():
            grabbed = self.cap.grab()
            frame_time = time.time()
            if grabbed:
                _, frame = self.cap.retrieve()
                with self.lock:
                    self.grabbed = True
                    self.frame = frame
                    self.frame_time = frame_time
        self.running = False

    def start(self):
        self.cap = cv2.VideoCapture(self.stream)
        self.grabbed, self.frame = self.cap.read()
        self.frame_time = time.time()
        self.running = True
        self.thread = threading.Thread(target=self.update)
        self.thread.daemon = True
        self.thread.start()

    def release(self):
        self.running = False
        self.thread.join()
        self.cap.release()

    def next_frame(self) -> tuple[None, None] | tuple[float, cv2.typing.MatLike]:
        """
        Return the most recently abutted from from this video stream together
        with the associated timestamp in seconds.
        """
        if not self.running:
            return None, None
        with self.lock:
            return self.frame_time, self.frame


class OnlineVideoSource(VideoSource):
    """
    This class implements live input from cameras using OpenCV. Importantly, this
    is intended for online processing, so for the timing functions it uses the
    realtime instead of the framerate of the camera. Since cameras queried this
    way will not be synchronized, we simply return the median timestamp across
    all returned frames.
    """

    def __init__(self, streams: list[int | str], cameras: None | list[Camera] = None, resize: None | tuple[int, int] = None):
        """
        Create a new online video source. A list of streams pointing to either
        the camera index, or possibly an ip camera address must be given,
        together with associated camera parameters.
        """
        self.streams = [ThreadedVideoStream(s) for s in streams]
        self.cameras = cameras or [Camera() for _ in streams]
        self.resize = resize

    def start(self):
        for stream in self.streams:
            stream.start()

    def release(self):
        for stream in self.streams:
            stream.release()

    def next_frames(self) -> tuple[None, None, None] | tuple[float, list[torch.Tensor], list[Camera]]:
        device = self.cameras[0].intrinsic.device
        cameras = []
        frames = []
        timestamps = []
        for cam, cap in zip(self.cameras, self.streams):
            timestamp, frame = cap.next_frame()
            if timestamp is None or frame is None:
                return None, None, None
            if self.resize is not None:
                cam = copy.copy(cam)
                cam.resize((frame.shape[1], frame.shape[0]), self.resize)
                frame = cv2.resize(frame, self.resize)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(torch.tensor(frame, device=device))
            timestamps.append(timestamp)
            cameras.append(cam)
        return float(np.median(timestamps)), frames, self.cameras

    def to(self, *args, **kargs):
        """ Apply the PyTorch `.to` method to all contained cameras. """
        self.cameras = [cam.to(*args, **kargs) for cam in self.cameras]
        return self
