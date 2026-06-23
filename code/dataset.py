
import os
import io
import json
import zipfile
import tarfile
import urllib.request

import gdown
import torch

import source
from camera import Camera


def download_file(filename: str, data_url: str, first_n: None | int = None):
    """ Download a file from the given url and put it into the given filename. """
    try:
        with urllib.request.urlopen(data_url) as req_stream:
            data = req_stream.read(first_n)
            with open(filename, "wb") as file:
                file.write(data)
    except:
        print(f"warning: skipped {data_url}")


def download_zip(folder: str, data_url: str):
    """ Download a zip archive from the given url and extract it into the given folder. """
    try:
        with urllib.request.urlopen(data_url) as req_stream:
            with zipfile.ZipFile(io.BytesIO(req_stream.read())) as zip_file:
                zip_file.extractall(folder)
    except:
        print(f"warning: skipped {data_url}")


def download_tar(folder: str, data_url: str):
    """ Download a tar archive from the given url and extract it into the given folder. """
    try:
        with urllib.request.urlopen(data_url) as req_stream:
            with tarfile.open(fileobj=io.BytesIO(req_stream.read()), mode="r:*") as zip_file:
                zip_file.extractall(folder)
    except:
        print(f"warning: skipped {data_url}")


class SalsaDataset:
    """ Class for handling the Salsa dataset (https://tev.fbk.eu/resources/salsa). """

    fps = 15
    scenes: list[str] = ["PosterSession", "CocktailParty"]

    def __init__(self, path: str = "./data/salsa"):
        """ Create an instance of the class with the data stored in the given directory. """
        self.path = path

    def download(self):
        """
        Download the dataset, including video files and camera calibration
        parameters from Google Drive. Download is skipped if already present.
        """
        os.makedirs(self.path, exist_ok=True)
        camera_ids = ["1DYHJoTZtKDzv7HIIZVGK3PChQi-XaQx1", "1JZT_SVSb1o1iOy3I6bzkcwId4WnH14UZ",
                      "164H1WzfinUbPiCuPNecK4_d6gyc9qStk", "1cUU9n1webV3OGJcvFD_xQN1tPWzlxYJr"]
        for i, id in enumerate(camera_ids):
            gdown.cached_download(  # type: ignore
                id=id, path=f"{self.path}/cam{i}.ini")
        poster_ids = ["1F-Q-t2UlGrK6GEl5Df0T72Sb_CxX60fg", "1_Vx1HO4UUBQMsBR8sk4LeiTUTZs4OB-4",
                      "1ZzGQH3CXFg6AbwkcMWkJYGTJ0qlldwRM", "1Kaviox0ZQHq0UYcIjkGIjp7z1fX7kNbh"]
        os.makedirs(f"{self.path}/PosterSession", exist_ok=True)
        for i, id in enumerate(poster_ids):
            gdown.cached_download(  # type: ignore
                id=id, path=f"{self.path}/PosterSession/cam{i}.avi")
        party_ids = ["18h6z8DzcFxbJVZeWuUK571bO710CwR9R", "1_0AQWAJiKV3E--Dki1mpLZPAdrU2JByg",
                     "1y02UrxsyLom_OkBwiQAJk7iTqU1tsZo2", "1panv-zguLPDquar5kmGo_akozMoNYpBe"]
        os.makedirs(f"{self.path}/CocktailParty", exist_ok=True)
        for i, id in enumerate(party_ids):
            gdown.cached_download(  # type: ignore
                id=id, path=f"{self.path}/CocktailParty/cam{i}.avi")

    def get_source(self, name: str = "PosterSession", cams: int | list[int] = 4) -> source.OfflineVideoSource:
        """
        Load one of the two video sequences from the dataset into a video source
        for further processing. The name can be wither "PosterSession" (default)
        or "CocktailParty".
        """
        streams = [f"{self.path}/{name}/cam{i}.avi"
                   for i in (cams if isinstance(cams, list) else range(cams))]
        cameras = []
        for i in cams if isinstance(cams, list) else range(cams):
            cameras.append(Camera.from_file(f"{self.path}/cam{i}.ini"))
        return source.OfflineVideoSource(streams, cameras)


class H3wbDataset:
    """ Class for handling the H3WB dataset (https://github.com/wholebody3d/wholebody3d). """

    fps = 10

    def __init__(self, path: str = "./data/h3wb"):
        """ Create an instance of the class with the data stored in the given directory. """
        self.path = path

    def download(self):
        """
        Download the dataset from Google Drive. Download is skipped if already present.
        Note that we only download the annotations, not the images. This means
        this dataset can be used in this project only for the training of the
        learnable Kalman filter, not for end-to-end evaluation.
        """
        os.makedirs(self.path, exist_ok=True)
        gdown.cached_download(  # type: ignore
            id="1LZh4Jsg3_ZKBF0iEPiexzoGHE4srLgfC", path=f"{self.path}/h3wb_train.npz")


class D3pwDataset:
    """ Class for handling the 3DPW dataset (https://virtualhumans.mpi-inf.mpg.de/3DPW/). """

    endpoint: str = "https://virtualhumans.mpi-inf.mpg.de/3DPW"
    fps = 30

    def __init__(self, path: str = "./data/3dpw"):
        """ Create an instance of the class with the data stored in the given directory. """
        self.path = path

    @property
    def scenes(self) -> list[str]:
        return os.listdir(f"{self.path}/imageFiles")

    def download(self):
        """ Download the dataset from official source. Download is skipped if already present. """
        os.makedirs(self.path, exist_ok=True)
        if not os.path.exists(f"{self.path}/imageFiles"):
            download_zip(self.path, f"{self.endpoint}/imageFiles.zip")
        if not os.path.exists(f"{self.path}/sequenceFiles"):
            download_zip(self.path, f"{self.endpoint}/sequenceFiles.zip")


class CmuPanopticDataset:
    """ Class for handling the CMU Panoptic dataset (http://domedb.perception.cs.cmu.edu/index.html). """

    endpoint: str = "http://domedb.perception.cs.cmu.edu/webdata/dataset"
    hd_fps = 29.97
    vga_fps = 25
    coco17_indices = [1, 15, 17, 16, 18, 3, 9, 4,
                      10, 5, 11, 6, 12, 7, 13, 8, 14]
    scenes: list[str] = [
        "171204_pose1", "171204_pose2", "171204_pose3", "171204_pose4", "171204_pose5",
        "171204_pose6", "171026_pose1", "171026_pose2", "171026_pose3",
        "170221_haggling_b1", "170221_haggling_b2", "170221_haggling_b3", "170221_haggling_m1",
        "170221_haggling_m2", "170221_haggling_m3", "170224_haggling_a1", "170224_haggling_a2",
        "170224_haggling_a3", "170224_haggling_b1", "170224_haggling_b2", "170224_haggling_b3",
        "170228_haggling_a1", "170228_haggling_a2", "170228_haggling_a3", "170228_haggling_b1",
        "170228_haggling_b2", "170228_haggling_b3", "170404_haggling_a1", "170404_haggling_a2",
        "170404_haggling_a3", "170404_haggling_b1", "170404_haggling_b2", "170404_haggling_b3",
        "170407_haggling_a1", "170407_haggling_a2", "170407_haggling_a3", "170407_haggling_b1",
        "170407_haggling_b2", "170407_haggling_b3",
        "171026_cello1", "171026_cello2", "171026_cello3", "161029_flute1",
        "161029_piano1", "161029_piano2", "161029_piano3", "161029_piano4",
        "160906_band1", "160906_band2", "160906_band3",
        "160422_ultimatum1", "160226_ultimatum1", "160224_ultimatum1", "160224_ultimatum2",
        "160422_mafia2", "160226_mafia1", "160226_mafia2", "160224_mafia1", "160224_mafia2",
        "160422_haggling1", "160226_haggling1", "160224_haggling1",  "161202_haggling1",
        "170307_dance1", "170307_dance2", "170307_dance3", "170307_dance4", "170307_dance5",
        "170307_dance6", "160317_moonbaby1", "160317_moonbaby2", "160317_moonbaby3",
        "170915_toddler2", "170915_toddler3", "170915_toddler4", "160906_ian5", "160906_ian3",
        "160906_ian2", "160906_ian1", "160401_ian3", "160401_ian2", "160401_ian1",
        "170915_office1", "170407_office2", "160906_pizza1", "161029_tools1",
        "161029_build1", "161029_sports1",
    ]
    val_scenes: list[str] = [
        '171204_pose5', '171026_pose2', '170221_haggling_m2', '170224_haggling_b2',
        '170228_haggling_b2', '170404_haggling_b2', '161029_piano3',
    ]
    test_scenes: list[str] = [
        "171204_pose6", "161029_piano4", "170915_office1", "161029_build1", "160224_haggling1",
    ]
    vga_panels = [
        1, 19, 14, 6, 16, 9, 5, 10, 18, 15, 3, 8, 4, 20, 11, 13, 7, 2, 17, 12, 9, 5, 6, 3, 15, 2, 12, 14, 16, 10, 4, 13, 20, 8, 17, 19,
        18, 9, 4, 6, 1, 20, 1, 11, 7, 7, 14, 15, 3, 2, 16, 13, 3, 15, 17, 9, 20, 19, 8, 11, 5, 8, 18, 10, 12, 19, 5, 6, 16, 12, 4, 6, 20,
        13, 4, 10, 15, 12, 17, 17, 16, 1, 5, 3, 2, 18, 13, 16, 8, 19, 13, 11, 10, 7, 3, 2, 18, 10, 1, 17, 10, 15, 14, 4, 7, 9, 11, 7,
        20, 14, 1, 12, 1, 6, 11, 18, 7, 8, 9, 3, 15, 19, 4, 16, 18, 1, 11, 8, 4, 10, 20, 13, 6, 16, 7, 6, 16, 17, 12, 5, 17, 4, 8, 20,
        12, 17, 14, 2, 19, 14, 18, 15, 11, 11, 9, 9, 2, 13, 5, 15, 20, 18, 8, 3, 19, 11, 9, 2, 13, 14, 5, 9, 17, 9, 7, 6, 12, 16, 18,
        17, 13, 15, 17, 20, 4, 2, 2, 12, 4, 1, 16, 4, 11, 1, 16, 12, 18, 9, 7, 20, 1, 10, 10, 19, 5, 8, 14, 8, 4, 2, 9, 20, 14, 17, 11,
        3, 12, 3, 13, 6, 5, 16, 3, 5, 10, 19, 1, 11, 13, 17, 18, 2, 5, 14, 19, 15, 8, 8, 9, 3, 6, 16, 15, 18, 20, 4, 13, 2, 11, 20, 7,
        13, 15, 18, 10, 20, 7, 5, 2, 15, 6, 13, 4, 17, 7, 3, 19, 19, 3, 10, 2, 12, 10, 7, 7, 12, 11, 19, 8, 9, 6, 10, 6, 15, 10, 11, 3,
        16, 1, 5, 14, 6, 5, 13, 20, 14, 4, 18, 10, 14, 14, 1, 19, 8, 14, 19, 3, 6, 6, 3, 13, 17, 8, 20, 15, 18, 2, 2, 16, 5, 19, 15, 9,
        12, 19, 17, 8, 9, 3, 7, 1, 12, 7, 13, 1, 14, 5, 12, 11, 2, 16, 1, 18, 4, 18, 10, 16, 11, 7, 5, 1, 16, 9, 4, 15, 1, 7, 10, 14, 3,
        2, 17, 13, 19, 20, 15, 10, 4, 8, 16, 14, 5, 6, 20, 12, 5, 18, 7, 1, 8, 11, 5, 13, 1, 16, 14, 18, 12, 15, 2, 12, 3, 8, 12, 17,
        8, 20, 9, 2, 6, 9, 6, 12, 3, 20, 15, 20, 13, 3, 14, 1, 4, 8, 6, 10, 7, 17, 13, 18, 19, 10, 20, 12, 19, 2, 15, 10, 8, 19, 11, 19,
        11, 2, 4, 6, 2, 11, 8, 7, 18, 14, 4, 12, 14, 7, 9, 7, 11, 18, 16, 16, 17, 16, 15, 4, 15, 9, 17, 13, 3, 6, 17, 17, 20, 19, 11, 5,
        3, 1, 18, 4, 10, 5, 9, 13, 1, 5, 9, 6, 14
    ]
    vga_nodes = [
        1, 14, 3, 15, 12, 12, 8, 6, 13, 12, 12, 17, 7, 17, 21, 17, 4, 6, 12, 18, 2, 18, 5, 4, 2, 17, 12, 10, 18, 8, 18, 5, 10, 10, 17,
        1, 18, 7, 12, 9, 13, 5, 6, 18, 16, 9, 16, 8, 8, 10, 21, 22, 16, 16, 21, 16, 14, 6, 14, 11, 11, 20, 4, 22, 4, 22, 20, 19, 15,
        15, 15, 12, 2, 2, 3, 3, 20, 22, 5, 9, 3, 16, 23, 22, 20, 8, 8, 9, 2, 16, 14, 16, 16, 14, 1, 13, 16, 12, 10, 15, 18, 6, 13, 10,
        7, 10, 4, 1, 7, 21, 8, 6, 4, 7, 9, 10, 11, 8, 4, 6, 10, 4, 5, 6, 21, 21, 6, 6, 19, 20, 20, 20, 14, 19, 22, 22, 23, 19, 9, 15, 23,
        23, 23, 23, 19, 2, 8, 2, 8, 19, 19, 23, 23, 19, 19, 23, 24, 24, 2, 14, 12, 2, 12, 14, 12, 2, 14, 15, 11, 6, 6, 21, 4, 5, 5, 4,
        2, 10, 5, 10, 7, 3, 7, 9, 8, 9, 3, 7, 9, 9, 7, 2, 5, 5, 5, 5, 7, 8, 8, 4, 7, 11, 9, 7, 5, 3, 5, 7, 6, 8, 9, 8, 7, 8, 8, 3, 8, 7, 6, 11,
        7, 2, 9, 9, 2, 11, 12, 7, 4, 6, 6, 7, 4, 4, 9, 18, 1, 5, 6, 5, 10, 11, 5, 9, 6, 11, 12, 1, 10, 11, 6, 9, 7, 11, 5, 1, 2, 12, 11, 11,
        3, 3, 21, 11, 10, 2, 3, 10, 11, 19, 5, 11, 13, 12, 20, 13, 3, 5, 9, 11, 8, 4, 6, 4, 7, 12, 10, 8, 11, 19, 14, 23, 10, 1, 3, 12,
        4, 3, 10, 9, 2, 3, 20, 4, 11, 2, 20, 20, 2, 23, 10, 3, 22, 22, 1, 12, 12, 21, 4, 22, 23, 22, 18, 10, 18, 22, 11, 3, 18, 13, 18,
        3, 3, 13, 2, 1, 3, 20, 20, 4, 20, 14, 14, 20, 20, 14, 14, 22, 18, 21, 20, 22, 20, 22, 9, 22, 21, 21, 22, 21, 22, 20, 21, 21,
        21, 21, 23, 17, 21, 13, 20, 13, 13, 15, 17, 1, 23, 23, 23, 18, 13, 16, 15, 19, 17, 17, 22, 21, 17, 14, 1, 13, 13, 14, 14,
        16, 19, 17, 18, 1, 13, 18, 24, 19, 16, 13, 18, 18, 15, 23, 17, 14, 19, 17, 1, 19, 13, 19, 1, 15, 17, 13, 23, 13, 19, 24, 15,
        15, 19, 15, 17, 1, 16, 24, 21, 23, 14, 24, 15, 24, 24, 1, 16, 15, 24, 1, 17, 17, 15, 24, 1, 16, 16, 19, 13, 15, 22, 24, 23,
        17, 16, 18, 1, 24, 24, 24, 17, 24, 24, 17, 16, 24, 14, 15, 16, 15, 24, 24, 24, 18
    ]

    def __init__(self, path: str = "./data/panoptic"):
        """ Create an instance of the class with the data stored in the given directory. """
        self.path = path

    def download_scene(self, name: str, num_hd_cams: int = 0, num_vga_cams: int = 0, first_n: None | int = None):
        os.makedirs(f"{self.path}/{name}", exist_ok=True)
        if not os.path.exists(f"{self.path}/{name}/calibration.json"):
            download_file(f"{self.path}/{name}/calibration.json",
                          f"{self.endpoint}/{name}/calibration_{name}.json")
        if not os.path.exists(f"{self.path}/{name}/hdPose3d_stage1_coco19"):
            download_tar(
                f"{self.path}/{name}", f"{self.endpoint}/{name}/hdPose3d_stage1_coco19.tar")
        if not os.path.exists(f"{self.path}/{name}/vgaPose3d_stage1_coco19"):
            download_tar(
                f"{self.path}/{name}", f"{self.endpoint}/{name}/vgaPose3d_stage1_coco19.tar")
        for i in range(num_hd_cams):
            filename = f"hd_00_{i:02d}.mp4"
            if not os.path.exists(f"{self.path}/{name}/{filename}"):
                download_file(f"{self.path}/{name}/{filename}",
                              f"{self.endpoint}/{name}/videos/hd_shared_crf20/{filename}",
                              first_n)
        for i in range(num_vga_cams):
            filename = f"vga_{self.vga_panels[i]:02d}_{self.vga_nodes[i]:02d}.mp4"
            if not os.path.exists(f"{self.path}/{name}/{filename}"):
                download_file(f"{self.path}/{name}/{filename}",
                              f"{self.endpoint}/{name}/videos/vga_shared_crf10/{filename}",
                              first_n)

    def is_valid_scene(self, name: str, vga_gt=True, hd_gt=False, vga=True, hd=False) -> bool:
        """
        Determine whether for the given scene we have the desired data. Some
        scenes in the CMU Panoptic dataset don"t contain some or all of the data.
        """
        if not os.path.exists(f"{self.path}/{name}/calibration.json"):
            return False
        if vga_gt and not os.path.exists(f"{self.path}/{name}/vgaPose3d_stage1_coco19"):
            return False
        if hd_gt and not os.path.exists(f"{self.path}/{name}/hdPose3d_stage1_coco19"):
            return False
        if vga and not os.path.exists(f"{self.path}/{name}/vga_01_01.mp4"):
            return False
        if hd and not os.path.exists(f"{self.path}/{name}/hd_00_00.mp4"):
            return False
        return True

    def download(self, num_hd_cams: int = 0, num_vga_cams: int = 4, scenes: None | list[str] = None):
        """ Download the dataset from official source. Download is skipped if already present. """
        os.makedirs(self.path, exist_ok=True)
        for scene in scenes or self.scenes:
            self.download_scene(scene, num_hd_cams, num_vga_cams)

    def load_cam(self, calib, name: str) -> Camera:
        cam_calib = [c for c in calib["cameras"] if c["name"] == name][0]
        return Camera.from_dict(cam_calib)

    def get_source(self, scene: str, num_hd_cams: int = 0, num_vga_cams: int = 4) -> source.OfflineVideoSource:
        """
        Load one of the scenes from the dataset into a video source for further
        processing. The name should be one of the ones in `CmuPanopticDataset.scenes`.
        """
        with open(f"{self.path}/{scene}/calibration.json") as f:
            calib = json.load(f)
        streams = []
        cameras = []
        for i in range(num_hd_cams):
            name = f"00_{i:02d}"
            streams.append(f"{self.path}/{scene}/hd_{name}.mp4")
            cameras.append(self.load_cam(calib, name))
        for i in range(num_vga_cams):
            name = f"{self.vga_panels[i]:02d}_{self.vga_nodes[i]:02d}"
            streams.append(f"{self.path}/{scene}/vga_{name}.mp4")
            cameras.append(self.load_cam(calib, name))
        return source.OfflineVideoSource(streams, cameras)

    def load_ground_truth_vga(self, scene: str, num_vga_cams: int = 4) -> tuple[list[Camera], list[list], float]:
        """
        Load the ground truth data in the same format as produced in the evaluation
        application of the tracking system. This can be used to perform evaluation.
        For visualization purposes it also generates the first few camera positions.
        """
        with open(f"{self.path}/{scene}/calibration.json") as f:
            calib = json.load(f)
        cameras = []
        for i in range(num_vga_cams):
            name = f"{self.vga_panels[i]:02d}_{self.vga_nodes[i]:02d}"
            cameras.append(self.load_cam(calib, name))
        ann_path = f"{self.path}/{scene}/vgaPose3d_stage1_coco19"
        files = sorted(f for f in os.listdir(ann_path))
        last_idx = -1
        frames = []
        for file in files:
            idx = int(files[0][12:-5])
            frames.extend([[]] * (idx - last_idx - 1))
            with open(f"{ann_path}/{file}") as f:
                ann = json.load(f)
                frames.append([
                    {
                        "id": b["id"],
                        "kpts": torch.tensor(b["joints19"])
                        .view(19, 4)[self.coco17_indices, :3]
                        .tolist(),
                        "conf": torch.tensor(b["joints19"])
                        .view(19, 4)[self.coco17_indices, 3]
                        .tolist(),
                    } for b in ann["bodies"]
                ])
            last_idx = idx
        return cameras, frames, self.vga_fps
