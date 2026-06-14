
import configparser
from dataclasses import dataclass

import torch


@dataclass
class Camera:
    """
    A simple class to encapsulate all of the intrinsic and extrinsic camera
    calibration parameters necessary for mapping a 3d point into the camera
    perspective. Uses PyTorch tensors.

    >>> cam = Camera()
    >>> cam.rotation.shape
    torch.Size([3, 3])
    >>> cam.translation
    tensor([0., 0., 0.])
    """
    # Extrinsics
    rotation: torch.Tensor = torch.eye(3)
    translation: torch.Tensor = torch.zeros(3)
    # Intrinsics
    intrinsic: torch.Tensor = torch.eye(3)
    distortion: torch.Tensor = torch.zeros(5)

    def load_ini(self, path: str):
        """
        Load the calibration parameters from a file in a .ini format. This will
        replace the values in this instance with those in the file. If some
        sections are missing in the file, those parameters will simply be left
        as is.

        >>> import tempfile, os
        >>> ini_content = '''[Extrinsics]
        ... R11 = 1.0\\nR12 = 0.0\\nR13 = 0.0
        ... R21 = 0.0\\nR22 = 1.0\\nR23 = 0.0
        ... R31 = 0.0\\nR32 = 0.0\\nR33 = 1.0
        ... T1 = 1.0\\nT2 = 2.0\\nT3 = 3.0
        ... [Intrinsics]
        ... f = 500.0\\nmu = 1.0\\nmv = 1.0\\nu0 = 320.0\\nv0 = 240.0
        ... [Distortion=pinhole]
        ... k1 = 0.1\\nk2 = 0.02\\np1 = 0.001\\np2 = 0.002\\nk3 = 0.003
        ... '''
        >>> with tempfile.NamedTemporaryFile('w', delete=False) as f:
        ...     temp_path, _ = f.name, f.write(ini_content)
        >>> cam = Camera()
        >>> cam.load_ini(temp_path)
        >>> os.remove(temp_path)
        >>> cam.translation
        tensor([-1., -2., -3.])
        >>> cam.intrinsic[0, 2].item()
        320.0
        """
        config = configparser.ConfigParser()
        with open(path, "r") as file:
            config.read_file(file)
        if config.has_section("Extrinsics"):
            self.rotation = torch.tensor([
                [float(config["Extrinsics"][f"R{row + 1}{col + 1}"])
                 for col in range(3)]
                for row in range(3)
            ])
            t = torch.tensor([
                float(config["Extrinsics"][f"T{row + 1}"]) for row in range(3)
            ])
            self.translation = -self.rotation @ t
        if config.has_section("Intrinsics"):
            section = config["Intrinsics"]
            f = float(section["f"])
            self.intrinsic = torch.tensor([
                [-f * float(section["mu"]), 0.0, float(section["u0"])],
                [0.0, -f * float(section["mv"]), float(section["v0"])],
                [0.0, 0.0, 1.0],
            ])
        if config.has_section("Distortion=pinhole"):
            section = config["Distortion=pinhole"]
            self.distortion = torch.tensor([
                float(section["k1"]), float(section["k2"]),
                float(section["p1"]), float(section["p2"]),
                float(section["k3"]),
            ])

    def scale(self, scale: float):
        """
        Scale the camera extrinsics by the given scale.
        """
        self.translation = scale * self.translation

    def resize(self, old: tuple[int, int], new: tuple[int, int]):
        """
        Modify the camera intrinsics to accommodate a image resizing. 
        """
        s_x, s_y = new[0] / old[0], new[1] / old[1]
        self.intrinsic = torch.stack([
            s_x * self.intrinsic[0],
            s_y * self.intrinsic[1],
            self.intrinsic[2],
        ])

    def crop(self, x: int, y: int):
        """
        Modify the camera intrinsics to accommodate a image crop. You must
        specify the top left corner of the crop.
        """
        self.intrinsic = torch.stack([
            self.intrinsic[:, 0:2],
            self.intrinsic[:, 2] - torch.tensor([x, y, 0], dtype=torch.float),
        ], dim=1)

    def project_pinhole(self, points: torch.Tensor, eps=1e-5) -> torch.Tensor:
        """
        Project a set of 3d points to 2d locations on the cameras image plane.
        This computes normalized camera coordinates and does not take into acount
        camera intrinsics or distortion.

        >>> cam = Camera(translation=torch.tensor([0.0, 0.0, 1.0]))
        >>> pts = torch.tensor([1.0, 1.0, 1.0])
        >>> cam.project_pinhole(pts).tolist()
        [0.5, 0.5]
        """
        *Bs, M = points.shape
        points_cam = (self.rotation @ points.view(-1, 3, 1)).squeeze(-1) \
            + self.translation
        xy, z = points_cam[..., 0:2], points_cam[..., 2:3]
        z = torch.clamp(z, min=eps)
        return (xy / z).view(*Bs, M//3*2)

    def distortion_params(self, xy_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the distortions radial scale, as well as x and y tangential
        offsets for this cameras parameters.

        >>> cam = Camera(distortion=torch.tensor([0.1, 0.05, 0.0, 0.0, 0.0]))
        >>> xy_norm = torch.tensor([1.0, 1.0])
        >>> scale, xy_off = cam.distortion_params(xy_norm)
        >>> round(scale.item(), 5)
        1.4
        >>> xy_off.tolist()
        [0.0, 0.0]
        """
        k1, k2, _, _, k3 = self.distortion
        p12 = self.distortion[2:4]
        xy_norm_sqr = xy_norm * xy_norm
        r2 = torch.sum(xy_norm_sqr, dim=-1, keepdim=True)
        r4 = r2 * r2
        r6 = r4 * r2
        scale = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
        xy_prod = torch.prod(xy_norm, dim=-1, keepdim=True)
        xy_off = 2.0 * p12 * xy_prod + \
            torch.flip(p12, [0]) * (r2 + 2.0 * xy_norm_sqr)
        return scale, xy_off

    def distort_points(self, points: torch.Tensor) -> torch.Tensor:
        """
        Distort the given points in normalized camera coordinates so that they
        represent pixel coordinates on the camera.
        """
        # Apply Tsai distortion
        scale, xy_off = self.distortion_params(points)
        xy_dist = points * scale + xy_off
        # Apply camera intrinsics
        uv = (self.intrinsic[0:2, 0:2] @ xy_dist.unsqueeze(-1)).squeeze(-1) \
            + self.intrinsic[0:2, 2]
        return uv

    def project(self, points: torch.Tensor, eps=1e-7) -> torch.Tensor:
        """
        Project a set of 3d points to 2d locations on the cameras image plane. The
        last dimensions of the input should be the points, and the others can be
        an arbitrarily number of batch dimensions.

        >>> cam = Camera()
        >>> pts = torch.tensor([1.0, 2.0, 2.0])
        >>> cam.project(pts).tolist()
        [0.5, 1.0]
        """
        return self.distort_points(self.project_pinhole(points, eps))

    def undistort_points(self, points: torch.Tensor, num_iters: int = 10) -> torch.Tensor:
        """
        Undistort a set of 2d points on the cameras image plane from pixel space
        to normalized camera coordinates. The result space matches the output
        of the below `project_pinhole` method. Input pixels should have shape
        of (..., N*2).

        >>> cam = Camera()
        >>> pts = torch.tensor([[0.5, 1.0], [0.75, 15.0]])
        >>> cam.undistort_points(pts).tolist()
        [[0.5, 1.0], [0.75, 15.0]]
        """
        *Bs, M = points.shape
        xy = torch.linalg.solve(
            self.intrinsic[0:2, 0:2],
            (points.view(-1, 2) - self.intrinsic[0:2, 2]).unsqueeze(-1)
        ).squeeze(-1)
        xy_norm = xy.clone()
        for _ in range(num_iters):
            scale, xy_off = self.distortion_params(xy_norm)
            xy_norm = (xy - xy_off) / scale
        return xy_norm.view(*Bs, M)

    def undistort_covars(self, covars: torch.Tensor) -> torch.Tensor:
        """
        Approximately undistort a covariance matrix. This is useful for cases
        where we know covariances in pixel space, put we want to convert the
        values, and therefore also the covariances to normalized camera coordinates.
        This is useful for example in an extended Kalman filter, if we want to
        remove the camera distortion non-linearity from the observation step.
        Shape of `covars` is expected to be (..., N*2, N*2) for N points. The
        batch dimensions are assumed to be independent.

        >>> cam = Camera()
        >>> pts = torch.tensor([[[0.5, 1.0], [1.0, 0.5]], [[1.0, 0.0], [0.0, 2.0]]])
        >>> cam.undistort_covars(pts).tolist()
        [[[0.5, 1.0], [1.0, 0.5]], [[1.0, 0.0], [0.0, 2.0]]]
        >>> pts = torch.tensor([[[0.5, 1.0], [1.0, 0.5]]])
        >>> cam.undistort_covars(pts).tolist()
        [[[0.5, 1.0], [1.0, 0.5]]]
        """
        *_, M = covars.shape
        intr = torch.kron(
            torch.eye(M // 2, device=covars.device), self.intrinsic[0:2, 0:2])
        covars = torch.linalg.solve(intr, covars.mT).mT
        covars = torch.linalg.solve(intr, covars)
        return covars

    def to(self, *args, **kargs):
        """
        Apply the PyTorch `.to` method to all contained tensors.

        >>> cam = Camera()
        >>> _ = cam.to(torch.float64)
        >>> cam.rotation.dtype
        torch.float64
        """
        self.rotation = self.rotation.to(*args, **kargs)
        self.translation = self.translation.to(*args, **kargs)
        self.intrinsic = self.intrinsic.to(*args, **kargs)
        self.distortion = self.distortion.to(*args, **kargs)
        return self

    def __getitem__(self, key: str):
        if key == "R":
            return self.rotation.tolist()
        if key == "t":
            return self.translation.tolist()
        if key == "K":
            return self.intrinsic.tolist()
        raise KeyError


def inv_sqrt_sym(matrix: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """ Compute the inverse square-root of the given batch of matrices. """
    diag, vecs = torch.linalg.eigh(matrix)
    inv_sqrt = 1.0 / (torch.sqrt(diag) + eps)
    return vecs @ torch.diag_embed(inv_sqrt) @ vecs.mT


def triangulate_undistorted(cams: list[Camera], points: list[torch.Tensor], covars: None | list[torch.Tensor] = None) -> torch.Tensor:
    """
    Triangulate multiple points using multiple camera views. This is similar
    to `triangulate`, but the points must have be undistorted beforehand.

    >>> cam1 = Camera(translation=torch.tensor([0.9, 0.0, 0.0]))
    >>> cam2 = Camera(translation=torch.tensor([-1.1, 0.0, 0.0]))
    >>> p1 = torch.tensor([1.0, 0.1])
    >>> p2 = torch.tensor([-1.0, 0.1])
    >>> res = triangulate_undistorted([cam1, cam2], [p1, p2])
    >>> res.shape
    torch.Size([3])
    >>> [round(x, 2) for x in res.tolist()]
    [0.1, 0.1, 1.0]
    """
    mats = []
    vec = []
    for cam, pts, cov in zip(cams, points, covars or [None for _ in range(len(cams))]):
        r, t = cam.rotation, cam.translation
        A = r[0:2] - pts.unsqueeze(-1) * r[2]
        b = (pts * t[2] - t[0:2]).unsqueeze(-1)
        if cov is None:
            mats.append(A)
            vec.append(b)
        else:
            inv_sqrt_cov = inv_sqrt_sym(cov)
            mats.append(inv_sqrt_cov @ A)
            vec.append(inv_sqrt_cov @ b)
    lstsq = torch.linalg.lstsq(torch.cat(mats, dim=-2), torch.cat(vec, dim=-2))
    return lstsq.solution.squeeze(-1)


def triangulate(cams: list[Camera], points: list[torch.Tensor]) -> torch.Tensor:
    """
    Triangulate multiple points using multiple camera views. All input tensors
    must have the same shape, with the last dimension having size 2 and an arbitrary
    number of batch dimensions in front. The output will have the same batch
    dimensions but a final dimension of size 3.

    >>> cam1 = Camera(translation=torch.tensor([1.1, 0.0, 0.0]))
    >>> cam2 = Camera(translation=torch.tensor([-0.9, 0.0, 0.0]))
    >>> p1 = torch.tensor([1.0, -0.1])
    >>> p2 = torch.tensor([-1.0, -0.1])
    >>> res = triangulate([cam1, cam2], [p1, p2])
    >>> [round(x, 2) for x in res.tolist()]
    [-0.1, -0.1, 1.0]
    """
    xy = [cam.undistort_points(pts) for cam, pts in zip(cams, points)]
    return triangulate_undistorted(cams, xy)
