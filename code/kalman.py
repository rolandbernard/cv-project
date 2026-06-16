
from functools import lru_cache
from typing import Callable

import torch


class LinearPhysics:
    """
    This class implements a linear physic for a Kalman filter. The dynamics must
    be specified in continuous form, with the discretization computed and cached
    on demand. Both the prediction and the update step and in theory work on batches
    of targets.
    """

    def __init__(self, dyn_mat: torch.Tensor, dyn_cov: torch.Tensor, init_mean: torch.Tensor, init_cov: torch.Tensor):
        super().__init__()
        self.dyn_mat = dyn_mat
        self.dyn_cov = dyn_cov
        self.init_mean = init_mean
        self.init_cov = init_cov
        self.get_dyn = lru_cache()(self._get_dyn)

    def _get_dyn(self, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
        """ Create a new dynamics and covariance matrix for the given timestamp. """
        return discretize(dt, self.dyn_mat, self.dyn_cov)

    def predict(self, dt: float, mean: torch.Tensor, cov: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply the internal prediction logic from this Kalman filter for the given
        amount of time having passed and return the new means and covariances for
        the targets.
        """
        dyn_mat, dyn_cov = self.get_dyn(dt)
        return predict(mean, cov, dyn_mat, dyn_cov)

    def to(self, *args, **kargs):
        """ Apply the PyTorch `.to` method to the contained model. """
        self.dyn_mat = self.dyn_mat.to(*args, **kargs)
        self.dyn_cov = self.dyn_cov.to(*args, **kargs)
        self.init_mean = self.init_mean.to(*args, **kargs)
        self.init_cov = self.init_cov.to(*args, **kargs)
        self.get_dyn.cache_clear()
        return self


class ConstrainedPhysics(LinearPhysics):
    """
    A physics model that incorporates rigid body constraints between keypoints.
    The state is assumed to contain positions, velocities, and limb lengths.
    """

    def __init__(
        self, dyn_mat: torch.Tensor, dyn_cov: torch.Tensor, init_mean: torch.Tensor,
        init_cov: torch.Tensor, constraints: torch.Tensor, point_mix: torch.Tensor,
        constr_cov: torch.Tensor, num_keypoint: int = 17
    ):
        super().__init__(dyn_mat, dyn_cov, init_mean, init_cov)
        self.constraints = constraints
        self.point_mix = point_mix
        self.num_keypoint = num_keypoint
        self.constr_cov = constr_cov

    def compute_distances(self, x: torch.Tensor) -> torch.Tensor:
        """ Compute the constrained distances. """
        *Bs, _ = x.shape
        points = x[:self.num_keypoint*3].view(*Bs, -1, 3)
        points = torch.concat([
            points,
            (points[..., self.point_mix[:, 0], :]
             + points[..., self.point_mix[:, 1], :]) * 0.5
        ], dim=-2)
        pi = points[..., self.constraints[:, 0], :]
        pj = points[..., self.constraints[:, 1], :]
        return torch.linalg.vector_norm(pi - pj, dim=-1)

    def pseudo_obs(self, x: torch.Tensor) -> torch.Tensor:
        """ Compute the constraint violation. """
        return self.compute_distances(x) - x[..., self.constraints[:, 2]]

    def to(self, *args, **kargs):
        super().to(*args, **kargs)
        self.constraints = self.constraints.to(*args, **kargs)
        self.point_mix = self.point_mix.to(*args, **kargs)
        self.constr_cov = self.constr_cov.to(*args, **kargs)
        return self


def discretize(dt: float, dyn_mat: torch.Tensor, dyn_cov: torch.Tensor):
    """ Discretize the given continuous-time matrices using the given time. """
    # Use a second order approximation for now.
    N, N = dyn_mat.shape
    dyn_mat = torch.eye(N, device=dyn_mat.device) + dyn_mat * dt \
        + dyn_mat @ dyn_mat * (dt * dt * 0.5)
    dyn_cov = dyn_cov * dt \
        + (dyn_mat @ dyn_cov + dyn_cov @ dyn_mat.mT) * (dt * dt * 0.5)
    return dyn_mat, dyn_cov


single_batch_block_diag = torch.vmap(torch.block_diag)


def batched_block_diag(mats: list[torch.Tensor]) -> torch.Tensor:
    *Bs, _, _ = mats[0].shape
    batched = single_batch_block_diag(*[
        mat.view(-1, *mat.shape[-2:]) for mat in mats
    ])
    _, N, N = batched.shape
    return batched.view(*Bs, N, N)


def merge_obs(
    obs_mat: list[torch.Tensor], obs_mean: list[torch.Tensor], obs_cov: list[torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Merge multiple observations into single matrix assuming the observations are
    independent. This allows multiple different observations to be performed with
    a single update step.
    """
    comb_mat = torch.concat(obs_mat, dim=-2)
    comb_mean = torch.concat(obs_mean, dim=-1)
    comb_cov = batched_block_diag(obs_cov)
    return comb_mat, comb_mean, comb_cov


def emerge_obs(
    obs: list[Callable[[torch.Tensor], torch.Tensor]], obs_mean: list[torch.Tensor], obs_cov: list[torch.Tensor]
) -> tuple[Callable[[torch.Tensor], torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    Merge multiple observations into single observation. Version for Extended
    Kalman filters (intended for the auto-differentiating version).
    """
    def comb_obs(x):
        return torch.concat([o(x) for o in obs], dim=-1)
    comb_mean = torch.concat(obs_mean, dim=-1)
    comb_cov = batched_block_diag(obs_cov)
    return comb_obs, comb_mean, comb_cov


def predict(mean: torch.Tensor, cov: torch.Tensor, dyn_mat: torch.Tensor, dyn_cov: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply the prediction logic from this Kalman filter for the dynamics matrix
    and covariances having passed and return the new means and covariances for
    the targets.
    """
    return (dyn_mat @ mean.unsqueeze(-1)).squeeze(-1), dyn_mat @ cov @ dyn_mat.mT + dyn_cov


def update_res(
    mean: torch.Tensor, cov: torch.Tensor, obs_mat: torch.Tensor, obs_res: torch.Tensor, obs_cov: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply an update when explicitly given the observation residual. This method
    can be reused both for linear and Extended Kalman filtering.
    """
    inov = obs_mat @ cov @ obs_mat.mT + obs_cov
    gain = cov @ torch.linalg.solve(inov.mT, obs_mat).mT
    *_, N, N = cov.shape
    return (
        mean + (gain @ obs_res.unsqueeze(-1)).squeeze(-1),
        (torch.eye(N, device=gain.device) - gain @ obs_mat) @ cov
    )


def update(
    mean: torch.Tensor, cov: torch.Tensor, obs_mat: torch.Tensor, obs_mean: torch.Tensor, obs_cov: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply an update given some observation. The observation is given using mean,
    covariance, and the matrix to extract it from the state.
    """
    res = obs_mean - (obs_mat @ mean.unsqueeze(-1)).squeeze(-1)
    return update_res(mean, cov, obs_mat, res, obs_cov)


def eupdate_ex(
    mean: torch.Tensor, cov: torch.Tensor, obs_mean: torch.Tensor, obs_cov: torch.Tensor,
    obs: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply update using Extended Kalman filtering give an explicit formulation of
    the Jacobian as a function.
    """
    jac, pred = obs(mean)
    res = obs_mean - pred
    return update_res(mean, cov, jac, res, obs_cov)


def batched_jacobian(f: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """ Compute the Jacobian over arbitrarily batch dimension of the given function f. """
    def func(x):
        res = f(x)
        return res, res
    *Bs, N = x.shape
    jac, val = torch.vmap(torch.func.jacrev(func, has_aux=True))(x.view(-1, N))
    *_, M, N = jac.shape
    return jac.reshape(*Bs, M, N), val.reshape(*Bs, M)


def eupdate(
    mean: torch.Tensor, cov: torch.Tensor,
    obs_mean: torch.Tensor, obs_cov: torch.Tensor, obs: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply update using Extended Kalman filtering give an observation function,
    but using PyTorch functionality to automatically compute the Jacobian.
    """
    return eupdate_ex(mean, cov, obs_mean, obs_cov, lambda x: batched_jacobian(obs, x))
