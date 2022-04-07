from typing import Tuple, Union, Optional

from scipy.sparse import issparse, csr_matrix
from moscot.backends.ott._output import SinkhornOutput as Output

from ott.core import sinkhorn
from ott.geometry import pointcloud
import numpy as np
import jax.numpy as jnp
import numpy.typing as npt


def distance_between_pushed_masses(
    gex_data_source: npt.ArrayLike,
    gex_data_target: npt.ArrayLike,
    output: Union[npt.ArrayLike, csr_matrix, Output],
    true_coupling: Union[npt.ArrayLike, csr_matrix],
    eps: float = 0.1,
    seed: Optional[int] = None,
    n_samples: Optional[int] = None,
) -> float:
    source_space_cost = _distance_pushed_masses(
        gex_data_source, output, true_coupling, forward=False, eps=eps, random=False, seed=seed, n_samples=n_samples
    )
    target_space_cost = _distance_pushed_masses(
        gex_data_target, output, true_coupling, forward=True, eps=eps, random=False, seed=seed, n_samples=n_samples
    )
    independent_coupling_source_space_cost = _distance_pushed_masses(
        gex_data_source, output, true_coupling, forward=False, random=True, seed=seed, n_samples=n_samples
    )
    independent_coupling_target_space_cost = _distance_pushed_masses(
        gex_data_target, output, true_coupling, forward=True, random=True, seed=seed, n_samples=n_samples
    )
    source_error = source_space_cost / independent_coupling_source_space_cost
    target_error = target_space_cost / independent_coupling_target_space_cost
    mean_error = (source_error + target_error) / 2
    return mean_error


def _get_masses_moscot(
    i: int,
    output: Union[npt.ArrayLike, csr_matrix, Output],
    true_coupling: Union[npt.ArrayLike, csr_matrix],
    n: int,
    m: int,
    forward: bool,
    random: bool,
) -> Tuple[npt.ArrayLike]:
    if forward:
        pushed_mass_true = true_coupling[i, :]
        pushed_mass_true /= pushed_mass_true.sum()
        mass = np.zeros(n, dtype="float64")
        mass[i] = 1
        pushed_mass = output.push(mass).squeeze()
        weight_factor = jnp.sum(pushed_mass)
        if random:
            pushed_mass = output.b[:, -1]
            pushed_mass /= pushed_mass.sum()
        else:
            pushed_mass /= weight_factor
    else:
        pushed_mass_true = true_coupling.T[i, :]
        pushed_mass_true /= pushed_mass_true.sum()
        mass = np.zeros(m, dtype="float64")
        mass[i] = 1
        pushed_mass = output.pull(mass).squeeze()
        weight_factor = jnp.sum(pushed_mass)
        if random:
            pushed_mass = output.a[:, -1]
            pushed_mass /= pushed_mass.sum()
        else:
            pushed_mass /= weight_factor
    return pushed_mass_true.astype("float64"), pushed_mass, weight_factor


def _get_masses_ndarray(
    i: int,
    output: Union[npt.ArrayLike, csr_matrix, Output],
    true_coupling: Union[npt.ArrayLike, csr_matrix],
    forward: bool,
    random: bool,
) -> Tuple[npt.ArrayLike, ...]:
    if forward:
        pushed_mass_true = true_coupling[i, :]
        pushed_mass_true /= pushed_mass_true.sum()
        weight_factor = output[i, :].sum()
        if random:
            pushed_mass = output.sum(0)
            pushed_mass /= pushed_mass.sum()
        else:
            pushed_mass = output[i, :]
            pushed_mass /= weight_factor
    else:
        pushed_mass_true = true_coupling.T[i, :]
        pushed_mass_true /= pushed_mass_true.sum()
        weight_factor = output.T[i, :].sum()
        if random:
            pushed_mass = output.sum(1)
            pushed_mass /= pushed_mass.sum()
        else:
            pushed_mass = output.T[i, :]
            pushed_mass /= weight_factor
    return pushed_mass_true.astype("float64"), pushed_mass.astype("float64"), weight_factor


def _distance_pushed_masses(
    gex_data: npt.ArrayLike,
    output: Union[npt.ArrayLike, csr_matrix, Output],
    true_coupling: Union[npt.ArrayLike, csr_matrix],
    forward: bool,
    eps: float = 0.5,
    random: bool = False,
    seed: Optional[int] = None,
    n_samples: Optional[int] = None,
) -> float:
    rng = np.random.RandomState(seed=seed)
    n, m = output.shape if isinstance(output, np.ndarray) else output.solution.shape
    wasserstein_d = 0
    total_weight = 0
    if n_samples is None:
        samples = range(n if forward else m)
    else:
        samples = rng.choice(n if forward else m, size=n_samples)
    for i in samples:
        print(i)
        if isinstance(output, np.ndarray):
            pushed_mass_true, pushed_mass, weight_factor = _get_masses_ndarray(
                i, output, true_coupling, forward, random
            )
        elif isinstance(output, Output):
            pushed_mass_true, pushed_mass, weight_factor = _get_masses_moscot(
                i, output, true_coupling, n, m, forward, random
            )
        if issparse(pushed_mass_true):
            pushed_mass_true = np.squeeze(pushed_mass_true.A)
        geom = pointcloud.PointCloud(gex_data, gex_data, epsilon=eps, scale_cost="mean")
        out = sinkhorn.sinkhorn(geom, pushed_mass_true, pushed_mass, max_iterations=1e7)
        wasserstein_d += float(out.reg_ot_cost) * weight_factor
        total_weight = +weight_factor
        del geom
        del out

    return wasserstein_d / (len(samples) * len(pushed_mass) * total_weight)
