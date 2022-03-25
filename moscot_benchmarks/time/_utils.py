from typing import Tuple, Union, Optional

from scipy.sparse import csr_matrix
from moscot.backends.ott._output import SinkhornOutput as Output

from ott.core import sinkhorn
from ott.geometry import pointcloud
import numpy as np
import numpy.typing as npt


def distance_between_pushed_masses(
    gex_data_source: npt.ArrayLike,
    gex_data_target: npt.ArrayLike,
    output: Union[npt.ArrayLike, csr_matrix, Output],
    true_coupling: Union[npt.ArrayLike, csr_matrix],
    eps: float = 0.1,
    seed: Optional[int] = None,
) -> float:
    source_cost = _distance_pushed_masses(
        gex_data_source, output, true_coupling, forward=False, eps=eps, random=False, seed=seed
    )
    target_cost = _distance_pushed_masses(
        gex_data_target, output, true_coupling, forward=True, eps=eps, random=False, seed=seed
    )

    independent_coupling_source_cost = _distance_pushed_masses(
        gex_data_source, output, true_coupling, forward=False, random=True
    )
    independent_coupling_target_cost = _distance_pushed_masses(
        gex_data_target, output, true_coupling, forward=True, random=True
    )

    source_error = source_cost / independent_coupling_source_cost
    target_error = target_cost / independent_coupling_target_cost
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
        if random:
            pushed_mass = output.b if forward else output.a
        else:
            mass = np.zeros(n)
            mass[i] = 1
            pushed_mass = output.push(mass)
    else:
        pushed_mass_true = true_coupling.T[i, :]
        if random:
            pushed_mass = output.a
        else:
            mass = np.zeros(n)
            mass[i] = 1
            pushed_mass = output.push(mass)
        mass = np.zeros(m)
        mass[i] = 1
        pushed_mass = output.pull(mass)
    return pushed_mass_true, pushed_mass


def _get_masses_ndarray(
    i: int,
    output: Union[npt.ArrayLike, csr_matrix, Output],
    true_coupling: Union[npt.ArrayLike, csr_matrix],
    forward: bool,
    random: bool,
) -> Tuple[npt.ArrayLike]:
    if forward:
        pushed_mass_true = true_coupling[i, :]
        if random:
            pushed_mass = output.sum(0)
        else:
            pushed_mass = output[i, :]
    else:
        pushed_mass_true = true_coupling.T[i, :]
        if random:
            pushed_mass = output.sum(1)
        else:
            pushed_mass = output.T[i, :]
    return pushed_mass_true, pushed_mass


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
    n, m = output.shape if isinstance(output, np.ndarray) else output.shape
    wasserstein_d = 0
    if n_samples is None:
        samples = range(n if forward else m)
    else:
        samples = rng.choice(n if forward else m, size=n_samples)

    for i in samples:
        if isinstance(output, np.ndarray):
            pushed_mass_true, pushed_mass = _get_masses_ndarray(i, output, true_coupling, forward, random)
        elif isinstance(output, Output):
            pushed_mass_true, pushed_mass = _get_masses_moscot(i, output, true_coupling, n, m, forward, random)
        weight_factor = pushed_mass.sum()
        pushed_mass = pushed_mass / weight_factor
        pushed_mass_true /= pushed_mass_true.sum()
        geom = pointcloud.PointCloud(gex_data, gex_data, epsilon=eps)
        out = sinkhorn.sinkhorn(geom, pushed_mass_true, pushed_mass)
        wasserstein_d += float(out.reg_ot_cost) * weight_factor

    return wasserstein_d / len(samples)
