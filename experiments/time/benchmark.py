from pathlib import Path
import sys

root = Path(__file__).parent.parent.parent.absolute()

sys.path.insert(0, str(root / "moscot_benchmarks"))
from typing import Any, Tuple, Union, Literal, Callable, Optional

from utils import benchmark_time, benchmark_memory
from sacred import Experiment
from scipy.stats import entropy
from scipy.sparse import csr_matrix
import seml
import numpy as np
from anndata import AnnData

ex = Experiment()
seml.setup_logger(ex)


def _benchmark_wot(
    C: np.ndarray,
    benchmark_f: Callable,
    validate_ot: bool,
    adata: AnnData,
    true_coupling: Optional[Union[np.ndarray, csr_matrix]],
    key: str,
    key_value_1: Any,
    key_value_2: Any,
    epsilon: float,
    lambda_1: float,
    lambda_2: float,
    threshold: float,
    max_iterations: int,
    seed: Optional[int] = None,
    n_val_samples: Optional[int] = None,
):
    import wot

    from jax.config import config

    config.update("jax_enable_x64", True)  # need this for "distance_between_pushed_masses"

    from time_utils import distance_between_pushed_masses

    ot_model = wot.ot.OTModel(
        adata,
        day_field=key,
        epsilon=epsilon,
        lambda1=lambda_1,
        lambda2=lambda_2,
        threshold=threshold,
        max_iter=max_iterations,
        local_pca=0,
    )

    benchmarked_f = benchmark_f(ot_model.compute_transport_map)
    benchmark_result, ot_result = benchmarked_f(key_value_1, key_value_2, cost_matrix=C)

    if validate_ot:
        gex_data_source = adata[adata.obs[key] == key_value_1].obsm["X_pca"]
        gex_data_target = adata[adata.obs[key] == key_value_2].obsm["X_pca"]
        error = distance_between_pushed_masses(
            gex_data_source, gex_data_target, ot_result.X, true_coupling, seed=seed, n_samples=n_val_samples
        )
        return benchmark_result, error, entropy(ot_result.X.flatten())
    return benchmark_result


def _benchmark_moscot(
    benchmark_f: Callable,
    validate_ot: bool,
    adata: AnnData,
    true_coupling: Optional[Union[np.ndarray, csr_matrix]],
    key: str,
    key_value_1: Any,
    key_value_2: Any,
    epsilon: float,
    lambda_1: float,
    lambda_2: float,
    threshold: float,
    max_iterations: int,
    rank: Optional[int] = None,
    online: Optional[int] = None,
    seed: Optional[int] = None,
    jit: Optional[bool] = False,
    gamma: Optional[float] = None,
    n_val_samples: Optional[int] = None,
):
    from jax.config import config

    config.update("jax_enable_x64", True)

    from moscot.backends.ott import SinkhornSolver
    from moscot.problems.time._lineage import TemporalProblem
    from time_utils import distance_between_pushed_masses

    if rank is None:
        solver = SinkhornSolver(jit=jit, threshold=threshold, max_iterations=max_iterations)
    else:
        solver = SinkhornSolver(
            jit=jit, threshold=threshold, max_iterations=max_iterations, rank=rank, gamma=gamma
        )  # , seed=seed)

    tp = TemporalProblem(adata, solver=solver)
    tp.prepare(key, subset=[(key_value_1, key_value_2)], policy="sequential", joint_attr="X_pca")

    benchmarked_f = benchmark_f(tp.solve)
    benchmark_result, ot_result = benchmarked_f(
        epsilon=epsilon,
        scale_cost="mean",
        tau_a=lambda_1 / (lambda_1 + epsilon),
        tau_b=lambda_2 / (lambda_2 + epsilon),
        online=online,
    )

    if validate_ot:
        gex_data_source = adata[adata.obs[key] == key_value_1].obsm[
            "X_pca"
        ]  # TODO: do we want to measure this in GEX or PCA space
        gex_data_target = adata[adata.obs[key] == key_value_2].obsm[
            "X_pca"
        ]  # TODO: do we want to measure this in GEX or PCA space
        error = distance_between_pushed_masses(
            gex_data_source,
            gex_data_target,
            ot_result[key_value_1, key_value_2],
            true_coupling,
            seed=seed,
            n_samples=n_val_samples,
        )
        return benchmark_result, error, entropy(ot_result[key_value_1, key_value_2].solution.transport_matrix.flatten())
    return benchmark_result


@ex.post_run_hook
def collect_stats(_run):
    seml.collect_exp_stats(_run)


@ex.config
def config():
    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(seml.create_mongodb_observer(db_collection, overwrite=overwrite))


@ex.automain
def benchmark(
    run: int,
    benchmark_mode: Literal["time", "cpu_memory"],
    validate_ot: bool,
    model: Literal["moscot", "WOT"],
    dirs: Tuple[str, str],
    key: str,
    epsilon: float,
    lambda_1: float,
    lambda_2: float,
    threshold: float,
    max_iterations: int,
    local_pca: int,
    rank: Optional[int] = None,
    online: Optional[int] = None,
    seed: Optional[int] = None,
    jit: Optional[bool] = False,
    gamma: Optional[float] = None,
    n_val_samples: Optional[int] = None,
):
    from sklearn.metrics import pairwise_distances

    import scanpy as sc

    if benchmark_mode == "time":
        benchmark_f = benchmark_time
    elif benchmark_mode == "cpu_memory":
        benchmark_f = benchmark_memory
    else:
        raise NotImplementedError

    anndata_dir, true_coupling_dir = dirs
    adata = sc.read_h5ad(anndata_dir)
    true_coupling = sc.read_h5ad(true_coupling_dir).X
    key_value_2 = adata.obs[key].max()
    key_value_1 = key_value_2 - 1
    adata = adata[adata.obs[key].isin((key_value_1, key_value_2))].copy()

    adata_1 = adata[adata.obs[key] == key_value_1].copy()
    adata_2 = adata[adata.obs[key] == key_value_2].copy()
    sc.tl.pca(adata_1, n_comps=local_pca)
    sc.tl.pca(adata_2, n_comps=local_pca)
    bdata = adata_1.concatenate(adata_2)

    if model == "WOT":
        C = pairwise_distances(adata_1.obsm["X_pca"], adata_2.obsm["X_pca"], metric="sqeuclidean")
        C /= C.mean()
        del adata_1
        del adata_2
        del adata

        return _benchmark_wot(
            C=C,
            benchmark_f=benchmark_f,
            validate_ot=validate_ot,
            adata=bdata,
            true_coupling=true_coupling,
            key=key,
            key_value_1=key_value_1,
            key_value_2=key_value_2,
            epsilon=epsilon,
            lambda_1=lambda_1,
            lambda_2=lambda_2,
            threshold=threshold,
            max_iterations=max_iterations,
            n_val_samples=n_val_samples,
        )
    elif model == "moscot":
        del adata_1
        del adata_2
        del adata

        return _benchmark_moscot(
            benchmark_f=benchmark_f,
            validate_ot=validate_ot,
            adata=bdata,
            true_coupling=true_coupling,
            key=key,
            key_value_1=key_value_1,
            key_value_2=key_value_2,
            epsilon=epsilon,
            lambda_1=lambda_1,
            lambda_2=lambda_2,
            threshold=threshold,
            max_iterations=max_iterations,
            rank=rank,
            online=online,
            seed=seed,
            jit=jit,
            gamma=gamma,
            n_val_samples=n_val_samples,
        )
    else:
        raise NotImplementedError
