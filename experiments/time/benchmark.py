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
    key_value_1: Any,
    key_value_2: Any,
    epsilon: float,
    lambda_1: float,
    lambda_2: float,
    n_val_samples: Optional[int] = None,
):
    import wot

    from jax.config import config

    config.update("jax_enable_x64", True)  # need this for "distance_between_pushed_masses"
    from experiments.time.time_utils import distance_between_pushed_masses

    ot_model = wot.ot.OTModel(
        adata,
        day_field="day",
        epsilon=epsilon,
        lambda1=lambda_1,
        lambda2=lambda_2,
        local_pca=0,
    )

    benchmarked_f = benchmark_f(ot_model.compute_transport_map)
    benchmark_result, ot_result = benchmarked_f(key_value_1, key_value_2, cost_matrix=C)

    if validate_ot:
        gex_data_source = adata[adata.obs["day"] == key_value_1].obsm["X_pca"]
        gex_data_target = adata[adata.obs["day"] == key_value_2].obsm["X_pca"]
        error = distance_between_pushed_masses(
            gex_data_source, gex_data_target, ot_result.X, true_coupling, n_samples=n_val_samples
        )
        return {"benchmark_result": benchmark_result, "error": error, "entropy": entropy(ot_result.X.flatten()), "converged": 2}
    return {"benchmark_result": benchmark_result}


def _benchmark_moscot(
    benchmark_f: Callable,
    validate_ot: bool,
    adata: AnnData,
    true_coupling: Optional[Union[np.ndarray, csr_matrix]],
    key_value_1: Any,
    key_value_2: Any,
    epsilon: float,
    lambda_1: float,
    lambda_2: float,
    rank: Optional[int] = None,
    batch_size: Optional[int] = None,
    seed: Optional[int] = None,
    n_val_samples: Optional[int] = None,
):
    from jax.config import config

    config.update("jax_enable_x64", True)

    from moscot.problems.time import TemporalProblem

    from experiments.time.time_utils import distance_between_pushed_masses

    if rank == -1:
        kwargs = {} #{"jit":jit, "threshold":threshold, "max_iterations":max_iterations}
    else:
        kwargs = {"rank":rank}
        #kwargs = {
        #    "jit":jit, "threshold":1e-3, "max_iterations":max_iterations, "rank":rank, "gamma":gamma,
        #} # seed is set internally in ott-jax

    tp = TemporalProblem(adata)
    tp.prepare("day", joint_attr="X_pca")

    benchmarked_f = benchmark_f(tp.solve)
    benchmark_result, ot_result = benchmarked_f(
        epsilon=epsilon,
        scale_cost="mean",
        tau_a = 1 if rank > -1 else lambda_1 / (lambda_1 + epsilon),
        tau_b = 1 if rank > -1 else lambda_2 / (lambda_2 + epsilon),
        batch_size=batch_size, #TODO: check with Mike
        **kwargs
    )

    if validate_ot:
        gex_data_source = adata[adata.obs["day"] == key_value_1].obsm["X_pca"]
        gex_data_target = adata[adata.obs["day"] == key_value_2].obsm["X_pca"]
        error = distance_between_pushed_masses(
            gex_data_source,
            gex_data_target,
            ot_result[key_value_1, key_value_2],
            true_coupling,
            seed=seed,
            n_samples=n_val_samples,
        )
        return {
            "benchmark_result": benchmark_result,
            "error": np.array(error),
            "entropy": np.array(entropy(ot_result[key_value_1, key_value_2].solution.transport_matrix.flatten())),
            "converged": ot_result[key_value_1, key_value_2].solution.converged
        }
    return {"benchmark_result": benchmark_result}


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
    fpath: Union[str, Tuple[str, str]],
    epsilon: float,
    lambda_1: float,
    lambda_2: float,
    max_iterations: int,
    rank: Optional[int] = None,
    batch_size: Optional[int] = None,
    seed: Optional[int] = None,
    jit: Optional[bool] = False,
    gamma: Optional[float] = None,
    n_val_samples: Optional[int] = None,
    initializer: Optional[str] = None,
):
    from sklearn.metrics import pairwise_distances
    import anndata
    from scipy.sparse import dok_matrix, csr_matrix
    from time_utils import prepare_data

    import scanpy as sc

    if benchmark_mode == "time":
        benchmark_f = benchmark_time
    elif benchmark_mode == "cpu_memory":
        benchmark_f = benchmark_memory
    else:
        raise NotImplementedError

    if run == -1:
        adata = sc.read_h5ad(fpath)
        true_coupling, rna_arrays, _ = prepare_data(adata, np.log2(len(adata)+2)-3)
        del adata
        adata_early = anndata.AnnData(csr_matrix((rna_arrays["early"].shape[0], 1)))
        adata_early.obs_names = ["a"+str(n) for n in adata_early.obs_names]
        adata_late = anndata.AnnData(csr_matrix((rna_arrays["late"].shape[0], 1)))  
        adata_early.obs["day"] = 0
        adata_early.obsm["X_pca"] = rna_arrays["early"]
        adata_late.obs["day"] = 1
        adata_late.obsm["X_pca"] = rna_arrays["late"]
        
        adata_concat = anndata.concat([adata_early, adata_late])
        del adata_early
        del adata_late

        adata.save("new_path")
        return None

    adata= sc.read(fpath)

    if model == "WOT":
        C = pairwise_distances(adata[adata.obs["time"]==0].obsm["X_pca"], adata[adata.obs["time"]==1].obsm["X_pca"], metric="sqeuclidean")
        C /= C.mean()
        del rna_arrays

        return _benchmark_wot(
            C=C,
            benchmark_f=benchmark_f,
            validate_ot=validate_ot,
            adata=adata_concat,
            true_coupling=true_coupling,
            key_value_1=0,
            key_value_2=1,
            epsilon=epsilon,
            lambda_1=lambda_1,
            lambda_2=lambda_2,
            max_iterations=max_iterations,
            n_val_samples=n_val_samples,
        )
    elif model == "moscot":
        del rna_arrays
        return _benchmark_moscot(
            benchmark_f=benchmark_f,
            validate_ot=validate_ot,
            adata=adata_concat,
            true_coupling=true_coupling,
            key_value_1=0,
            key_value_2=1,
            epsilon=epsilon,
            lambda_1=lambda_1,
            lambda_2=lambda_2,
            max_iterations=max_iterations,
            rank=rank,
            batch_size=batch_size,
            seed=seed,
            jit=jit,
            gamma=gamma,
            n_val_samples=n_val_samples,
            initializer=initializer,
        )
    else:
        raise NotImplementedError
