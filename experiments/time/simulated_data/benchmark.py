import numpy as np
from anndata import AnnData
from typing import Callable, Optional, Any, Union, Literal
from functools import wraps, partial
from memory_profiler import memory_usage
from time import perf_counter
from sacred import Experiment
import seml
import os
import logging
import pandas as pd
from moscot_benchmarks.utils import benchmark_memory, benchmark_time
from scipy.sparse import csr_matrix

ex = Experiment()
seml.setup_logger(ex)


def _benchmark_wot(
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
    local_pca: int,
    seed: Optional[int] = None,
):
    import scanpy as sc
    import wot
    from anndata import AnnData
    from jax.config import config

    config.update("jax_enable_x64", True)  # need this for "distance_between_pushed_masses"
    from moscot_benchmarks.time._utils import distance_between_pushed_masses

    # TODO: check it uses PCA space
    adata = adata[adata.obs[key].isin((key_value_1, key_value_2))].copy()

    # TODO: check whether we can do PCA beforehand
    ot_model = wot.ot.OTModel(
        adata,
        day_field=key,
        epsilon=epsilon,
        lambda1=lambda_1,
        lambda2=lambda_2,
        threshold=threshold,
        max_iter=max_iterations,
        local_pca=local_pca,
    )

    benchmarked_f = benchmark_f(ot_model.compute_transport_map)

    benchmark_result, ot_result = benchmarked_f(key_value_1, key_value_2)

    if validate_ot:
        gex_data_source = adata[adata.obs[key] == key_value_1].X #TODO: do we want to measure this in GEX or PCA space
        gex_data_target = adata[adata.obs[key] == key_value_2].X #TODO: do we want to measure this in GEX or PCA space
        error = distance_between_pushed_masses(
            gex_data_source, gex_data_target, ot_result.transport_matrix, true_coupling, seed=seed
        )  # TODO check this is correct attribute
        return benchmark_result, error
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
    local_pca: int,
    rank: Optional[int] = None,
    online: bool = False,
    seed: Optional[int] = None,
    jit: Optional[bool] = False,
):
    from jax.config import config

    config.update("jax_enable_x64", True)
    from moscot.backends.ott import SinkhornSolver
    from moscot.problems.time._lineage import TemporalProblem
    from moscot_benchmarks.time._utils import distance_between_pushed_masses

    adata = adata[adata.obs[key].isin((key_value_1, key_value_2))].copy()

    if rank is None:
        solver = SinkhornSolver(jit=jit, threshold=threshold, max_iterations=max_iterations)
    else:
        solver = SinkhornSolver(jit=jit, threshold=threshold, max_iterations=max_iterations, rank=rank)

    tp = TemporalProblem(adata, solver=solver)
    tp.prepare(key, subset=[(key_value_1, key_value_2)], policy="sequential", callback_kwargs={"n_comps": local_pca})

    benchmarked_f = benchmark_f(tp.solve)
    benchmark_result, ot_result = benchmarked_f(
        epsilon=epsilon, tau_a=lambda_1 / (lambda_1 + epsilon), tau_b=lambda_2 / (lambda_2 + epsilon), online=online
    )

    if validate_ot:
        gex_data_source = adata[adata.obs[key] == key_value_1].X #TODO: do we want to measure this in GEX or PCA space
        gex_data_target = adata[adata.obs[key] == key_value_2].X #TODO: do we want to measure this in GEX or PCA space
        error = distance_between_pushed_masses(
            gex_data_source, gex_data_target, ot_result[key_value_1, key_value_2], true_coupling, seed=seed
        )  # TODO check this is correct attribute
        return benchmark_result, error
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
    anndata_dir: str,
    key_true_coupling: str,
    key: str,
    key_value_1: Any,
    key_value_2: Any,
    epsilon: float,
    lambda_1: float,
    lambda_2: float,
    threshold: float,
    max_iterations: int,
    local_pca: int,
    rank: Optional[int] = None,
    online: bool = False,
    seed: Optional[int] = None,
    jit: Optional[bool] = False
):
    import scanpy as sc

    if benchmark_mode == "time":
        benchmark_f = benchmark_time
    elif benchmark_mode == "cpu_memory":
        benchmark_f = benchmark_memory
    else:
        raise NotImplementedError

    adata = sc.read_h5ad(anndata_dir)
    if key_true_coupling not in adata.uns:
        raise ValueError(f"{key_true_coupling} not found in `adata.uns`.")
    true_coupling = adata.uns[key_true_coupling]

    if model == "WOT":
        return _benchmark_wot(
            benchmark_f=benchmark_f,
            validate_ot=validate_ot,
            adata=adata,
            true_coupling=true_coupling,
            key=key,
            key_value_1=key_value_1,
            key_value_2=key_value_2,
            epsilon=epsilon,
            lambda_1=lambda_1,
            lambda_2=lambda_2,
            threshold=threshold,
            max_iterations=max_iterations,
            local_pca=local_pca,
        )
    elif model == "moscot":
        return _benchmark_moscot(
            benchmark_f=benchmark_f,
            validate_ot=validate_ot,
            adata=adata,
            true_coupling=true_coupling,
            key=key,
            key_value_1=key_value_1,
            key_value_2=key_value_2,
            epsilon=epsilon,
            lambda_1=lambda_1,
            lambda_2=lambda_2,
            threshold=threshold,
            max_iterations=max_iterations,
            local_pca=local_pca,
            rank=rank,
            online=online,
            seed=seed,
            jit=jit,
            )
    else:
        raise NotImplementedError
