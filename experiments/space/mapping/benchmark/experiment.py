from typing import List, Literal, Callable, Optional
from pathlib import Path

from sacred import Experiment
import seml

import numpy as np

from anndata import AnnData
import scanpy as sc

from moscot_benchmarks.utils import benchmark_time, benchmark_memory

ex = Experiment()
seml.setup_logger(ex)


def _benchmark_moscot(
    adata_sc: AnnData,
    adata_spatial: AnnData,
    var_names: List[str],
    benchmark_f: Callable,
    benchmark_mode: str,
    epsilon: float,
    alpha: float,
    rank: int,
    max_iterations: int,
    jit: Optional[bool] = False,
    gamma: Optional[float] = None,
):
    from jax.config import config

    config.update("jax_enable_x64", True)
    config.update("jax_platform_name", "cpu")
    from moscot.problems.space import MappingProblem

    mp = MappingProblem(adata_sc, adata_spatial).prepare(
        sc_attr={"attr": "obsm", "key": "X_pca_protein"},
        spatial_key={"attr": "obsm", "key": "X_pca"},
        joint_attr=None,
        var_names=var_names,
    )

    benchmarked_f = benchmark_f(mp.solve)
    benchmark_result, _ = benchmarked_f(
        alpha=alpha,
        epsilon=epsilon,
        rank=rank,
        max_iterations=max_iterations,
        jit=jit,
        gamma=gamma,
    )

    results = {
        "benchmark_result": benchmark_result,
        "benchmark_mode": benchmark_mode,
        "adata_sc_size": adata_sc.shape[0],
        "adata_spatial_size": adata_spatial.shape[0],
    }

    return results


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
    benchmark_mode: Literal["time", "cpu_memory"],
    path_data: str,
    adata_sc_file: str,
    adata_spatial_file: str,
    adata_cite_file: str,
    fraction: float,
    epsilon: float,
    alpha: float,
    max_iterations: int,
    rank: Optional[int] = None,
    seed: Optional[int] = None,
    jit: Optional[bool] = False,
    gamma: Optional[float] = None,
):
    if benchmark_mode == "time":
        benchmark_f = benchmark_time
    elif benchmark_mode == "cpu_memory":
        benchmark_f = benchmark_memory
    else:
        raise NotImplementedError

    # read data and processing
    path_data = Path(path_data)
    adata_sc = sc.read(path_data / adata_sc_file)
    adata_p = sc.read(path_data / adata_cite_file)
    sc.pp.pca(adata_p, n_comps=30)
    adata_sc.obsm["X_pca_protein"] = adata_p.obsm["X_pca"].copy()
    adata_spatial = sc.read(path_data / adata_spatial_file)

    rng = np.random.default_rng(seed)

    n_sc = round(adata_sc.shape[0] * fraction)
    n_spatial = round(adata_spatial.shape[0] * fraction)

    idx_sc = rng.choice(adata_sc.obs_names, size=n_sc)
    idx_spatial = rng.choice(adata_spatial.obs_names, size=n_spatial)

    adata_sc = adata_sc[idx_sc].copy()
    adata_spatial = adata_spatial[idx_spatial].copy()

    adata_spatial.var_names = [v.capitalize() for v in adata_spatial.var_names.values]
    var_names = list(set(adata_sc.var_names.values).intersection(adata_spatial.var_names.values))
    sc.pp.pca(adata_spatial, n_comps=30)

    return _benchmark_moscot(
        adata_sc=adata_sc,
        adata_spatial=adata_spatial,
        var_names=var_names,
        benchmark_f=benchmark_f,
        benchmark_mode=benchmark_mode,
        epsilon=epsilon,
        alpha=alpha,
        rank=rank,
        max_iterations=max_iterations,
        jit=jit,
        gamma=gamma,
    )
