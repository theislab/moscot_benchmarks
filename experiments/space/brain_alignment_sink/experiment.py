from typing import Dict, Tuple
from pathlib import Path

from sacred import Experiment
import seml
import pandas as pd

from anndata import AnnData
import anndata as ad

ex = Experiment()
seml.setup_logger(ex)


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
def align_large(path_data: str, dataset: int, params: Dict, path_results: str):
    def _moscot(adata: AnnData, params: Dict, path_results: str) -> None:
        from jax.config import config

        config.update("jax_enable_x64", True)
        import time

        import moscot as mt

        alpha = params["alpha"]
        results = {}

        start = time.time()
        ap = (
            mt.problems.space.AlignmentProblem(adata)
            .prepare(batch_key="batch")
            .solve(
                rank=500,
                max_iterations=100,
                threshold=1e-7,
                epsilon=0,
                alpha=alpha,
                linear_solver_kwargs={
                    "epsilon": 0,
                    "threshold": 1e-5,
                },
            )
        )
        end = time.time()
        ap.save

        results["time"] = end - start
        return results

    def _read_process_anndata(path_data: str, dataset: int, seed: int) -> Tuple[AnnData, AnnData, pd.DataFrame]:
        adata = ad.read(path_data)
        slices = {"slice1": ["1_1", "1_2", "1_3"], "slice2": ["2_1", "2_2", "2_3"]}
        adata = adata[adata.obs.batch.isin(slices[dataset])].copy()

        return adata

    # read data and processing
    adata = _read_process_anndata(path_data, dataset)
    seml.utils.make_hash(ex.current_run.config)

    return _moscot(
        adata=adata,
        params=params,
        path_results=Path(path_results),
    )
