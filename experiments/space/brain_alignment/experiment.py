from typing import Dict, Tuple, Sequence
from pathlib import Path

from sacred import Experiment
import seml

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
    def _moscot(
        adata: AnnData,
        params: Dict,
        slices: Sequence[str],
        path_results: str,
        unique_id: str,
    ) -> None:
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
                threshold=1e-5,
                epsilon=0,
                alpha=alpha,
                linear_solver_kwargs={
                    "epsilon": 0,
                    "threshold": 1e-5,
                },
            )
        )
        end = time.time()
        ap.save(path_results, file_prefix=f"{unique_id}_{alpha}")

        ap.align(reference=slices[0], mode="warp")
        ap.align(reference=slices[0], mode="affine")

        adata1 = adata[adata.obs.batch.isin([slices[0], slices[1]])].copy()
        adata2 = adata[adata.obs.batch.isin([slices[0], slices[2]])].copy()

        adata1.write(Path(path_results) / f"{unique_id}_{alpha}_1.h5ad")
        adata2.write(Path(path_results) / f"{unique_id}_{alpha}_2.h5ad")

        results["time"] = end - start
        return results

    def _read_process_anndata(path_data: str, dataset: int) -> Tuple[AnnData, Sequence[str]]:
        import scanpy as sc

        adata = ad.read(path_data)
        all_slices = {"slice1": ["1_1", "1_2", "1_3"], "slice2": ["2_1", "2_2", "2_3"]}
        slices = all_slices[dataset]
        adata = adata[adata.obs.batch.isin(slices)].copy()

        sc.pp.subsample(adata, fraction=0.01)
        return adata, slices

    # read data and processing
    adata, slices = _read_process_anndata(path_data, dataset)
    unique_id = str(seml.utils.make_hash(ex.current_run.config))

    return _moscot(
        adata=adata,
        params=params,
        slices=slices,
        path_results=Path(path_results),
        unique_id=unique_id,
    )
