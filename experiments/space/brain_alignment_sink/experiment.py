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
def align_large(path_data: str, adatas: str, params: Dict, path_results: str):
    def _moscot(
        adata1: AnnData,
        adata2: AnnData,
        params: Dict,
        path_results: str,
        unique_id: str,
    ) -> None:
        from jax.config import config

        import scanpy as sc

        config.update("jax_enable_x64", True)
        import time

        from moscot.problems.generic import SinkhornProblem

        import numpy as np

        epsilon = params["epsilon"]
        genes = params["genes"]
        results = {}

        if genes:
            sc.pp.pca(adata1)
            sc.pp.pca(adata2)
            adata1.obsm["pca_spatial"] = np.hstack([adata1.obsm["X_pca"], adata1.obsm["spatial_norm_affine"]])
            adata2.obsm["pca_spatial"] = np.hstack([adata2.obsm["X_pca"], adata2.obsm["spatial_norm_affine"]])
            linear_cost = "pca_spatial"
        else:
            linear_cost = "spatial_norm_affine"

        start = time.time()
        sp1 = (
            SinkhornProblem(adata=adata1)
            .prepare(
                key="batch_final",
                # max_iterations=100,
                joint_attr={"attr": "obsm", "key": linear_cost},
            )
            .solve(epsilon=epsilon, threshold=1e-5, batch_size=10_000)
        )
        sp2 = (
            SinkhornProblem(adata=adata2)
            .prepare(
                key="batch_final",
                # max_iterations=2,
                joint_attr={"attr": "obsm", "key": linear_cost},
            )
            .solve(epsilon=epsilon, threshold=1e-5, batch_size=10_000)
        )
        end = time.time()
        sp1.save(path_results, file_prefix=f"{unique_id}_1_{epsilon}")
        sp2.save(path_results, file_prefix=f"{unique_id}_2_{epsilon}")

        adata1_final = adata1[adata1.obs.batch_final.isin(["0"])].copy()
        adata2_final = adata1[adata1.obs.batch_final.isin(["1"])].copy()

        adata1_copy = adata2[adata2.obs.batch_final.isin(["0"])].copy()
        adata3_final = adata2[adata2.obs.batch_final.isin(["1"])].copy()

        spatial2 = sp1[("0", "1")].solution.to("cpu").push(adata1_final.obsm["spatial_norm"], scale_by_marginals=True)
        spatial3 = sp2[("0", "1")].solution.to("cpu").push(adata1_copy.obsm["spatial_norm"], scale_by_marginals=True)

        # get min max and normalize
        spatial1 = adata1_final.obsm["spatial_norm"].copy()
        spatial1.min()
        spatial1.max()

        adata1_final.obsm["spatial_norm_final"] = (spatial1 - spatial1.min()) / (spatial1.max() - spatial1.min())
        adata2_final.obsm["spatial_norm_final"] = (spatial2 - spatial2.min()) / (spatial2.max() - spatial2.min())
        adata3_final.obsm["spatial_norm_final"] = (spatial3 - spatial3.min()) / (spatial3.max() - spatial3.min())

        adata_final = ad.concat([adata1_final, adata2_final, adata3_final], label="batch_final", keys=["0", "1", "2"])
        adata_final.write(path_results / f"{unique_id}_sink.h5ad")
        results["time"] = end - start
        return results

    def _read_process_anndata(path_data: str, adatas: str) -> Tuple[AnnData, Sequence[str]]:
        pass

        adata1 = ad.read(path_data / (adatas + ".h5ad"))
        adata2 = ad.read(path_data / (adatas[:-1] + "2.h5ad"))

        adata1.obs["batch_final"] = adata1.obs["batch"].replace(
            {adata1.obs["batch"].unique()[0]: "0", adata1.obs["batch"].unique()[1]: "1"}
        )
        adata2.obs["batch_final"] = adata2.obs["batch"].replace(
            {adata2.obs["batch"].unique()[0]: "0", adata2.obs["batch"].unique()[1]: "1"}
        )
        # sc.pp.subsample(adata1, fraction=0.01)
        # sc.pp.subsample(adata2, fraction=0.01)
        return adata1, adata2

    # read data and processing
    adata1, adata2 = _read_process_anndata(Path(path_data), adatas)
    unique_id = str(seml.utils.make_hash(ex.current_run.config))

    return _moscot(
        adata1=adata1,
        adata2=adata2,
        params=params,
        path_results=Path(path_results),
        unique_id=unique_id,
    )
