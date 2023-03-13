from typing import Dict, Tuple, Optional
from pathlib import Path
import time

from sacred import Experiment
import seml
import pandas as pd

import numpy as np

from anndata import AnnData
import scanpy as sc
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
def benchmark(path_data: str, dataset: int, seed: int, method: str, params: Dict, path_results: str):
    def _gimvi(
        adata_sc: AnnData,
        adata_sp_train: AnnData,
        true_df: pd.DataFrame,
        method: str,
        params: Optional[Dict] = None,
        path_results: str = None,
        unique_id: str = None,
    ):
        from scvi.external import GIMVI

        epochs, n_latent = params["epochs"], params["n_latent"]  # 200

        GIMVI.setup_anndata(adata_sp_train, layer="counts")
        GIMVI.setup_anndata(adata_sc, layer="counts")
        model = GIMVI(adata_sc, adata_sp_train, n_latent=n_latent)

        start = time.perf_counter()
        model.train(epochs)
        end = time.perf_counter()

        _, gene_pred = model.get_imputed_values(normalized=True)

        adata_pred = AnnData(gene_pred, dtype=np.float_)
        adata_pred.var.index = adata_sc.var_names

        pred_df = sc.get.obs_df(adata_pred, keys=true_df.columns.tolist())

        corr_results = _corr_results(true_df, pred_df)
        corr_results.to_csv(Path(path_results) / f"{unique_id}_corr_results.csv")

        results = {
            "time": end - start,
            "method": method,
            "method_info": None,
            "adata_sc_size": adata_sc.shape[0],
            "adata_sp_size": adata_sp_train.shape[0],
            "gene_size": len(true_df.columns),
            "corr_results_mean": corr_results.mean(0),
            "corr_results_median": corr_results.median(0),
            "corr_results_var": corr_results.var(0),
            "corr_results_max": corr_results.max(0),
            "corr_results_min": corr_results.min(0),
        }

        return results

    def _tangram(
        adata_sc: AnnData,
        adata_sp_train: AnnData,
        true_df: pd.DataFrame,
        method: str,
        params: Optional[Dict] = None,
        path_results: str = None,
        unique_id: str = None,
    ):
        import torch
        import tangram as tg

        learning_rate, num_epochs = params["learning_rate"], params["num_epochs"]
        tg.pp_adatas(adata_sc, adata_sp_train, genes=true_df.columns.tolist(), gene_to_lowercase=False)
        device = torch.device("cuda")

        start = time.perf_counter()
        ad_map = tg.map_cells_to_space(
            adata_sc,
            adata_sp_train,
            device=device,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
        )
        end = time.perf_counter()

        ad_ge = tg.project_genes(ad_map, adata_sc)
        # true_df.columns = [a.lower() for a in true_df.columns]
        pred_df = sc.get.obs_df(ad_ge, keys=true_df.columns.tolist())

        corr_results = _corr_results(true_df, pred_df)
        corr_results.to_csv(Path(path_results) / f"{unique_id}_corr_results.csv")

        results = {
            "time": end - start,
            "method": method,
            "method_info": {"kl_reg": ad_map.uns["training_history"]["kl_reg"][-1]},
            "adata_sc_size": adata_sc.shape[0],
            "adata_sp_size": adata_sp_train.shape[0],
            "gene_size": len(true_df.columns),
            "corr_results_mean": corr_results.mean(0),
            "corr_results_median": corr_results.median(0),
            "corr_results_var": corr_results.var(0),
            "corr_results_max": corr_results.max(0),
            "corr_results_min": corr_results.min(0),
        }

        return results

    def _moscot(
        adata_sc: AnnData,
        adata_sp_train: AnnData,
        true_df: pd.DataFrame,
        method: str,
        params: Optional[Dict] = None,
        path_results: str = None,
        unique_id: str = None,
    ) -> None:
        from jax.config import config

        config.update("jax_enable_x64", True)
        from moscot.problems.space import MappingProblem

        epsilon, alpha = params["epsilon"], params["alpha"]
        adata_sp_train.obsm["spatial"] = (
            adata_sp_train.obsm["spatial"] - adata_sp_train.obsm["spatial"].mean()
        ) / adata_sp_train.obsm["spatial"].std()
        prob = MappingProblem(adata_sc=adata_sc, adata_sp=adata_sp_train)
        prob = prob.prepare(
            sc_attr={"attr": "obsm", "key": "X_pca"},
            var_names=adata_sp_train.var_names.values,
            normalize=True,
        )

        start = time.perf_counter()
        prob = prob.solve(epsilon=epsilon, alpha=alpha, max_iterations=100, threshold=1e-5)
        end = time.perf_counter()

        converged = prob.solutions[list(prob.solutions.keys())[0]].converged
        cost = prob.solutions[list(prob.solutions.keys())[0]].cost

        adata_pred = prob.impute(var_names=true_df.columns.tolist(), device="cpu")
        pred_df = sc.get.obs_df(adata_pred, keys=true_df.columns.tolist())

        corr_results = _corr_results(true_df, pred_df)

        corr_results.to_csv(Path(path_results) / f"{unique_id}_corr_results.csv")

        results = {
            "time": end - start,
            "method": method,
            "method_info": {"converged": converged, "cost": cost},
            "adata_sc_size": adata_sc.shape[0],
            "adata_sp_size": adata_sp_train.shape[0],
            "gene_size": len(true_df.columns),
            "corr_results_mean": corr_results.mean(0),
            "corr_results_median": corr_results.median(0),
            "corr_results_var": corr_results.var(0),
            "corr_results_max": corr_results.max(0),
            "corr_results_min": corr_results.min(0),
        }

        return results

    def _read_process_anndata(path_data: str, dataset: int, seed: int) -> Tuple[AnnData, AnnData, pd.DataFrame]:
        adata_sp = ad.read(Path(path_data) / f"dataset{dataset}_sp.h5ad")

        rng = np.random.default_rng(seed)
        if "highly_variable" in adata_sp.var.columns:
            adata_sp = adata_sp[:, adata_sp.var.highly_variable].copy()
            n_genes = 100
        else:
            n_genes = 10

        adata_sp_a = sc.pp.subsample(adata_sp, fraction=0.5, copy=True, random_state=seed)
        adata_sp_b = adata_sp[~np.in1d(adata_sp.obs_names, adata_sp_a.obs_names)].copy()

        test_var = rng.choice(adata_sp.var_names, n_genes, replace=False).tolist()
        train_var = adata_sp_a.var_names[~np.in1d(adata_sp_a.var_names, test_var)].tolist()
        true_df = sc.get.obs_df(adata_sp_b, keys=test_var)

        adata_sp_a_train = adata_sp_a[:, train_var].copy()
        adata_sp_b_train = adata_sp_b[:, train_var].copy()

        sc.tl.pca(adata_sp_b_train)
        sc.tl.pca(adata_sp_a_train)
        adata_sp_a.obsm["X_pca"] = adata_sp_a_train.obsm["X_pca"].copy()
        return adata_sp_a, adata_sp_b_train, true_df

    def _corr_results(true_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
        corr_pearson = pred_df.corrwith(true_df, method="pearson")
        corr_spearman = pred_df.corrwith(true_df, method="spearman")
        out = pd.concat([corr_pearson, corr_spearman], axis=1)
        out.columns = ["pearson", "spearman"]
        return out

    # read data and processing
    adata_sp_a, adata_sp_b, true_df = _read_process_anndata(path_data, dataset, seed)
    unique_id = seml.utils.make_hash(ex.current_run.config)

    if method == "MOSCOT":
        return _moscot(
            adata_sc=adata_sp_a,
            adata_sp_train=adata_sp_b,
            true_df=true_df,
            method=method,
            params=params,
            path_results=path_results,
            unique_id=unique_id,
        )
    elif method == "GIMVI":
        return _gimvi(
            adata_sc=adata_sp_a,
            adata_sp_train=adata_sp_b,
            true_df=true_df,
            method=method,
            params=params,
            path_results=path_results,
            unique_id=unique_id,
        )
    elif method == "TANGRAM":
        return _tangram(
            adata_sc=adata_sp_a,
            adata_sp_train=adata_sp_b,
            true_df=true_df,
            method=method,
            params=params,
            path_results=path_results,
            unique_id=unique_id,
        )
    else:
        raise ValueError("Method not implemented")
