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
def benchmark(path_data: str, dataset: int, split: int, method: str, params: Dict, path_results: str):

    # read data and processing
    adata_sc, adata_sp_train, true_df = _read_process_anndata(path_data, dataset, split)
    unique_id = seml.utils.make_hash(ex.current_run.config)

    if method == "MOSCOT":
        return _moscot(
            adata_sc=adata_sc,
            adata_sp_train=adata_sp_train,
            true_df=true_df,
            method=method,
            params=params,
            path_results=path_results,
            unique_id=unique_id,
        )


def _gimvi(
    adata_sc: AnnData,
    adata_sp_train: AnnData,
    true_df: pd.DataFrame,
    method: str,
    params: Optional[Dict] = None,
):
    from scvi.external import GIMVI

    epochs = params["epochs"]

    GIMVI.setup_anndata(adata_sp_train, layer="counts")
    GIMVI.setup_anndata(adata_sc, layer="counts")
    model = GIMVI(adata_sc, adata_sp_train)

    start = time.perf_counter()
    model.train(epochs)
    end = time.perf_counter()

    results = {
        "time": end - start,
        "method": method,
        # "method_info": {"kl_reg": ad_map.uns["training_history"]["kl_reg"][-1]},
        "adata_sc_size": adata_sc.shape[0],
        "adata_sp_size": adata_sp_train.shape[0],
        # "corr_results": corr_results,
    }

    return results


def _tangram(
    adata_sc: AnnData,
    adata_sp_train: AnnData,
    true_df: pd.DataFrame,
    method: str,
    params: Optional[Dict] = None,
):
    import torch
    import tangram as tg

    device = torch.device("cuda:0")
    tg.pp_adatas(adata_sc, adata_sp_train, genes=adata_sp_train.var_names.tolist())

    start = time.perf_counter()
    ad_map = tg.map_cells_to_space(adata_sc, adata_sp_train, device=device)
    end = time.perf_counter()

    ad_ge = tg.project_genes(ad_map, adata_sc)
    true_df.columns = [a.lower() for a in true_df.columns]
    pred_df = sc.get.obs_df(ad_ge, keys=true_df.columns.tolist())

    corr_results = _corr_results(true_df, pred_df)

    results = {
        "time": end - start,
        "method": method,
        "method_info": {"kl_reg": ad_map.uns["training_history"]["kl_reg"][-1]},
        "adata_sc_size": adata_sc.shape[0],
        "adata_sp_size": adata_sp_train.shape[0],
        "corr_results": corr_results,
    }

    return results


def _moscot(
    adata_sc: AnnData,
    adata_sp_train: AnnData,
    true_df: pd.DataFrame,
    method: str,
    params: Optional[Dict] = None,
) -> None:
    from jax.config import config

    config.update("jax_enable_x64", True)
    from moscot.problems.space import MappingProblem

    epsilon, alpha, obsm_key = params["epsilon"], params["alpha"], params["obsm_key"]
    prob = MappingProblem(adata_sc=adata_sc, adata_sp=adata_sp_train)
    prob = prob.prepare(
        sc_attr={"attr": "obsm", "key": "X_pca"},
        sp_attr={"attr": "obsm", "key": obsm_key},
        joint_attr=None,
        callback="local-pca",
    )

    start = time.perf_counter()
    prob = prob.solve(epsilon=epsilon, alpha=alpha)
    end = time.perf_counter()

    converged = prob.solutions[list(prob.solutions.keys())[0]].converged
    cost = prob.solutions[list(prob.solutions.keys())[0]].cost

    adata_pred = prob.impute(var_names=true_df.columns.values, device="cpu")
    pred_df = sc.get.obs_df(adata_pred, keys=true_df.columns.tolist())

    corr_results = _corr_results(true_df, pred_df)

    results = {
        "time": end - start,
        "method": method,
        "method_info": {"converged": converged, "cost": cost},
        "adata_sc_size": adata_sc.shape[0],
        "adata_sp_size": adata_sp_train.shape[0],
        "corr_results": corr_results,
    }

    return results


def _read_process_anndata(path_data: str, dataset: int, split: int) -> Tuple[AnnData, AnnData, pd.DataFrame]:
    adata_sc = ad.read(Path(path_data) / f"dataset{dataset}_sc.h5ad")
    adata_sp = ad.read(Path(path_data) / f"dataset{dataset}_sp.h5ad")
    adata_insitu = ad.read(Path(path_data) / f"dataset{dataset}_insitu.h5ad")

    train_split = f"train_{split}"
    test_split = f"test_{split}"

    test_var = list(set(adata_sc.var_names).intersection(set(adata_sp.var_names[adata_sp.var[test_split]])))

    adata_sp_train = adata_sp[:, adata_sp.var[train_split]].copy()
    true_df = sc.get.obs_df(adata_insitu, keys=test_var)
    sc.tl.pca(adata_sp_train)
    adata_sp_train.obsm["X_pca_spatial"] = np.hstack(
        [adata_sp_train.obsm["X_pca"].copy(), adata_sp_train.obsm["spatial"].copy()]
    )

    assert not np.any(np.in1d(adata_sp_train.var_names.values, true_df.columns.values)), "overlap train test set"
    return adata_sc, adata_sp_train, true_df


def _corr_results(true_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
    corr_pearson = pred_df.corrwith(true_df, method="pearson")
    corr_spearman = pred_df.corrwith(true_df, method="spearman")
    return pd.concat([corr_pearson, corr_spearman], axis=1)
