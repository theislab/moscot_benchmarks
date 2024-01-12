from typing import Dict, Tuple, Optional
from pathlib import Path
import os
import time

from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
import hydra
import pandas as pd

import anndata as ad


def _read_process_anndata(path_data: str, dataset: int, seed: int) -> Tuple[ad.AnnData, ad.AnnData, pd.DataFrame]:
    import pandas as pd

    import numpy as np

    import scanpy as sc

    adata_sp = ad.read(Path(path_data) / f"dataset{dataset}_sp.h5ad")
    adata_sp.var_names = pd.Index([a.lower() for a in adata_sp.var_names])

    rng = np.random.default_rng(seed)
    if "highly_variable" in adata_sp.var.columns:
        adata_sp = adata_sp[:, adata_sp.var.highly_variable].copy()
        n_genes = 100
    else:
        n_genes = 10

    adata_sp_a = sc.pp.subsample(adata_sp, fraction=0.5, copy=True, random_state=seed)
    adata_sp_b = adata_sp[~np.in1d(adata_sp.obs_names, adata_sp_a.obs_names)].copy()

    test_var = rng.choice(adata_sp.var_names, n_genes, replace=False).tolist()
    train_var = adata_sp.var_names[~np.in1d(adata_sp.var_names, test_var)].tolist()
    true_df = sc.get.obs_df(adata_sp_b, keys=test_var)

    adata_sp_a_train = adata_sp_a[:, train_var].copy()
    adata_sp_b_train = adata_sp_b[:, train_var].copy()

    sc.tl.pca(adata_sp_b_train)
    sc.tl.pca(adata_sp_a_train)

    adata_sp_a.obsm["X_pca"] = adata_sp_a_train.obsm["X_pca"].copy()
    assert len(adata_sp_a.var_names) == (len(adata_sp_b_train.var_names) + len(test_var))
    test_df = sc.get.obs_df(adata_sp_a, keys=test_var)
    np.testing.assert_array_equal(test_df.columns, true_df.columns)
    return adata_sp_a, adata_sp_b_train, true_df


def benchmark(cfg):
    import pandas as pd

    import numpy as np

    from anndata import AnnData
    import scanpy as sc

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
        adata_pred.var_names = adata_sc.var_names

        pred_df = sc.get.obs_df(adata_pred, keys=true_df.columns.tolist())
        np.testing.assert_array_equal(pred_df.columns, true_df.columns)

        corr_results = _corr_results(true_df, pred_df)
        corr_results.to_csv(path_results / "corr_results.csv")

        results = {
            "time": end - start,
            "method": method,
            "adata_sc_size": adata_sc.shape[0],
            "adata_sp_size": adata_sp_train.shape[0],
            "gene_size": len(true_df.columns),
        }

        for corr in ["pearson", "spearman"]:
            results[f"corr_results_mean_{corr}"] = corr_results[corr].mean()
            results[f"corr_results_median_{corr}"] = corr_results[corr].median()
            results[f"corr_results_var_{corr}"] = corr_results[corr].var()
            results[f"corr_results_max_{corr}"] = corr_results[corr].max()
            results[f"corr_results_min_{corr}"] = corr_results[corr].min()

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
        tg.pp_adatas(adata_sc, adata_sp_train, genes=adata_sp_train.var_names.tolist(), gene_to_lowercase=True)
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
        pred_df = sc.get.obs_df(ad_ge, keys=true_df.columns.tolist())
        np.testing.assert_array_equal(pred_df.columns, true_df.columns)

        corr_results = _corr_results(true_df, pred_df)
        corr_results.to_csv(path_results / "corr_results.csv")

        results = {
            "time": end - start,
            "method": method,
            "kl_reg": ad_map.uns["training_history"]["kl_reg"][-1],
            "adata_sc_size": adata_sc.shape[0],
            "adata_sp_size": adata_sp_train.shape[0],
            "gene_size": len(true_df.columns),
        }

        for corr in ["pearson", "spearman"]:
            results[f"corr_results_mean_{corr}"] = corr_results[corr].mean()
            results[f"corr_results_median_{corr}"] = corr_results[corr].median()
            results[f"corr_results_var_{corr}"] = corr_results[corr].var()
            results[f"corr_results_max_{corr}"] = corr_results[corr].max()
            results[f"corr_results_min_{corr}"] = corr_results[corr].min()

        return results

    def _moscot(
        adata_sc: AnnData,
        adata_sp_train: AnnData,
        true_df: pd.DataFrame,
        method: str,
        params: Optional[Dict] = None,
        path_results: str = None,
        unique_id: str = None,
    ) -> dict:
        from jax.config import config

        config.update("jax_enable_x64", True)
        from moscot.problems.space import MappingProblem

        epsilon, alpha, rep, cost = params["epsilon"], params["alpha"], params["rep"], params["cost"]

        if rep == "pca":
            joint_attr = {"attr": "obsm", "key": "X_pca"}
        elif rep == "x":
            joint_attr = {"attr": "X"}
        else:
            raise ValueError("Rep not implemented")

        prob = MappingProblem(adata_sc=adata_sc, adata_sp=adata_sp_train)
        prob = prob.prepare(
            sc_attr={"attr": "obsm", "key": "X_pca"},
            joint_attr=joint_attr,
            var_names=adata_sp_train.var_names.values,
            normalize=True,
            cost=cost,
        )

        start = time.perf_counter()
        prob = prob.solve(epsilon=epsilon, alpha=alpha, max_iterations=5000, threshold=1e-5)
        end = time.perf_counter()

        converged = prob.solutions[list(prob.solutions.keys())[0]].converged
        cost = prob.solutions[list(prob.solutions.keys())[0]].cost

        adata_pred = prob.impute(var_names=true_df.columns.tolist(), device="cpu")
        pred_df = sc.get.obs_df(adata_pred, keys=true_df.columns.tolist())
        np.testing.assert_array_equal(pred_df.columns, true_df.columns)
        corr_results = _corr_results(true_df, pred_df)

        corr_results.to_csv(path_results / "corr_results.csv")

        results = {
            "time": end - start,
            "method": method,
            "converged": converged,
            "cost": cost,
            "adata_sc_size": adata_sc.shape[0],
            "adata_sp_size": adata_sp_train.shape[0],
            "gene_size": len(true_df.columns),
        }

        for corr in ["pearson", "spearman"]:
            results[f"corr_results_mean_{corr}"] = corr_results[corr].mean()
            results[f"corr_results_median_{corr}"] = corr_results[corr].median()
            results[f"corr_results_var_{corr}"] = corr_results[corr].var()
            results[f"corr_results_max_{corr}"] = corr_results[corr].max()
            results[f"corr_results_min_{corr}"] = corr_results[corr].min()

        return results

    def _corr_results(true_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
        pred_df = pred_df[true_df.columns].copy()
        corr_pearson = pred_df.corrwith(true_df, method="pearson")
        corr_spearman = pred_df.corrwith(true_df, method="spearman")
        out = pd.concat([corr_pearson, corr_spearman], axis=1)
        out.columns = ["pearson", "spearman"]
        return out

    path_data = cfg.paths.path_data
    dataset = cfg.dataset
    seed = cfg.seed
    method = cfg.method.name

    unique_id = HydraConfig.get().job.override_dirname
    path_results = Path(cfg.paths.path_results) / unique_id
    os.makedirs(path_results, exist_ok=True)
    # read data and processing
    adata_sp_a, adata_sp_b, true_df = _read_process_anndata(path_data, dataset, seed)

    if method == "MOSCOT":
        params = {
            "epsilon": cfg.method.epsilon,
            "alpha": cfg.method.alpha,
            "rep": cfg.method.rep,
            "cost": cfg.method.cost,
        }
        results = _moscot(
            adata_sc=adata_sp_a,
            adata_sp_train=adata_sp_b,
            true_df=true_df,
            method=method,
            params=params,
            path_results=path_results,
            unique_id=unique_id,
        )
        pd.DataFrame(results, index=[0]).to_csv(path_results / "results.csv")
    elif method == "GIMVI":
        params = {"epochs": cfg.method.epochs, "n_latent": cfg.method.n_latent}
        results = _gimvi(
            adata_sc=adata_sp_a,
            adata_sp_train=adata_sp_b,
            true_df=true_df,
            method=method,
            params=params,
            path_results=path_results,
            unique_id=unique_id,
        )
        pd.DataFrame(results, index=[0]).to_csv(path_results / "results.csv")
    elif method == "TANGRAM":
        params = {"learning_rate": cfg.method.learning_rate, "num_epochs": cfg.method.num_epochs}
        results = _tangram(
            adata_sc=adata_sp_a,
            adata_sp_train=adata_sp_b,
            true_df=true_df,
            method=method,
            params=params,
            path_results=path_results,
            unique_id=unique_id,
        )
        pd.DataFrame(results, index=[0]).to_csv(path_results / "results.csv")
    else:
        raise ValueError("Method not implemented")


@hydra.main(config_path=".", config_name="config.yaml", version_base="1.2")
def main(cfg: DictConfig) -> None:
    sweep_params = HydraConfig.get().job.override_dirname
    print("SWEEP PARAMS", sweep_params)
    benchmark(cfg)


if __name__ == "__main__":
    main()

# python experiment.py +method.name=moscot +method.epsilon=1,10 +method.lambda=10 --multirun
