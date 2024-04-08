from pathlib import Path
import os

from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
import hydra
import pandas as pd

from anndata import AnnData


def process_adata(cfg: DictConfig) -> tuple[AnnData, AnnData]:
    from sklearn.preprocessing import StandardScaler

    import numpy as np

    import scanpy as sc

    StandardScaler()
    path_data = Path(cfg.paths.path_data)
    adata_rna = sc.read(path_data / cfg.paths.adata_rna)
    adata_atac = sc.read(path_data / cfg.paths.adata_atac)

    spatial = adata_atac.obsm["spatial"]
    adata_atac.obsm["spatial_scaled"] = (spatial - spatial.mean()) / spatial.std()
    adata_rna.obsm["spatial_scaled"] = adata_atac.obsm["spatial_scaled"].copy()
    # adata_rna.obsm["spatial_scaled"] = np.hstack([adata_rna.obsm["X_pca"], adata_rna.obsm["spatial_scaled"]])
    adata_rna.obsm["X_pca_ATAC"] = np.hstack([adata_atac.obsm["X_pca"], adata_rna.obsm["X_pca"]])
    adata_rna.obsm["ATAC"] = pd.DataFrame(
        adata_atac.X.copy(), columns=adata_atac.var_names.values, index=adata_rna.obs_names
    )
    adata_sc = sc.pp.subsample(adata_rna, fraction=0.5, copy=True, random_state=42)
    adata_sp = adata_rna[~np.in1d(adata_rna.obs_names, adata_sc.obs_names)].copy()
    return adata_sc, adata_sp


def _corr_results(true_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
    pred_df = pred_df[true_df.columns].copy()
    corr_pearson = pred_df.corrwith(true_df, method="pearson")
    corr_spearman = pred_df.corrwith(true_df, method="spearman")
    out = pd.concat([corr_pearson, corr_spearman], axis=1)
    out.columns = ["pearson", "spearman"]
    return out


def benchmark(cfg):
    from datetime import datetime

    from jax.config import config
    import numpy as np

    config.update("jax_enable_x64", True)

    unique_id = HydraConfig.get().job.override_dirname
    path_results = Path(cfg.paths.path_results) / unique_id

    os.makedirs(path_results, exist_ok=True)

    adata_sc, adata_spatial = process_adata(cfg)

    adata_sc.uns["marker_peaks"] = np.unique(adata_sc.uns["marker_peaks"]).tolist()  # gene is duplicated
    atac_real = pd.DataFrame(
        adata_spatial.obsm["ATAC"].loc[:, adata_sc.uns["marker_peaks"]].copy(),
        columns=adata_sc.uns["marker_peaks"],
        index=adata_spatial.obs_names,
    )

    if cfg.method.name == "MOSCOT":
        from moscot.problems.space import MappingProblem

        mp = MappingProblem(adata_sc, adata_spatial).prepare(
            sc_attr={"attr": "obsm", "key": cfg.moscot.sc_attr},
            joint_attr={"attr": "X"},
            spatial_key="spatial_scaled",
            normalize_spatial=False,
            cost=cfg.moscot.cost,
        )

        start_time = datetime.now()

        mp = mp.solve(
            alpha=cfg.moscot.alpha,
            epsilon=cfg.moscot.epsilon,
            tau_a=cfg.moscot.tau_a,
            tau_b=1,
            max_iterations=7000,
            threshold=1e-5,
        )
        end_time = datetime.now()

        mp.problems[("src", "tgt")].solution.plot_costs(save=path_results / "costs.png")

        # impute atac
        atac = adata_sc.obsm["ATAC"].loc[:, adata_sc.uns["marker_peaks"]].values
        spatial_atac = mp.problems[("src", "tgt")].solution.pull(
            atac,
            scale_by_marginals=True,
        )
        atac_predicted = pd.DataFrame(spatial_atac, columns=adata_sc.uns["marker_peaks"], index=adata_spatial.obs_names)

        np.testing.assert_array_equal(atac_predicted.columns, atac_real.columns)
        corr_results = _corr_results(atac_real, atac_predicted)
        corr_results.to_csv(path_results / "corr_results.csv")

        results = {"time": end_time - start_time}
        # get results
        for corr in ["pearson", "spearman"]:
            results[f"corr_results_mean_{corr}"] = corr_results[corr].mean()
            results[f"corr_results_median_{corr}"] = corr_results[corr].median()
            results[f"corr_results_var_{corr}"] = corr_results[corr].var()
            results[f"corr_results_max_{corr}"] = corr_results[corr].max()
            results[f"corr_results_min_{corr}"] = corr_results[corr].min()

        cell_trans = mp.cell_transition(
            source="src",
            target="tgt",
            aggregation_mode="annotation",
            source_groups="ATAC_clusters",
            target_groups="ATAC_clusters",
        )
        results["mean_diagonal_cell_transition"] = np.diag(cell_trans).mean()

    elif cfg.method.name == "TANGRAM":
        import torch
        import tangram as tg

        import scanpy as sc
        import anndata as ad

        tg.pp_adatas(adata_sc, adata_spatial, genes=adata_sc.var_names.tolist(), gene_to_lowercase=True)
        device = torch.device("cuda")

        start_time = datetime.now()
        ad_map = tg.map_cells_to_space(
            adata_sc,
            adata_spatial,
            device=device,
        )
        end_time = datetime.now()
        atac = adata_sc.obsm["ATAC"].loc[:, adata_sc.uns["marker_peaks"]].values
        adata_atac = ad.AnnData(atac, obs=adata_sc.obs)
        adata_atac.var_names = adata_sc.uns["marker_peaks"]
        ad_ge = tg.project_genes(ad_map, adata_atac)
        ad_ge.var_names = adata_sc.uns["marker_peaks"]
        atac_predicted = sc.get.obs_df(ad_ge, keys=adata_sc.uns["marker_peaks"])
        atac_real = atac_real.loc[:, adata_sc.uns["marker_peaks"]]
        np.testing.assert_array_equal(atac_predicted.columns, atac_real.columns)
        corr_results = _corr_results(atac_real, atac_predicted)
        results = {"time": end_time - start_time}
        results["method"] = "TANGRAM"
        # get results
        for corr in ["pearson", "spearman"]:
            results[f"corr_results_mean_{corr}"] = corr_results[corr].mean()
            results[f"corr_results_median_{corr}"] = corr_results[corr].median()
            results[f"corr_results_var_{corr}"] = corr_results[corr].var()
            results[f"corr_results_max_{corr}"] = corr_results[corr].max()
            results[f"corr_results_min_{corr}"] = corr_results[corr].min()

    corr_results.to_csv(path_results / "corr_results.csv")
    pd.DataFrame(results, index=[0]).to_csv(path_results / "results.csv")


@hydra.main(config_path=".", config_name="config.yaml", version_base="1.2")
def main(cfg: DictConfig) -> None:
    sweep_params = HydraConfig.get().job.override_dirname
    print("SWEEP PARAMS", sweep_params)
    benchmark(cfg)


if __name__ == "__main__":
    main()
