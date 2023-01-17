"""From From https://github.com/TUM-DAML/seml/blob/master/examples/advanced_example_experiment.py ."""

import logging
import warnings

from sacred import Experiment
import seml

import numpy as np

warnings.filterwarnings("ignore")

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


class ExperimentWrapper:
    """
    Experiment wrapper.

    Initialize models and datasets to run experiment from config file.
    """

    def __init__(self, init_all=True):
        pass

        if init_all:
            self.init_all()

    @ex.capture(prefix="data")
    def init_dataset(self, dataset: dict):
        """Perform train-test split and dataset loading."""
        import numpy as np

        import scanpy as sc

        dataset_logger = logging.getLogger()
        dataset_logger.info("Reading adata.")

        adata = sc.read(dataset["adata"])

        rng = np.random.default_rng(dataset["seeds"])

        adata = adata[adata.obs.synth_batch == str(dataset["batch"])].copy()
        adata.X = rng.normal(size=adata.X.shape) + adata.X.copy()
        p = dataset["fraction"]
        idx_subsample = rng.choice([True, False], size=(adata.shape[0]), p=[p, 1 - p])
        adata = adata[idx_subsample].copy()

        self.adata = adata

        dataset_logger.info("Init dataset finished.")

    @ex.capture(prefix="model")
    def init_model(self, solver, tool):
        from moscot.problems.space import AlignmentProblem
        import pandas as pd

        self.model_type = tool["type"]
        if self.model_type == "moscot":
            self.adata.obs["batch"] = pd.Categorical(self.adata.obs["batch"].astype(str))
            self.problem = AlignmentProblem(self.adata).prepare(batch_key="batch")

            self.alpha = solver["alpha"]
            self.epsilon = solver["epsilon"]
            self.rank = solver["rank"]
        elif self.model_type == "gpsa":
            self.kernel = solver["kernel"]
            self.n_epochs = solver["n_epochs"]
            self.lr = solver["lr"]
        elif self.model_type == "paste":
            self.alpha = solver["alpha"]
            self.dissimilarity = solver["dissimilarity"]
            self.norm = solver["norm"]
        else:
            raise ValueError("Wrong model type.")

    def init_all(self):
        """Sequentially run the sub-initializers of the experiment."""
        self.init_dataset()
        self.init_model()

    @ex.capture(prefix="train")
    def train(self, training: dict):

        if self.model_type == "moscot":
            test_results = compute_moscot(self.problem, self.adata, self.epsilon, self.rank, self.alpha)
        elif self.model_type == "gpsa":
            test_results = compute_gpsa(self.adata, self.kernel, self.n_epochs, self.lr)
        elif self.model_type == "paste":
            test_results = compute_gpsa(self.adata, self.alpha, self.dissimilarity, self.norm)
        else:
            raise ValueError("Wrong model type.")
        test_results["n_obs"] = self.adata.shape[0]
        test_results["n_var"] = self.adata.shape[1]

        return test_results


def compute_gpsa(adata, kernel, n_epochs, lr):
    from time import perf_counter

    from gpsa import rbf_kernel, matern12_kernel
    from sklearn.metrics import mean_squared_error
    from gpsa.models.vgpsa import VariationalGPSA
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    X = adata.obsm["spatial"].copy()
    Y = adata.X.copy()
    view_idx = [np.where(adata.obs.batch.values == ii)[0] for ii in range(2)]
    n_samples_list = [len(x) for x in view_idx]

    x = torch.from_numpy(X).float().clone()
    y = torch.from_numpy(Y).float().clone()

    data_dict = {
        "expression": {
            "spatial_coords": x,
            "outputs": y,
            "n_samples_list": n_samples_list,
        }
    }
    N_SPATIAL_DIMS = 2
    M_G = 50
    M_X_PER_VIEW = 50
    FIXED_VIEW_IDX = 0
    N_LATENT_GPS = {"expression": None}

    N_EPOCHS = 1000
    KERNEL = rbf_kernel if kernel == "RBF" else matern12_kernel

    model = VariationalGPSA(
        data_dict,
        n_spatial_dims=N_SPATIAL_DIMS,
        m_X_per_view=M_X_PER_VIEW,
        m_G=M_G,
        data_init=True,
        minmax_init=False,
        grid_init=False,
        n_latent_gps=N_LATENT_GPS,
        mean_function="identity_fixed",
        kernel_func_warp=KERNEL,
        kernel_func_data=KERNEL,
        fixed_view_idx=FIXED_VIEW_IDX,
    ).to(device)

    view_idx, Ns, _, _ = model.create_view_idx_dict(data_dict)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    def train(model, loss_fn, optimizer):
        model.train()

        # Forward pass
        G_means, G_samples, F_latent_samples, F_samples = model.forward(
            {"expression": x}, view_idx=view_idx, Ns=Ns, S=5
        )

        # Compute loss
        loss = loss_fn(data_dict, F_samples)

        # Compute gradients and take optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item(), G_means

    start = perf_counter()
    for _ in range(N_EPOCHS):
        _, G_means = train(model, model.loss_fn, optimizer)
    compute_time = perf_counter() - start

    adata.obsm["spatial_aligned"] = G_means["expression"].detach().numpy()
    ad1 = adata[adata.obs.batch == 0].copy()
    ad2 = adata[adata.obs.batch == 1].copy()
    _, comm1, comm2 = np.intersect1d(ad1.obs.idx, ad2.obs.idx, return_indices=True)

    test_results = {}
    test_results["mse"] = mean_squared_error(ad1.obsm["spatial_aligned"][comm1], ad2.obsm["spatial_aligned"][comm2])
    test_results["time"] = compute_time

    return test_results


def compute_moscot(problem, adata, epsilon, rank, alpha):
    from time import perf_counter

    from sklearn.metrics import mean_squared_error

    start = perf_counter()
    problem = problem.solve(
        epsilon=epsilon,
        rank=rank,
        alpha=alpha,
        max_iterations=100,
        threshold=1e-5,
    )
    compute_time = perf_counter() - start
    problem.align(reference="0")
    ad1 = adata[adata.obs.batch == "0"].copy()
    ad2 = adata[adata.obs.batch == "1"].copy()

    _, comm1, comm2 = np.intersect1d(ad1.obs.idx, ad2.obs.idx, return_indices=True)

    test_results = {}
    test_results["mse"] = mean_squared_error(ad1.obsm["spatial_norm_warp"][comm1], ad2.obsm["spatial_norm_warp"][comm2])
    test_results["time"] = compute_time

    return test_results


def compute_paste(adata, alpha, dissimilarity, norm):
    from time import perf_counter

    from sklearn.metrics import mean_squared_error
    import paste as pst

    batch1 = adata[adata.obs.batch == 0].copy()
    batch2 = adata[adata.obs.batch == 1].copy()
    start = perf_counter()
    pi12 = pst.pairwise_align(batch1, batch2, alpha=alpha, dissimilarity=dissimilarity, norm=norm)
    compute_time = perf_counter() - start
    pi12norm = pi12 / pi12.sum(0)
    out = (batch1.obsm["spatial"].T @ pi12norm).T
    batch1.obsm["spatial_warp"] = batch1.obsm["spatial"]
    batch2.obsm["spatial_warp"] = out

    _, comm1, comm2 = np.intersect1d(batch1.obs.idx, batch2.obs.idx, return_indices=True)

    test_results = {}
    test_results["mse"] = mean_squared_error(batch1.obsm["spatial_warp"][comm1], batch2.obsm["spatial_warp"][comm2])
    test_results["time"] = compute_time

    return test_results


# We can call this command, e.g., from a Jupyter notebook with init_all=False to get an "empty" experiment wrapper,
# where we can then for instance load a pretrained model to inspect the performance.
@ex.command(unobserved=True)
def get_experiment(init_all=False):
    print("get_experiment")
    experiment = ExperimentWrapper(init_all=init_all)
    return experiment


# This function will be called by default. Note that we could in principle manually pass an experiment instance,
# e.g., obtained by loading a model from the database or by calling this from a Jupyter notebook.
@ex.automain
def train(experiment=None):
    if experiment is None:
        experiment = ExperimentWrapper()
    return experiment.train()
