from typing import Tuple, Union, Optional, List, Dict, Any
 
from scipy.sparse import issparse, csr_matrix, dok_matrix
from moscot.backends.ott import OTTOutput
from moscot.problems.base import BirthDeathProblem
from anndata import AnnData
from ott.core import sinkhorn
from Bio import Phylo
from ott.geometry import pointcloud
from copy import deepcopy
import numpy as np
import jax.numpy as jnp
import numpy.typing as npt
import io
import networkx as nx
from numbers import Number


import pickle
from copy import deepcopy
from logging import getLogger
from pathlib import Path

from jax.config import config
from scipy.sparse import issparse

config.update("jax_enable_x64", True)

import io
import warnings
from copy import deepcopy
from types import MappingProxyType
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple, Union

import cassiopeia as cas
import lineageot.core as lot_core
import lineageot.evaluation as lot_eval
import lineageot.inference as lot_inf
import moscot
import networkx as nx
import numpy as np
import numpy.typing as npt
import pandas as pd
import scanpy as sc
from anndata import AnnData
from Bio import Phylo
from jax import numpy as jnp
from sklearn.metrics.pairwise import euclidean_distances

CASETTE_SIZE = 4
N_CASETTES = 8



Output = OTTOutput


def distance_between_pushed_masses(
    gex_data_source: npt.ArrayLike,
    gex_data_target: npt.ArrayLike,
    output: Union[npt.ArrayLike, csr_matrix, BirthDeathProblem],
    true_coupling: Union[npt.ArrayLike, csr_matrix],
    eps: float = 0.1,
    seed: Optional[int] = None,
    n_samples: Optional[int] = None,
    ) -> float:
    source_space_cost = _distance_pushed_masses(
        gex_data_source, output, true_coupling, forward=False, eps=eps, random=False, seed=seed, n_samples=n_samples
    )
    target_space_cost = _distance_pushed_masses(
        gex_data_target, output, true_coupling, forward=True, eps=eps, random=False, seed=seed, n_samples=n_samples
    )
    independent_coupling_source_space_cost = _distance_pushed_masses(
        gex_data_source, output, true_coupling, forward=False, random=True, seed=seed, n_samples=n_samples
    )
    independent_coupling_target_space_cost = _distance_pushed_masses(
        gex_data_target, output, true_coupling, forward=True, random=True, seed=seed, n_samples=n_samples
    )
    source_error = source_space_cost / independent_coupling_source_space_cost
    target_error = target_space_cost / independent_coupling_target_space_cost
    mean_error = (source_error + target_error) / 2
    return mean_error


def _get_masses_moscot(
    i: int,
    output: Union[npt.ArrayLike, csr_matrix, BirthDeathProblem],
    true_coupling: Union[npt.ArrayLike, csr_matrix],
    n: int,
    m: int,
    forward: bool,
    random: bool,
    ) -> Tuple[npt.ArrayLike]:
    if forward:
        pushed_mass_true = true_coupling[i, :]
        pushed_mass_true /= pushed_mass_true.sum()
        mass = np.zeros(n, dtype="float64")
        mass[i] = 1
        pushed_mass = output.solution.push(mass).squeeze()
        weight_factor = jnp.sum(pushed_mass)
        if random:
            pushed_mass = output.b[:, -1]
            pushed_mass /= pushed_mass.sum()
        else:
            pushed_mass /= weight_factor
    else:
        pushed_mass_true = true_coupling.T[i, :]
        pushed_mass_true /= pushed_mass_true.sum()
        mass = np.zeros(m, dtype="float64")
        mass[i] = 1
        pushed_mass = output.solution.pull(mass).squeeze()
        weight_factor = jnp.sum(pushed_mass)
        if random:
            pushed_mass = output.a[:, -1]
            pushed_mass /= pushed_mass.sum()
        else:
            pushed_mass /= weight_factor
    return pushed_mass_true.astype("float64"), pushed_mass, weight_factor


def _get_masses_ndarray(
    i: int,
    output: Union[npt.ArrayLike, csr_matrix],
    true_coupling: Union[npt.ArrayLike, csr_matrix],
    forward: bool,
    random: bool) -> Tuple[npt.ArrayLike, ...]:
    if forward:
        pushed_mass_true = true_coupling[i, :]
        pushed_mass_true /= pushed_mass_true.sum()
        weight_factor = output[i, :].sum()
        if random:
            pushed_mass = output.sum(0)
            pushed_mass /= pushed_mass.sum()
        else:
            pushed_mass = output[i, :]
            pushed_mass /= weight_factor
    else:
        pushed_mass_true = true_coupling.T[i, :]
        pushed_mass_true /= pushed_mass_true.sum()
        weight_factor = output.T[i, :].sum()
        if random:
            pushed_mass = output.sum(1)
            pushed_mass /= pushed_mass.sum()
        else:
            pushed_mass = output.T[i, :]
            pushed_mass /= weight_factor
    return pushed_mass_true.astype("float64"), pushed_mass.astype("float64"), weight_factor

 
def _distance_pushed_masses(
    gex_data: npt.ArrayLike,
    output: Union[npt.ArrayLike, csr_matrix, BirthDeathProblem],
    true_coupling: Union[npt.ArrayLike, csr_matrix],
    forward: bool,
    eps: float = 0.5,
    random: bool = False,
    seed: Optional[int] = None,
    n_samples: Optional[int] = None) -> float:
    rng = np.random.RandomState(seed=seed)
    n, m = output.shape if isinstance(output, np.ndarray) else output.solution.shape
    wasserstein_d = 0
    total_weight = 0
    if n_samples is None:
        samples = range(n if forward else m)
    else:
        samples = rng.choice(n if forward else m, size=n_samples)
    for i in samples:
        if isinstance(output, np.ndarray):
            pushed_mass_true, pushed_mass, weight_factor = _get_masses_ndarray(
                i, output, true_coupling, forward, random
            )
        elif isinstance(output, BirthDeathProblem):
            pushed_mass_true, pushed_mass, weight_factor = _get_masses_moscot(
                i, output, true_coupling, n, m, forward, random
            )
        else:
            raise TypeError(f"Return type is {type(output)}")
        if issparse(pushed_mass_true):
            pushed_mass_true = np.squeeze(pushed_mass_true.A)
        geom = pointcloud.PointCloud(gex_data, gex_data, epsilon=eps, scale_cost="mean")
        out = sinkhorn.sinkhorn(geom, pushed_mass_true, pushed_mass, max_iterations=1e7)
        wasserstein_d += float(out.reg_ot_cost) * weight_factor
        total_weight = +weight_factor
        del geom
        del out

    return wasserstein_d / (len(samples) * len(pushed_mass) * total_weight)



# take from Cassiopeia
def get_cassettes() -> List[int]:
    cassettes = [(CASETTE_SIZE * j) for j in range(0, N_CASETTES)]
    return cassettes


def silence_cassettes(
    character_array: np.ndarray, silencing_rate: float, missing_state: int = -1
) -> np.ndarray:
    updated_character_array = character_array.copy()
    cassettes = get_cassettes()
    cut_site_by_cassette = np.digitize(range(len(character_array)), cassettes)

    for cassette in range(1, N_CASETTES + 1):
        if np.random.uniform() < silencing_rate:
            indices = np.where(cut_site_by_cassette == cassette)
            left, right = np.min(indices), np.max(indices)
            for site in range(left, right + 1):
                updated_character_array[site] = missing_state

    return updated_character_array


def stochastic_silencing(
    barcodes: np.ndarray,
    stochastic_silencing_rate: float = 1e-2,
    stochastic_missing_data_state: int = -1,
) -> np.ndarray:
    assert 0 <= stochastic_silencing_rate <= 1.0, stochastic_silencing_rate
    barcodes_ss = np.zeros(barcodes.shape)
    for i, bc in enumerate(barcodes):
        barcodes_ss[i, :] = silence_cassettes(
            bc, stochastic_silencing_rate, stochastic_missing_data_state
        )
    return barcodes_ss


def run_moscot(
    edist: Optional[jnp.ndarray] = None,
    ldist: Optional[jnp.ndarray] = None,
    rna_dist: Optional[jnp.ndarray] = None,
    alpha: float = 0,
    epsilon: Optional[float] = None,
    rank: int = -1,
    scale_cost: Optional[Literal["mean", "max_cost"]] = "max_cost",
    **kwargs: Any,
) -> Tuple[np.ndarray, bool]:
    if alpha == 0:
        solver = moscot.backends.ott.SinkhornSolver(rank=rank)
        ot_prob = solver(
            xy=rna_dist,
            tags={"xy": "cost"},
            epsilon=epsilon,
            scale_cost=scale_cost,
            **kwargs,
        )
    elif alpha == 1:
        solver = moscot.backends.ott.GWSolver(epsilon=epsilon, rank=rank)
        ot_prob = solver(
            x=edist,
            y=ldist,
            tags={"x": "cost", "y": "cost"},
            epsilon=epsilon,
            scale_cost=scale_cost,
            **kwargs,
        )
    else:
        solver = moscot.backends.ott.FGWSolver(epsilon=epsilon, rank=rank)
        ot_prob = solver(
            xy=rna_dist,
            x=edist,
            y=ldist,
            tags={"xy": "cost", "x": "cost", "y": "cost"},
            epsilon=epsilon,
            alpha=alpha,
            scale_cost=scale_cost,
            **kwargs,
        )

    return ot_prob.transport_matrix, ot_prob.converged


def run_lot(
    barcode_arrays: Mapping[Literal["early", "late"], np.ndarray],
    rna_arrays: Mapping[Literal["early", "late"], np.ndarray],
    sample_times: Mapping[Literal["early", "late"], float],
    *,
    tree: Optional[nx.DiGraph] = None,
    epsilon: float = 0.05,
    normalize_cost: bool = True,
    **kwargs: Any,
) -> Tuple[np.ndarray, bool]:
    """Fits a LineageOT coupling between the cells at time_1 and time_2.
    In the process, annotates the lineage tree with observed and estimated cell states.
    Parameters
    ----------
    tree:
        The lineage tree fitted to cells at time_2. Nodes should already be annotated with times.
        Annotations related to cell state will be added.
    sample_times: Dict
        sampling times of late and early cells
    barcode_arrays: Dict
        barcode arrays of late and early cells
    rna_arrays: Dict
        expression space arrays of late and early cells
    epsilon : float (default 0.05)
        Entropic regularization parameter for optimal transport
    normalize_cost : bool (default True)
        Whether to rescale the cost matrix by its median before fitting a coupling.
        Normalizing this way allows us to choose a reasonable default epsilon for data of any scale
    Returns
    -------
    The coupling and whether marginals are satisfied.
    """
    time_key = "time"

    eadata = AnnData(
        rna_arrays["early"],
        obsm={"barcodes": barcode_arrays["early"]},
        dtype=np.float64,
    )
    ladata = AnnData(
        rna_arrays["late"], obsm={"barcodes": barcode_arrays["late"]}, dtype=np.float64
    )

    adata = eadata.concatenate(ladata, batch_key=time_key)
    adata.obs[time_key] = adata.obs[time_key].cat.rename_categories(
        {"0": sample_times["early"], "1": sample_times["late"]}
    )

    if tree is None:
        tree = lot_core.fit_tree(ladata, sample_times["late"])
    else:
        lot_inf.add_leaf_x(tree, ladata.X)

    with warnings.catch_warnings(record=True) as ws:
        coupling = lot_core.fit_lineage_coupling(
            adata,
            sample_times["early"],
            sample_times["late"],
            lineage_tree_t2=tree,
            epsilon=epsilon,
            normalize_cost=normalize_cost,
            **kwargs,
        )
    coupling = coupling.X.astype(np.float64)
    conv = not [w for w in ws if "did not converge" in str(w.message)]

    return coupling, conv and np.all(np.isfinite(coupling))


def cassiopeia_distances(
    barcodes: np.ndarray,
    solver: Literal["nj", "greedy", "ilp", "hybrid"] = "greedy",
    estim: Literal["mle", "bayesian", "const"] = "mle",
    bayesian_estimator_kwargs=MappingProxyType({}),
    only_tree: bool = False,
    hybrid_solver_kwargs=MappingProxyType({}),
    **kwargs: Any,
) -> np.ndarray:
    n = barcodes.shape[0]
    barcodes = pd.DataFrame(barcodes, index=map(str, range(n)))
    tree = cas.data.CassiopeiaTree(character_matrix=barcodes)

    # for ILPSolver
    kwargs.setdefault("convergence_time_limit", 600)  # 10mins
    kwargs.setdefault("maximum_potential_graph_layer_size", 10000)

    if estim == "bayesian":
        print(f"Setting solver to neighbor joining, because `estim={estim}`.")
        solver = "nj"

    if solver == "nj":
        solver = cas.solver.NeighborJoiningSolver(add_root=True)
    elif solver == "greedy":
        solver = cas.solver.VanillaGreedySolver()
    elif solver == "ilp":
        solver = cas.solver.ILPSolver(weighted=False, seed=1234, **kwargs)
    elif solver == "hybrid":
        ts = cas.solver.VanillaGreedySolver()
        bs = cas.solver.ILPSolver(weighted=False, seed=1234, **kwargs)
        solver = cas.solver.HybridSolver(
            top_solver=ts, bottom_solver=bs, **hybrid_solver_kwargs
        )
    else:
        raise NotImplementedError(f"Solver `{solver}` is not yet implemented.")

    solver.solve(tree, collapse_mutationless_edges=estim != "bayesian")
    if only_tree:
        G = deepcopy(tree._CassiopeiaTree__network)
        G = nx.relabel_nodes(G, dict(zip(map(str, range(n)), range(n))))
        root = [n for n in G.nodes if G.in_degree(n) == 0]
        assert len(root) == 1
        root = root[0]
        G = nx.relabel_nodes(G, {root: "root"})
        return tree, G, None

    if estim == "mle":
        estim = cas.tools.branch_length_estimator.IIDExponentialMLE()
        estim.estimate_branch_lengths(tree)
    elif estim == "bayesian":
        # root must have 1 child
        # otherwise, must be a full binary tree
        root = tree.root
        tree._CassiopeiaTree__add_node("synroot")
        tree._CassiopeiaTree__add_edge("synroot", root)
        tree._CassiopeiaTree__cache["root"] = "synroot"
        tree.reconstruct_ancestral_characters()
        # tree.set_character_states("synroot", [])
        # tree._CassiopeiaTree__network.nodes["synroot"]['character_states'] = []
        estim = cas.tools.branch_length_estimator.IIDExponentialBayesian(
            **bayesian_estimator_kwargs
        )
        estim.estimate_branch_lengths(tree)
        tree._CassiopeiaTree__remove_node("synroot")
        tree._CassiopeiaTree__cache["root"] = root
    elif estim == "const":
        pass
    else:
        raise NotImplementedError(estim)

    G = deepcopy(tree._CassiopeiaTree__network)
    G = nx.relabel_nodes(G, dict(zip(map(str, range(n)), range(n))))
    root = [n for n in G.nodes if G.in_degree(n) == 0]
    assert len(root) == 1
    root = root[0]
    G = nx.relabel_nodes(G, {root: "root"})
    if estim == "const":
        for e in G.edges:
            G.edges[e]["time"] = 1.0
        return lot_inf.compute_tree_distances(G)

    lot_inf.add_division_times_from_vertex_times(G, current_node="root")
    G = lot_inf.add_times_to_edges(G)

    return lot_inf.compute_tree_distances(G)


def process_data(
    rna_arrays: Mapping[Literal["early", "late"], np.ndim], *, n_pcs: int = 30
) -> np.ndarray:
    adata = AnnData(rna_arrays["early"], dtype=float).concatenate(
        AnnData(rna_arrays["late"], dtype=float),
        batch_key="time",
        batch_categories=["0", "1"],
    )
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata)
    sc.tl.pca(adata, use_highly_variable=False)

    return adata.obsm["X_pca"][:, :n_pcs].astype(float, copy=True)


def compute_dists(
    tree_type: Literal["gt", "bc", "cas_dists", "cas"],
    trees: Mapping[Literal["early", "late"], nx.DiGraph],
    rna_arrays: Mapping[Literal["early", "late"], np.ndarray],
    barcode_arrays: Mapping[Literal["early", "late"], np.ndarray],
    dist_cache: Optional[Path] = None,
    scale: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute distances for time_1, time_2 and joint.
    Parameters
    ----------
    trees:
        The lineage tree fitted to cells at time_1 and time_2. Nodes should already be annotated with times.
        Annotations related to cell state will be added.
    barcode_arrays: barcode arrays of late and early cells.
    tree_type: the type of distance to evaluate.
    dist_cache: Path to a cache file where to read/write the distance matrices.
    Returns
    -------
    Early and late distances.
    """

    def maybe_scale(arr: np.ndarray) -> np.ndarray:
        if np.any(~np.isfinite(edist)):
            raise ValueError("Non-finite values found in distances.")
        return arr / np.max(arr) if scale else arr

    if dist_cache is not None and dist_cache.is_file():
        logger.info(f"Loading distances from `{dist_cache}`")
        with open(dist_cache, "rb") as fin:
            return pickle.load(fin)

    if tree_type == "gt":
        edist = lot_inf.compute_tree_distances(trees["early"])
        ldist = lot_inf.compute_tree_distances(trees["late"])
    elif tree_type == "bc":
        edist = lot_inf.barcode_distances(barcode_arrays["early"])
        ldist = lot_inf.barcode_distances(barcode_arrays["late"])
        edist[np.isnan(edist)] = np.nanmax(edist)
        ldist[np.isnan(ldist)] = np.nanmax(ldist)
    elif tree_type == "cas_dists":  # cassiopeia w\o mle
        edist = cassiopeia_distances(barcode_arrays["early"], estim="const")
        ldist = cassiopeia_distances(barcode_arrays["late"], estim="const")
    elif tree_type == "cas":  # cassiopeia
        edist = cassiopeia_distances(barcode_arrays["early"], estim="mle")
        ldist = cassiopeia_distances(barcode_arrays["late"], estim="mle")
    else:
        raise NotImplementedError(f"Tree type `{tree_type}` not yet implemented.")

    rna_dist = euclidean_distances(rna_arrays["early"], rna_arrays["late"])
    edist, ldist, rna_dist = (
        maybe_scale(edist),
        maybe_scale(ldist),
        maybe_scale(rna_dist),
    )

    if dist_cache is not None:
        logger.info(f"Saving distances to `{dist_cache}`")
        with open(dist_cache, "wb") as fout:
            pickle.dump((edist, ldist, rna_dist), fout)

    return edist, ldist, rna_dist


def is_leaf(G: nx.DiGraph, n: Any) -> bool:
    return not list(nx.descendants(G, n))


def newick2digraph(tree: str) -> nx.DiGraph:
    def trav(clade, prev: Any, depth: int) -> None:
        nonlocal cnt
        if depth == 0:
            name = "root"
        else:
            name = clade.name
            if name is None:
                name = cnt
                cnt -= 1
            else:
                name = int(name[1:]) - 1

        G.add_node(name, node_depth=depth)
        if prev is not None:
            G.add_edge(prev, name)

        for c in clade.clades:
            trav(c, name, depth + 1)

    G = nx.DiGraph()
    cnt = -1
    tree = Phylo.read(io.StringIO(tree), "newick")
    trav(tree.clade, None, 0)

    start = max([n for n in G.nodes if n != "root"]) + 1
    for n in list(nx.dfs_preorder_nodes(G)):
        if n == "root":
            pass
        if is_leaf(G, n):
            continue

        assert start not in G.nodes
        G = nx.relabel_nodes(G, {n: start}, copy=False)
        start += 1

    return G


def annotate(
    G: nx.DiGraph,
    cell_arr_data: List[lot_inf.sim.Cell],
    meta: List[Dict[str, Any]],
    ttp: int = 100,
) -> nx.DiGraph:
    G = G.copy()
    n_leaves = len([n for n in G.nodes if not len(list(G.successors(n)))])
    assert (n_leaves & (n_leaves - 1)) == 0, f"{n_leaves} is not power of 2"
    max_depth = int(np.log2(n_leaves))

    n_expected_nodes = 2 ** (max_depth + 1) - 1
    assert len(G) == n_expected_nodes, "graph is not a full binary tree"

    if len(cell_arr_data) != n_expected_nodes:  # missing root, add after observed nodes
        dummy_cell = lot_inf.sim.Cell([], [])
        cell_arr_data += [dummy_cell]
        meta += [{}]

    for nid in G.nodes:
        depth = G.nodes[nid]["node_depth"]
        metadata = {
            **meta[nid],  # contains `depth`, which is different from `node_depth`
            "cell": cell_arr_data[nid],
            "nid": nid,
            "time": depth * ttp,
            "time_to_parent": ttp,
        }
        G.nodes[nid].update(metadata)

    for eid in G.edges:
        G.edges[eid].update({"time": ttp})

    return nx.relabel_nodes(G, {n_leaves: "root"}, copy=False)


def cut_at_depth(G: nx.DiGraph, *, max_depth: Optional[int] = None) -> nx.DiGraph:
    if max_depth is None:
        return deepcopy(G)
    selected_nodes = [n for n in G.nodes if G.nodes[n]["node_depth"] <= max_depth]
    G = deepcopy(G.subgraph(selected_nodes).copy())

    # relabel because of LOT
    leaves = [n for n in G.nodes if not len(list(G.successors(n)))]
    for new_name, n in enumerate(leaves):
        G = nx.relabel_nodes(G, {n: new_name}, copy=False)
    return G


def is_valid_edge(n1: Dict[str, Any], n2: Dict[str, Any]) -> bool:
    r"""Assumes the following state tree:
       /-4
      7
     / \-3
    5
     \ /-1
      6
       \-2
    """
    state_tree = nx.from_edgelist([(5, 6), (5, 7), (6, 1), (6, 2), (7, 3), (7, 4)])
    try:
        # parent, cluster, depth
        p1, c1, d1 = n1["parent"], n1["cluster"], n1["depth"]
        p2, c2, d2 = n2["parent"], n2["cluster"], n2["depth"]
    except KeyError:
        # no metadata, assume true
        return True

    # root, anything is permitted
    if (p1, c1, d1) == (5, 6, 0):
        return True

    # sanity checks
    assert p1 in [5, 6, 7], p1
    assert p2 in [5, 6, 7], p2
    assert c1 in [1, 2, 3, 4, 6, 7], c1
    assert c2 in [1, 2, 3, 4, 6, 7], c2

    if p1 == p2:
        if c1 == c2:
            # check if depth of a parent is <=
            return d1 <= d2
        # sanity check that clusters are valid siblings
        return (c1, c2) in state_tree.edges

    # parent-cluster relationship
    assert c1 == p2, (c1, p2)
    # valid transition
    assert (c1, c2) in state_tree.edges, (c1, c2)
    return True


def build_true_trees(
    rna: np.ndarray,
    barcodes: np.ndarray,
    meta: pd.DataFrame,
    *,
    tree: str,
    depth: int,
    n_pcs: int = 30,
    ttp: float = 100.0,
) -> Dict[Literal["early", "late"], nx.DiGraph]:
    cell_arr_adata = [
        lot_inf.sim.Cell(rna[nid], barcodes[nid]) for nid in range(rna.shape[0])
    ]
    metadata = [meta.iloc[nid].to_dict() for nid in range(rna.shape[0])]

    G = newick2digraph(tree)
    G = annotate(G, cell_arr_adata, metadata, ttp=ttp)
    for s, t in G.edges:
        sn, tn = G.nodes[s], G.nodes[t]
        assert is_valid_edge(sn, tn), (s, t)

    trees = {"early": cut_at_depth(G, max_depth=depth), "late": cut_at_depth(G)}
    rna_arrays = {
        kind: np.asarray(
            [
                trees[kind].nodes[n]["cell"].x
                for n in trees[kind].nodes
                if is_leaf(trees[kind], n)
            ]
        )
        for kind in ["early", "late"]
    }
    data = process_data(rna_arrays, n_pcs=n_pcs)

    n_early_leaves = len([n for n in trees["early"] if is_leaf(trees["early"], n)])
    data_early, data_late = data[:n_early_leaves], data[n_early_leaves:]

    for kind, data in zip(["early", "late"], [data_early, data_late]):
        i, G = 0, trees[kind]
        for n in G.nodes:
            if is_leaf(G, n):
                G.nodes[n]["cell"].x = data[i]
                i += 1
            else:
                G.nodes[n]["cell"].x = np.full((n_pcs,), np.nan)

    return trees


def prepare_data(
    adata: AnnData,
    depth: int,
    ssr: Optional[float] = None,
    n_pcs: int = 30,
    ttp: float = 100.0,
) -> Tuple[
    csr_matrix,
    Dict[Literal["early", "late"], nx.DiGraph],
    Dict[Literal["early", "late"], np.ndarray],
]:
    tree = adata.uns["tree"]
    rna = adata.X.A.copy() if issparse(adata.X) else adata.X.copy()
    barcodes = adata.obsm["barcodes"].copy()

    if ssr is not None:
        barcodes = stochastic_silencing(barcodes, stochastic_silencing_rate=ssr)
    true_trees = build_true_trees(
        rna, barcodes, meta=adata.obs, tree=tree, depth=depth, n_pcs=n_pcs, ttp=ttp
    )
    
    if ssr is not None:
        barcodes = stochastic_silencing(barcodes, stochastic_silencing_rate=ssr)
    true_trees = build_true_trees(
        rna, barcodes, meta=adata.obs, tree=tree, depth=depth, n_pcs=n_pcs, ttp=ttp
    )
    data_arrays = {
        "late": lot_inf.extract_data_arrays(true_trees["late"]),
        "early": lot_inf.extract_data_arrays(true_trees["early"]),
    }
    rna_arrays = {
        "early": data_arrays["early"][0],
        "late": data_arrays["late"][0],
    }
    
    barcode_arrays = {
        "early": data_arrays["early"][1],
        "late": data_arrays["late"][1],
    }

    return get_true_coupling(true_trees["early"], true_trees["late"]).tocsr(), rna_arrays, barcode_arrays
    
def get_leaf_descendants(tree, node):
    """
    Returns a list of the leaf nodes of the tree that are
    descendants of node
    """
    if tree.out_degree(node) == 0:
        return [node]
    else:
        children = tree.successors(node)
        leaf_descendants = []
        for child in children:
            leaf_descendants = leaf_descendants + get_leaf_descendants(tree, child)
        return leaf_descendants
    return

def get_true_coupling(early_tree, late_tree):
    """
    Returns the coupling between leaves of early_tree and their descendants in
    late_tree. Assumes that early_tree is a truncated version of late_tree
    The marginal over the early cells is uniform; if cells have different
    numbers of descendants, the marginal over late cells will not be uniform.
    """
    num_cells_early = len(lot_inf.get_leaves(early_tree)) - 1
    num_cells_late = len(lot_inf.get_leaves(late_tree)) - 1
    
    coupling = dok_matrix((num_cells_early, num_cells_late))
    
    cells_early = lot_inf.get_leaves(early_tree, include_root = False)
    
    
    for cell in cells_early:
        parent = next(early_tree.predecessors(cell))
        late_tree_cell = None
        for child in late_tree.successors(parent):
            if late_tree.nodes[child]['cell'].seed == early_tree.nodes[cell]['cell'].seed:
                late_tree_cell = child
                break
        if late_tree_cell == None:
            raise ValueError("A leaf in early_tree does not appear in late_tree. Cannot find coupling." +
                             "\nCheck whether either tree has been modified since truncating.")
        descendants = get_leaf_descendants(late_tree, late_tree_cell)
        coupling[cell, descendants] = 1/(num_cells_early*len(descendants))
    
    return coupling