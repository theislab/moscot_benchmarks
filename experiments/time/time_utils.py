from typing import Tuple, Union, Optional

from scipy.sparse import issparse, csr_matrix, dok_matrix
from moscot.backends.ott import LinearOutput, LRLinearOutput
from moscot.problems.base import BirthDeathProblem

from ott.core import sinkhorn
from ott.geometry import pointcloud
import numpy as np
import jax.numpy as jnp
import numpy.typing as npt

Output = Union[LinearOutput, LRLinearOutput]


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
    random: bool,
) -> Tuple[npt.ArrayLike, ...]:
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
    n_samples: Optional[int] = None,
) -> float:
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


def get_leaf_descendants(tree, node):
    """
    copied from https://github.com/aforr/LineageOT/blob/8c66c630d61da289daa80e29061e888b1331a05a/lineageot/inference.py#L657

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
    adapted from https://github.com/aforr/LineageOT/blob/8c66c630d61da289daa80e29061e888b1331a05a/lineageot/inference.py#L657

    Returns the coupling between leaves of early_tree and their descendants in
    late_tree. Assumes that early_tree is a truncated version of late_tree
    The marginal over the early cells is uniform; if cells have different
    numbers of descendants, the marginal over late cells will not be uniform.
    """
    num_cells_early = len(get_leaves(early_tree)) - 1
    num_cells_late = len(get_leaves(late_tree)) - 1
    
    coupling = dok_matrix((num_cells_early, num_cells_late))
    
    cells_early = get_leaves(early_tree, include_root = False)
    
    
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
    fpath: Path,
    *,
    depth: int,
    ssr: Optional[float] = None,
    n_pcs: int = 30,
    ttp: float = 100.0,
) -> Tuple[
    csr_matrix,
    Dict[Literal["early", "late"], np.ndarray],
]:
    adata = sc.read(fpath)
    tree = adata.uns["tree"]
    rna = adata.X.A.copy() if issparse(adata.X) else adata.X.copy()
    barcodes = adata.obsm["barcodes"].copy()

    if ssr is not None:
        barcodes = stochastic_silencing(barcodes, stochastic_silencing_rate=ssr)
    true_trees = build_true_trees(
        rna, barcodes, meta=adata.obs, tree=tree, depth=depth, n_pcs=n_pcs, ttp=ttp
    )
    data_arrays = {
        "late": lot_inf.extract_data_arrays(true_trees["late"]),
        "early": lot_inf.extract_data_arrays(true_trees["early"]),
    }

    true_coupling = get_true_coupling(true_trees["early"], true_trees["late"])

    rna_arrays = {
        "early": data_arrays["early"][0],
        "late": data_arrays["late"][0],
    }

    return true_coupling, rna_arrays
