These are the numpy dictionaries where total runtime and memory consumption every 0.01s is stored for the benchmark on the mouse atlas.

The files names are as {method}_{cell number in source and target in multiples of 1,000}_{CPU or GPU}.npy

method is one of the following:
  - WOT: Waddington OT, see Schiebinger et al., Optimal-Transport Analysis of Single-Cell Gene Expression Identifies Developmental Trajectories in Reprogramming, Cell, 2019
  - offline: moscot in offline mode, i.e. batch_size higher than number of cells
  - online: moscot in online_mode, where batch_size=1,000
  - LR_{rank}: moscot in low-rank mode where rank in {50, 200, 500, 2000}
