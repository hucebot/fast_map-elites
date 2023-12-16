#-- Necessary Imports
import numpy as np
from os.path import exists
import os, sys
from tqdm import tqdm, trange
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
from glob import glob
from matplotlib  import cm

import json

from ribs.archives import GridArchive
from ribs.emitters import EvolutionStrategyEmitter
from ribs.schedulers import Scheduler
from ribs.visualize import grid_archive_heatmap

sys.path.append('/Users/jmouret/git/resibots/fast_map-elites/build')
import pf_mapelites

DIM = 365

max_bound = 1

archive = GridArchive(solution_dim=DIM,
                      dims=(64, 64),
                      ranges=[(0, 1), (0, 1)],
                      learning_rate=0.01,
                      threshold_min=0.0)
result_archive = GridArchive(solution_dim=DIM,
                             dims=(64, 64),
                             ranges=[(0, 1), (0, 1)])


emitters = [
    EvolutionStrategyEmitter(
        archive,
        x0=np.ones(DIM) * 0.5,
        sigma0=0.5,
        ranker="imp",
        selection_rule="mu",
        restart_rule="basic",
        batch_size=36,
    ) for _ in range(15)
]

scheduler = Scheduler(archive, emitters, result_archive=result_archive)



def save_heatmap(archive, heatmap_path):
    plt.figure(figsize=(8, 6))
    grid_archive_heatmap(archive, vmin=0, vmax=100)
    plt.tight_layout()
    plt.savefig(heatmap_path)
    plt.close(plt.gcf())

def map_elites(solution_batch):
    """
    Args:
        solution_batch (np.ndarray): (batch_size, dim) batch of solutions.
    Returns:
        objective_batch (np.ndarray): (batch_size,) batch of objectives.
        measures_batch (np.ndarray): (batch_size, 2) batch of measures.
    """
    objective_batch = np.zeros((solution_batch.shape[0],))
    measures_batch = np.zeros((solution_batch.shape[0], 2))
    pf_mapelites.fit_meta(solution_batch, objective_batch, measures_batch)
    return objective_batch, measures_batch


metrics = {
    "QD Score": {
        "x": [0],
        "y": [0.0],
    },
    "Archive Coverage": {
        "x": [0],
        "y": [0.0],
    },
}

total_itrs = int(100_000 / (36 * len(emitters)))
log_freq=10
outdir = 'cma_me/'
outdir = Path(outdir)
if not outdir.is_dir():
    outdir.mkdir()

for itr in trange(1, total_itrs + 1, file=sys.stdout, desc='Iterations'):
    solution_batch = scheduler.ask()
    objective_batch, measure_batch = map_elites(solution_batch)
    scheduler.tell(objective_batch, measure_batch)

    # Output progress 
    tqdm.write(f"Iteration {itr:5d} | "
                   f"Archive Coverage: {result_archive.stats.coverage * 100:6.3f}%  "
                   f"Normalized QD Score: {result_archive.stats.norm_qd_score:6.3f}")
    
    # Logging and output.
    if itr == total_itrs:
        result_archive.data(return_type="pandas").to_csv(
            outdir / f"cma_me_archive.csv")

    # Record and display metrics.
    metrics["QD Score"]["x"].append(itr)
    metrics["QD Score"]["y"].append(result_archive.stats.norm_qd_score)
    metrics["Archive Coverage"]["x"].append(itr)
    metrics["Archive Coverage"]["y"].append(
        result_archive.stats.coverage)
    
    save_heatmap(result_archive,
                    str(outdir / f"cma_me_heatmap_{itr:05d}.png"))

# Plot metrics.
for metric, values in metrics.items():
    plt.plot(values["x"], values["y"])
    plt.title(metric)
    plt.xlabel("Iteration")
    plt.savefig(
        str(outdir / f"cma_me_{metric.lower().replace(' ', '_')}.png"))
    plt.clf()
with (outdir / f"cma_me_metrics.json").open("w") as file:
    json.dump(metrics, file, indent=2)