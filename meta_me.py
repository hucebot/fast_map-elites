#-- Necessary Imports
import numpy as np
from os.path import exists
import os, sys
import subprocess
from tqdm import tqdm, trange

import matplotlib.pyplot as plt
import matplotlib as mpl
from glob import glob
from matplotlib  import cm

from ribs.archives import GridArchive
from ribs.emitters import EvolutionStrategyEmitter
from ribs.schedulers import Scheduler

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
        x0=np.ones(DIM) / 0.5,
        sigma0=0.5,
        ranker="imp",
        selection_rule="mu",
        restart_rule="basic",
        batch_size=36,
    ) for _ in range(15)
]

scheduler = Scheduler(archive, emitters, result_archive=result_archive)
    

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
    pf_mapelites.fit(solution_batch, objective_batch, measures_batch)
    return objective_batch, measures_batch


total_itrs = 10_000
for itr in trange(1, total_itrs + 1, file=sys.stdout, desc='Iterations'):
    print(itr)
    solution_batch = scheduler.ask()
    print('ask ok:', solution_batch.shape)
    objective_batch, measure_batch = map_elites(solution_batch)
    scheduler.tell(objective_batch, measure_batch)

    # Output progress every 500 iterations or on the final iteration.
    #if itr % 500 == 0 or itr == total_itrs:
    tqdm.write(f"Iteration {itr:5d} | "
                   f"Archive Coverage: {result_archive.stats.coverage * 100:6.3f}%  "
                   f"Normalized QD Score: {result_archive.stats.norm_qd_score:6.3f}")