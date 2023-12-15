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

# parallel tqdm
from tqdm.contrib.concurrent import process_map


sys.path.append('/Users/jmouret/git/resibots/fast_map-elites/build')
import pf_mapelites


# some parameters
DIM = 5
max_bound = 1
batch_size = 36
total_itrs = int(100_000 / batch_size)
num_replicates = 5


# load the files
centroids = np.loadtxt(sys.argv[1])
data = np.loadtxt(sys.argv[2])
data_fit = np.loadtxt(sys.argv[3])

# final result
map_me_fit = np.zeros((centroids.shape[0]))

# wrapper
def fit_gp(fit_params, solution_batch):
    """
    Args:
        solution_batch (np.ndarray): (batch_size, dim) batch of solutions.
    Returns:
        objective_batch (np.ndarray): (batch_size,) batch of objectives.
        measures_batch (np.ndarray): (batch_size, 2) batch of measures.
    """
    objective_batch = np.zeros((solution_batch.shape[0],))
    measures_batch = np.zeros((solution_batch.shape[0], 2))
    pf_mapelites.fit_gp(fit_params, solution_batch, objective_batch, measures_batch)
    return objective_batch, measures_batch

# to be run in parallel
def run_cma_me(n_elite):
    if data_fit[n_elite] < -1e10:
        return -1e10 # skip
    qd = []
    for n in range(num_replicates):
        #### instantiate
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
                batch_size=batch_size,
            ) for _ in range(15)
        ]

        scheduler = Scheduler(archive, emitters, result_archive=result_archive)

        #### main loop
        for itr in range(1, total_itrs + 1):#, file=sys.stdout, desc='Iterations'):
            solution_batch = scheduler.ask()
            objective_batch, measure_batch = fit_gp(data[n_elite], solution_batch)
            scheduler.tell(objective_batch, measure_batch)

        qd += [result_archive.stats.norm_qd_score]
        tqdm.write(f"{n_elite}/{n}: {result_archive.stats.norm_qd_score:6.3f}")
    return np.median(qd)

# run  me-elites on each function
# for elite in range(data.shape[0]):
#     print("-- #elite:", elite, " / ", centroids.shape[0], " ME:", data_fit[elite])
#     if (data_fit[elite] < -1e10):
#         continue
if __name__ == '__main__':
    args = range(0, centroids.shape[0])
    result = process_map(run_cma_me, args, max_workers=4, chunksize=1)

    np.savetxt("map_me_fit.dat", result)


#         for itr in trange(1, total_itrs + 1, file=sys.stdout, desc='Iterations'):

#  # Output progress every 500 iterations or on the final iteration.
#             if itr % 500 == 0 or itr == total_itrs:
#                 tqdm.write(f"Iteration {itr:5d} | "
#                             f"Archive Coverage: {result_archive.stats.coverage * 100:6.3f}%  "
#                             f"Normalized QD Score: {result_archive.stats.norm_qd_score:6.3f}")