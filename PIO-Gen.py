import sys
from functools import partial
from math import ceil, sqrt
import numpy as np
from scipy import stats
from workers import Pigeon
from multiprocessing import Pool
from sklearn import metrics
import pandas as pd
import os


if len(sys.argv) < 5:
    print("Usage: python PIO-Gen.py <input_path> <output_path> <conversion factor> <cluster size>")
    sys.exit()
else:
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    alpha = float(sys.argv[3])
    cluster_size = int(sys.argv[4])

num_workers = 200
search_phase_epochs = 50
homing_phase_epochs = 50
cycles = 5


# extract dense matrix from tab delimited square matrix
with open(input_path, 'r') as f:
    contact_matrix = np.genfromtxt(f, delimiter="\t")

# delete last column if nan
if np.isnan(contact_matrix[0][len(contact_matrix[0]) - 1]):
    contact_matrix = np.delete(contact_matrix, len(contact_matrix[0]) - 1, 1)


def wrapper(distance, best, epoch, pigeon):
    return pigeon.search_phase(distance, best, epoch)


def homing_wrapper(distance, best, epoch, pigeon):
    return pigeon.homing_phase(distance, best, epoch)


def euclidean_distance(vector_one, vector_two):
    return sqrt(pow(vector_one[0] - vector_two[0], 2) + pow(vector_one[1] - vector_two[1], 2) +
                     pow(vector_one[2] - vector_two[2], 2))


def rmse(true_matrix, prediction_matrix):
    return metrics.mean_squared_error(true_matrix, prediction_matrix)


if __name__ == '__main__':
    print("Beginning on alpha. " + str(alpha))
    distance_map = np.power(contact_matrix, -alpha)
    distance_map[distance_map == np.inf] = np.nan
    workers = []
    length = len(distance_map)

    for i in range(num_workers):
        workers.append(Pigeon(length, cluster_size))

    with Pool(os.cpu_count()) as p:
        for cycle in range(cycles):

            # initialize search variables
            search_score = np.zeros(ceil(length / cluster_size))
            best_global = np.zeros([len(distance_map), 3])
            homing_score = 0
            anneal_score = 0

# ----------------------------------------------------------------------------------------------------------------------
# Search phase looks for the best 3D structure for each cluster
# ----------------------------------------------------------------------------------------------------------------------
            for i in range(0, search_phase_epochs):
                partial_search = partial(wrapper, distance_map, best_global, i)
                results = np.asarray(p.map(partial_search, workers))
                for j in range(num_workers):
                    iteration = 0
                    for k in range(0, length, cluster_size):
                        if results[j][0][iteration] > search_score[iteration]:
                            search_score[iteration] = results[j][0][iteration]
                            best_global[k:k+cluster_size] = results[j][1][k:k+cluster_size]
                        iteration += 1
                for j in range(num_workers):
                    workers[j].spearman_score_1 = results[j][0]
                    workers[j].best_local = results[j][1]
                    workers[j].velocities = results[j][2]
                    workers[j].distances = results[j][3]
                print("Search Phase: " + str(i+1) + " of " + str(search_phase_epochs)
                      + " in " + str(cycle+1) + " of " + str(cycles))
                print(search_score)
            for j in range(num_workers):
                workers[j].velocities = np.random.randn(length, 3)

# ----------------------------------------------------------------------------------------------------------------------
# Homing phase moves clusters to find their optimal location
# ----------------------------------------------------------------------------------------------------------------------

            for i in range(0, homing_phase_epochs):
                partial_home = partial(homing_wrapper, distance_map, best_global, i)
                results = np.asarray(p.map(partial_home, workers))
                for j in range(num_workers):
                    if results[j][0] > homing_score:
                        homing_score = results[j][0]
                        best_global = results[j][1]
                    workers[j].spearman_score_2 = results[j][0]
                    workers[j].best_local = results[j][1]
                    workers[j].cluster_velocities = results[j][2]
                    workers[j].distances = results[j][3]
                print("Homing Phase: " + str(i+1) + " of " + str(homing_phase_epochs)
                      + " in " + str(cycle+1) + " of " + str(cycles))
                print(homing_score)
            for j in range(num_workers):
                workers[j].cluster_velocities = np.random.randn(ceil(length/cluster_size), 3)

# ----------------------------------------------------------------------------------------------------------------------
# Save results
# ----------------------------------------------------------------------------------------------------------------------
    local_distance_map = np.zeros([length, length])
    for i in range(length):
        for j in range(i, length):
            local_distance_map[j][i] = local_distance_map[i][j] = \
                euclidean_distance(best_global[i], best_global[j])
    with open('Data/regularstructre.txt', 'r') as f:
        true_struct = [[float(num) for num in line.split(',')] for line in f]
    true_struct = np.asarray(true_struct)
    true_distance_map = np.zeros([length, length])
    for i in range(length):
        for j in range(i, length):
            true_distance_map[j][i] = true_distance_map[i][j] = \
                euclidean_distance(true_struct[i], true_struct[j])
    # find average spearman score by row
    sp_score = stats.spearmanr(true_distance_map, local_distance_map, axis=1)[0]
    sp_sum = 0
    for i in range(100):
        sp_sum += sp_score[i][i+100]
    sp_sum = sp_sum/100
    # find average pearson score by row
    pr_score = np.corrcoef(true_distance_map, local_distance_map)
    pr_sum = 0
    for i in range(100):
        pr_sum += pr_score[i][i + 100]
    pr_sum = pr_sum / 100
    # find RMSE
    rmse_sum = rmse(best_global, true_struct)

    print("Spearman: " + str(sp_sum))
    print("Pearson: " + str(pr_sum))
    print("RMSE: " + str(rmse_sum))
    '''
    print("Spearman: Calculated Distance Map vs Predicted Distance Map")
    print(map_sp_score[0])
    print("Pearson: Calculated Distance Map vs Predicted Distance Map")
    print(map_p_score[0])
    print("RMSE: Calculated Distance Map vs Predicted Distance Map")
    print(map_r_score)
    '''
    with open(output_path + ".log", 'w', newline='') as f:
        f.write("3D CHROMOSOME MODELING BY PIOGEN\n")
        f.write("Conversion factor: {}\n".format(alpha))
        f.write("Spearman: True Distance Map vs Predicted Distance Map: {}\n".format(sp_sum))
        f.write("Pearson: True Distance Map vs Predicted Distance Map: {}\n".format(pr_sum))
        f.write("RMSE: True Distance Map vs Predicted Distance Map: {}\n".format(rmse_sum))

    with open(output_path + ".pdb", 'w', newline='') as f:
        f.write("3D CHROMOSOME MODELING BY PIOGEN\n")
        for i in range(1, len(best_global) + 1):
            f.write('ATOM{:>7}   CA MET A{:<3}{:>13.{prec}f}{:>8.{prec}f}{:>8.{prec}f}  0.20 10.00\n'.format(
                i, i, best_global[i - 1][0], best_global[i - 1][1], best_global[i - 1][2], prec=3))
        for i in range(1, len(best_global)):
            f.write('CONECT{:>5}{:>5}\n'.format(i, i + 1))
