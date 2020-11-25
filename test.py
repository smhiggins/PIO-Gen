from math import sqrt
import numpy as np
from scipy import stats


def euclidean_distance(vector_one, vector_two):
    return sqrt(pow(vector_one[0] - vector_two[0], 2) + pow(vector_one[1] - vector_two[1], 2) +
                     pow(vector_one[2] - vector_two[2], 2))


with open('regularstructre.txt', 'r') as f:
    best_global = [[float(num) for num in line.split(',')] for line in f]
with open('regular90.txt', 'r') as f:
    contact_matrix = np.genfromtxt(f, delimiter="\t")
'''
with open("true.pdb", 'w', newline='') as f:
    f.write("3D CHROMOSOME MODELING BY PIOGEN\n")
    for i in range(1, len(best_global) + 1):
        f.write('ATOM{:>7}   CA MET A{:<3}{:>13.{prec}f}{:>8.{prec}f}{:>8.{prec}f}  0.20 10.00\n'.format(
            i, i, best_global[i - 1][0], best_global[i - 1][1], best_global[i - 1][2], prec=3))
    for i in range(1, len(best_global)):
        f.write('CONECT{:>5}{:>5}\n'.format(i, i + 1))
'''
for alpha in np.arange(.1, 3, .1):
    distance_map = np.power(contact_matrix, -alpha)
    distance_map[distance_map == np.inf] = np.nan
    length = len(contact_matrix)
    local_distance_map = np.zeros([length, length])
    for i in range(length):
        for j in range(i, length):
            local_distance_map[j][i] = local_distance_map[i][j] = \
                euclidean_distance(best_global[i], best_global[j])
    nas = np.logical_or(np.isnan(distance_map), np.isnan(local_distance_map))
    score = stats.pearsonr(distance_map[~nas], local_distance_map[~nas])
    print(str(alpha)+" score: ")
    print(score)