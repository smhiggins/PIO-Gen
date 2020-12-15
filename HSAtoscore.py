import sys
import re
import numpy as np
from sklearn import metrics
from math import sqrt
from scipy import stats

if len(sys.argv) < 2:
    print("Usage: python pdbtoscore.py <input_path>")
    sys.exit()
else:
    input_path = sys.argv[1]


def euclidean_distance(vector_one, vector_two):
    return sqrt(pow(vector_one[0] - vector_two[0], 2) + pow(vector_one[1] - vector_two[1], 2) +
                     pow(vector_one[2] - vector_two[2], 2))


def rmse(true_matrix, prediction_matrix):
    return metrics.mean_squared_error(true_matrix, prediction_matrix)

length = 100
with open(input_path, 'r') as f:
    input_data = [[float(num) for num in line.split('\t')[2:]] for line in f ]
np_input = np.asarray(input_data)
with open('Data/regularstructre.txt', 'r') as f:
    true_struct = [[float(num) for num in line.split(',')] for line in f]
true_struct = np.asarray(true_struct)
true_distance_map = np.zeros([length, length])
for i in range(length):
    for j in range(i, length):
        true_distance_map[j][i] = true_distance_map[i][j] = \
            euclidean_distance(true_struct[i], true_struct[j])


scale = []
scale.append(abs((max(np_input[:,0]) -min(np_input[:,0]))/(max(true_struct[:,0]) - min(true_struct[:,0]))))
scale.append(abs((max(np_input[:,1]) -min(np_input[:,1]))/(max(true_struct[:,1]) - min(true_struct[:,1]))))
scale.append(abs((max(np_input[:,2]) -min(np_input[:,2]))/(max(true_struct[:,2]) - min(true_struct[:,2]))))
scale = np.asarray(scale)

np_input = np_input/scale
center = true_struct[0] - np_input[0]
np_input = np_input + center
local_distance_map = np.zeros([length, length])
for i in range(length):
    for j in range(i, length):
        local_distance_map[j][i] = local_distance_map[i][j] = \
            euclidean_distance(np_input[i], np_input[j])


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
rmse_sum = rmse(np_input, true_struct)

print("Spearman: " + str(sp_sum))
print("Pearson: " + str(pr_sum))
print("RMSE: " + str(rmse_sum))

with open(input_path[:-4] + ".log", 'w', newline='') as f:
    f.write("Input: {}\n".format(input_path))
    f.write("Spearman: True Distance Map vs Predicted Distance Map: {}\n".format(sp_sum))
    f.write("Pearson: True Distance Map vs Predicted Distance Map: {}\n".format(pr_sum))
    f.write("RMSE: True Distance Map vs Predicted Distance Map: {}\n".format(rmse_sum))
