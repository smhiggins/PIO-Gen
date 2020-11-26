import numpy as np
import random
import math
from math import sqrt, ceil , exp
from scipy import stats
from sklearn import metrics


def euclidean_distance(vector_one, vector_two):
    return sqrt(pow(vector_one[0] - vector_two[0], 2) + pow(vector_one[1] - vector_two[1], 2) +
                     pow(vector_one[2] - vector_two[2], 2))


def rmse(true_matrix, prediction_matrix):
    return metrics.mean_squared_error(true_matrix, prediction_matrix)


class Pigeon:
    def __init__(self, length, cluster_size):
        self.length = length
        self.cluster_size = cluster_size
        self.velocities = np.random.randn(length, 3)
        self.cluster_velocities = np.random.randn(ceil(length/cluster_size), 3)
        self.distances = np.zeros([length, 3])
        for i in range(1, length):
            if i % cluster_size and i != 0:
                spread = .1
            else:
                spread = .1
            self.distances[i][0] = self.distances[i-1][0] + spread
            #self.distances[i][1] = random.random()
            #self.distances[i][2] = random.random()
        self.max_distance = .4
        self.map_factor = .01
        self.rnd = random.random()
        self.best_local = self.distances
        self.spearman_score_1 = np.zeros(ceil(length/self.cluster_size))
        self.spearman_score_2 = 0
        self.spearman_score_3 = 0

    def search_phase(self, distance_map, best_global, epoch):

        prev_velocities = self.velocities
        prev_distances = self.distances
        local_distance_map = np.zeros([self.length, self.length])
        spring = .1

        if epoch == 0:
            best_global = self.best_local
        for i in range(self.length):
            corrective_vel = np.zeros(3)
            if i > 0:
                temp = euclidean_distance(prev_distances[i-1], prev_distances[i])
                if temp > self.max_distance:
                    corrective_vel = (self.distances[i - 1] - self .distances[i]) * spring
            for j in range(3):
                self.velocities[i][j] = prev_velocities[i][j] * exp(-epoch * self.map_factor) + self.rnd * \
                                        ((self.best_local[i][j] + best_global[i][j])/2 - self.distances[i][j]) + corrective_vel[j]
                self.distances[i][j] = prev_distances[i][j] + self.velocities[i][j]

        for i in range(self.length):
            for j in range(i, self.length):
                local_distance_map[j][i] = local_distance_map[i][j] = \
                    euclidean_distance(self.distances[i], self.distances[j])

        nas = np.logical_or(np.isnan(distance_map), np.isnan(local_distance_map))
        cluster = 0

        for i in range(0, self.length, self.cluster_size):
            local_temp = local_distance_map[i:i + self.cluster_size]
            local_dist = distance_map[i:i + self.cluster_size]
            local_nas = nas[i:i + self.cluster_size]
            #score = stats.spearmanr(local_dist[~local_nas], local_temp[~local_nas])
            score = stats.pearsonr(local_dist[~local_nas], local_temp[~local_nas])
            if score[0] >= self.spearman_score_1[cluster]:
                self.spearman_score_1[cluster] = score[0]
                self.best_local[i:i + self.cluster_size] = self.distances[i:i + self.cluster_size]
            cluster += 1
            score_return = self.spearman_score_1

        return score_return, self.best_local, self.velocities, self.distances

    def homing_phase(self, distance_map, best_global, epoch):
        prev_cluster_velocities = self.cluster_velocities
        prev_distances = self.distances
        local_distance_map = np.zeros([self.length, self.length])
        if epoch == 0:
            best_global = self.best_local
        for i in range(0, self.length, self.cluster_size):
            sum_cluster = self.distances[i:i + self.cluster_size].sum(axis=0)/self.cluster_size
            sum_best = best_global[i:i + self.cluster_size].sum(axis=0)/self.cluster_size
            true_ind = ceil(i/self.cluster_size)
            for j in range(3):
                self.cluster_velocities[true_ind][j] = prev_cluster_velocities[true_ind][j] * math.exp(-epoch * self.map_factor) + self.rnd * \
                                        (sum_best[j] - sum_cluster[j])
            for k in range(i, i+self.cluster_size):
                for j in range(3):
                    self.distances[k][j] = prev_distances[k][j] + self.cluster_velocities[true_ind][j]

        for i in range(self.length):
            for j in range(i, self.length):
                local_distance_map[j][i] = local_distance_map[i][j] = \
                    euclidean_distance(self.distances[i], self.distances[j])

        nas = np.logical_or(np.isnan(distance_map), np.isnan(local_distance_map))
        #score = stats.spearmanr(distance_map[~nas], local_distance_map[~nas])
        score = stats.pearsonr(distance_map[~nas], local_distance_map[~nas])
        if score[0] >= self.spearman_score_2:
            self.spearman_score_2 = score[0]
            self.best_local = self.distances

        return self.spearman_score_2, self.best_local, self.cluster_velocities, self.distances
