# coding: utf-8
from os import getpid

from TimeSeries import TimeSeries
from Patterns import Templates
import numpy as np
from sklearn.cluster import DBSCAN
from WishartClusterizationAlgorithm import Wishart


def calc_distance_matrix(test_vectors, train_vectors):
    # Как будто бы надо попробовать переписать эту функцию под gpu и на понятный язык.
    shape = (test_vectors.shape[0], train_vectors.shape[1])
    distance_matrix = np.zeros(shape)
    for i in range(test_vectors.shape[1]):
        distance_matrix += (train_vectors[:, :, i] - np.repeat(test_vectors[:, i], train_vectors.shape[1]).reshape(-1,train_vectors.shape[1])) ** 2
    distance_matrix **= 0.5
    return distance_matrix


def count_elements_sorted(arr, elements):
    """
    Возвращает массив вхождений элементов из `elements` в `arr`.
    Предполагается, что `elements` отсортирован.
    """
    count_dict = {}
    for item in arr:
        if item in count_dict:
            count_dict[item] += 1
        else:
            count_dict[item] = 1

    # Возвращаем список вхождений для каждого элемента из `elements`
    return [count_dict.get(element, 0) for element in elements]


class TSProcessor:
    def __init__(self, time_series_list, window_index, template_length,
                 max_template_spread,
                 test_size, k=16, mu=0.45,threshold = 0.8):
        self.time_series_ = time_series_list[0]
        self.time_series_.split_train_val_test(window_index, test_size)
        self.templates_ = Templates(template_length, max_template_spread)
        self.templates_.create_train_set(time_series_list[1:])
        self.ts_number = len(time_series_list[1:])
        self.k, self.mu = k, mu
        self.threshold = threshold
    def pull(self, eps):
        steps = len(self.time_series_.test)
        values = self.time_series_.train
        values += self.time_series_.val
        size_of_series = len(values)
        values = np.array(values)
        values.resize(size_of_series + steps)
        values[-steps:] = np.nan
        n_trajectories = 1
        forecast_trajectories = np.full((steps, n_trajectories), np.nan)
        x_dim, y_dim, z_dim = self.templates_.train_set.shape
        vectors_continuation = np.full([x_dim, steps, z_dim], fill_value=np.inf)
        affiliation_continuation = np.full([x_dim, steps, z_dim], fill_value=0)
        train_vectors = np.hstack([self.templates_.train_set, vectors_continuation])
        # Мы создаем параллельно с обучающими векторами вектора,
        # которые имеют ту же форму и заполнены числами, равными номеру ряда,
        # по которому был получен обучающий вектор.
        # По ним можно определить принадлежность тому или иному ряду.
        affiliation_vectors = np.hstack([self.templates_.affiliation_matrix, affiliation_continuation])
        observation_indexes = self.templates_.observation_indexes
        affiliation_result = []

        prev_pred = np.full((steps, 3), np.nan)
        prev_size = np.full((steps, 3), np.nan)
        prev_spread = np.full((steps, 3), np.nan)
        for trajectory in range(n_trajectories):
            for step in range(steps):
                test_vectors = values[:size_of_series + step][observation_indexes]
                distance_matrix = calc_distance_matrix(test_vectors, train_vectors)
                distance_mask = distance_matrix < eps
                points = train_vectors[distance_mask][:, -1]
                affiliation_indexes = affiliation_vectors[distance_mask][:, -1]
                forecast_point, affiliation_step_result = self.freeze_point(points[~np.isnan(points)], 'cl', affiliation_indexes[~np.isnan(points)],self.threshold)
                size = np.sum(affiliation_step_result)
                # отсев по росту размера кластеров
                prev_size[step][2] = size
                if step >= 1:
                    prev_size[step][1] = np.copy(prev_size[step - 1][2])
                if step >= 2:
                    prev_size[step][0] = np.copy(prev_size[step - 1][1])
                    if prev_size[step][0] < prev_size[step][1] < prev_size[step][2]:
                        forecast_point = np.nan
                # # отсев по росту разброса
                # prev_spread[step][2] = np.max(points)-np.min(points) if points.shape[0] > 0 else 0
                # if step >= 1:
                #     prev_spread[step][1] = np.copy(prev_spread[step - 1][2])
                # if step >= 2:
                #     prev_spread[step][0] = np.copy(prev_spread[step - 1][1])
                #     if prev_spread[step][0] < prev_spread[step][1] < prev_spread[step][2]:
                #         forecast_point = np.nan
                affiliation_result.append(affiliation_step_result)
                forecast_trajectories[step, trajectory] = forecast_point
                values[size_of_series + step] = forecast_point

        changed_aff = np.array(affiliation_result)
        if len(changed_aff) == 0:
            return forecast_trajectories, values, np.full(self.ts_number, 0)
        return forecast_trajectories, values, changed_aff[-1]

    def freeze_point(self, points_pool, how, affiliation_indexes,threshold):
        result = None
        affiliation_result = np.full(self.ts_number,0)

        if points_pool.size == 0:
            result = np.nan
            return result,affiliation_result
        if how == 'mean':
            result = float(points_pool.mean())

        elif how == 'mf':
            points, counts = np.unique(points_pool, return_counts=True)
            result = points[counts.argmax()]
        elif how == 'cl':
            dbs = DBSCAN(0.01,min_samples=17)
            dbs.fit(points_pool.reshape(-1, 1))
            # print(np.unique(points_pool,return_counts=True))
            cluster_labels, cluster_sizes = np.unique(dbs.labels_[dbs.labels_ > -1], return_counts=True)
            # print(cluster_sizes.argmax() / sum(cluster_sizes))
            if cluster_labels.size > 0:
                biggest_cluster_center = points_pool[dbs.labels_ == cluster_labels[cluster_sizes.argmax()]].mean()

                affiliation_result = count_elements_sorted(
                    affiliation_indexes[dbs.labels_ == cluster_labels[cluster_sizes.argmax()]],
                    range(self.ts_number))
                result = biggest_cluster_center
            else:
                result = np.nan
        return result, affiliation_result
