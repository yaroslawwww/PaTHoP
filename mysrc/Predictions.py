import copy
import random

import pandas as pd
from dask.array import shape
from matplotlib import pyplot as plt
from numpy.ma.extras import average
from statsmodels.tsa.vector_ar.var_model import forecast
from tqdm import tqdm

from TimeSeries import TimeSeries
from Patterns import Templates
import numpy as np

from mysrc.WishartClusterizationAlgorithm import Wishart


def calc_distance_matrix(test_vectors: np.ndarray, train_vectors: np.ndarray, steps: int,
                         y_dim: int) -> np.ndarray:
    shape = (test_vectors.shape[0], train_vectors.shape[1])
    distance_matrix = np.zeros(shape)
    for i in range(test_vectors.shape[1]):
        distance_matrix += (train_vectors[:, :, i] - np.repeat(test_vectors[:, i], y_dim + steps).reshape(-1,
                                                                                                          y_dim + steps)) ** 2
    distance_matrix **= 0.5
    return distance_matrix


class TSProcessor:
    def __init__(self, time_series: TimeSeries, template_length: int = 4, max_template_spread: int = 10,
                 train_size: int = 5000, val_size: int = 0, test_size: int = 50, k: int = 5, mu: float = 0.1):
        self.time_series_ = time_series
        self.time_series_.split_train_val_test(train_size, val_size, test_size)
        self.templates_ = Templates(template_length, max_template_spread)
        self.templates_.create_train_set(time_series)
        self.k, self.mu = k, mu
        self.eps = 0.0056

    def validation(self):
        eps = self.eps
        steps = len(self.time_series_.val)
        values = self.time_series_.train
        size_of_series = len(values)
        values += self.time_series_.val
        values = np.array(values)
        x_dim, y_dim, z_dim = self.templates_.train_set.shape
        vectors_continuation = np.full([x_dim, steps, z_dim], fill_value=np.inf)
        train_vectors = np.hstack([self.templates_.train_set, vectors_continuation])
        observation_indexes = self.templates_.observation_indexes  # отрицательные номера элементов которые нужны для шаблона
        temperature = 0.8
        param = 4
        average_points_size = 0
        while average_points_size > 100 or average_points_size < 10:
            average_points_size = 0
            param *= temperature
            for step in  tqdm(range(steps)):
                test_vectors = values[:size_of_series + step][observation_indexes]
                distance_matrix = calc_distance_matrix(test_vectors, train_vectors, steps, y_dim)
                points = train_vectors[distance_matrix < eps][:, -1]
                average_points_size += points.size
            average_points_size /= steps
            if average_points_size > 100:
                print(eps, average_points_size)
                eps /= param

            elif average_points_size < 10:
                print(eps, average_points_size)

                eps *= param
            else:
                print(eps, average_points_size)

                self.eps = eps
        print(eps, average_points_size)

        return

    def pull(self, eps: float):
        steps = len(self.time_series_.test)
        values = self.time_series_.train
        values += self.time_series_.val
        size_of_series = len(values)
        values = np.array(values)
        values.resize(size_of_series + steps)
        values[-steps:] = np.nan
        forecast_trajectories = np.full((steps, 1), np.nan)
        x_dim, y_dim, z_dim = self.templates_.train_set.shape
        vectors_continuation = np.full([x_dim, steps, z_dim], fill_value=np.inf)
        train_vectors = np.hstack([self.templates_.train_set, vectors_continuation])
        observation_indexes = self.templates_.observation_indexes  # отрицательные номера элементов которые нужны для шаблона
        for step in tqdm(range(steps)):
            test_vectors = values[:size_of_series + step][observation_indexes]
            distance_matrix = calc_distance_matrix(test_vectors, train_vectors, steps, y_dim)
            points = train_vectors[distance_matrix < eps][:, -1]
            forecast_point = self.freeze_point(points, 'cl')
            forecast_trajectories[step, 0] = forecast_point
            values[size_of_series + step] = forecast_point  # необязательно
        return forecast_trajectories, values

    def freeze_point(self, points_pool: np.ndarray, how: str) -> float:
        result = None
        if points_pool.size == 0:
            result = np.nan
            return result
        if how == 'mean':
            result = float(points_pool.mean())

        elif how == 'mf':
            points, counts = np.unique(points_pool, return_counts=True)
            result = points[counts.argmax()]
        elif how == 'cl':
            if len(points_pool) < 16:
                dbs = Wishart(k=len(points_pool), mu=0.45)
            else:
                dbs = Wishart(k=16, mu=0.45)
            if points_pool.size > 100:
                points_pool = points_pool[:100]
            dbs.fit(points_pool.reshape(-1, 1))

            cluster_labels, cluster_sizes = np.unique(dbs.labels_[dbs.labels_ > -1], return_counts=True)

            if cluster_labels.size > 0:
                biggest_cluster_center = points_pool[dbs.labels_ == cluster_labels[cluster_sizes.argmax()]].mean()
                result = biggest_cluster_center
            else:
                result = np.nan
        return result
