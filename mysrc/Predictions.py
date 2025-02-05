

from TimeSeries import TimeSeries
from Patterns import Templates
import numpy as np

from WishartClusterizationAlgorithm import Wishart


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
    def __init__(self, time_series_list: list[TimeSeries], window_index: int, template_length: int = 4,
                 max_template_spread: int = 10,
                 test_size: int = 50, k: int = 16, mu: float = 0.45):
        self.time_series_ = time_series_list[0]
        self.time_series_.split_train_val_test(window_index,test_size)
        self.templates_ = Templates(template_length, max_template_spread)
        self.templates_.create_train_set(time_series_list)
        self.k, self.mu = k, mu

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
        for step in range(steps):
            test_vectors = values[:size_of_series + step][observation_indexes]
            distance_matrix = calc_distance_matrix(test_vectors, train_vectors, steps, y_dim)
            points = train_vectors[distance_matrix < eps][:, -1]
            forecast_point = self.freeze_point(points, 'cl')
            forecast_trajectories[step, 0] = forecast_point
            values[size_of_series + step] = forecast_point
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
            if len(points_pool) < self.k:
                wishart = Wishart(k=len(points_pool), mu=0.45)
            else:
                wishart = Wishart(k=self.k, mu=self.mu)
            np.random.shuffle(points_pool)
            # нужно если комп слаб
            # if points_pool.size > 100:
            #     result = self.freeze_point(points_pool, how='mean')
            #     return result
            wishart.fit(points_pool.reshape(-1, 1))

            cluster_labels, cluster_sizes = np.unique(wishart.labels_[wishart.labels_ > -1], return_counts=True)
            if cluster_labels.size > 0:
                biggest_cluster_center = points_pool[wishart.labels_ == cluster_labels[cluster_sizes.argmax()]].mean()
                result = biggest_cluster_center
            else:
                result = np.nan
        return result
