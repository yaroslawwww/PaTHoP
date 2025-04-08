# coding: utf-8
from WishartClusterizationAlgorithm import Wishart
# coding: utf-8
from TimeSeries import TimeSeries
import numpy as np


class Templates:
    def __init__(self, template_length, max_template_spread):
        self.train_set = None
        self.affiliation_matrix = None
        self.template_length = template_length
        self.max_template_spread = max_template_spread

        templates_quantity = max_template_spread ** (template_length - 1)
        templates = np.zeros((templates_quantity, template_length), dtype=int)

        for i in range(1, template_length):
            step_size = max_template_spread ** (template_length - i - 1)
            repeat_count = max_template_spread ** i

            block = np.repeat(np.arange(1, max_template_spread + 1), step_size)
            templates[:, i] = np.tile(block, repeat_count // max_template_spread) + templates[:, i - 1]

        self.templates = templates

        shapes = np.diff(templates, axis=1)
        self.observation_indexes = shapes[:, ::-1].cumsum(axis=1)[:, ::-1] * -1

    def add_data_to_train_set(self, data, all_train_sets):
        if len(data) == 0:
            return  # Пропускаем пустые ряды

        # Вычисляем размерности для текущего временного ряда
        x_dim = self.templates.shape[0]
        y_dim = 0
        for i in range(x_dim):
            template_window = self.templates[i][-1]
            y_dim = max(len(data) - template_window, y_dim)

        z_dim = self.templates.shape[1]

        if y_dim <= 0:
            return  # Ряд слишком короткий для любых шаблонов

        # Инициализируем train_set для текущего ряда
        individual_train_set = np.full((x_dim, y_dim, z_dim), np.inf, dtype=float)

        # Заполняем данными
        for i in range(len(self.templates)):
            current_template = self.templates[i]
            template_window = current_template[-1]
            n_windows = len(data) - template_window

            if n_windows > 0:
                time_series_indexes = current_template + np.arange(n_windows)[:, None]
                time_series_vectors = data[time_series_indexes]
                individual_train_set[i, :n_windows] = time_series_vectors

        all_train_sets.append(individual_train_set)

    def add_data_to_affiliation_matrix(self, data, affiliation_matrix, index):
        x_dim = self.templates.shape[0]
        y_dim = 0
        for i in range(x_dim):
            template_window = self.templates[i][-1]
            y_dim = max(len(data) - template_window, y_dim)

        z_dim = self.templates.shape[1]
        affiliation_matrix.append(np.full((x_dim, y_dim, z_dim), index, dtype=int))

    def create_train_set(self, time_series_list):
        all_train_sets = []  # Список для хранения train_set каждого временного ряда
        affiliation_matrix = []
        for i, time_series in enumerate(time_series_list):
            if time_series.train is not None:
                data = np.array(time_series.train)
            else:
                data = np.array(time_series.values)

            self.add_data_to_train_set(data, all_train_sets)
            self.add_data_to_affiliation_matrix(data, affiliation_matrix, i)
        if all_train_sets:
            self.train_set = np.concatenate(all_train_sets, axis=1)
            self.affiliation_matrix = np.concatenate(affiliation_matrix, axis=1)


def calc_distance_matrix(test_vectors, train_vectors):
    # Как будто бы надо попробовать переписать эту функцию под gpu.
    shape = (test_vectors.shape[0], train_vectors.shape[1])
    distance_matrix = np.zeros(shape)
    for i in range(test_vectors.shape[1]):
        distance_matrix += (train_vectors[:, :, i] - np.repeat(test_vectors[:, i], shape[1]).reshape(-1,
                                                                                                     shape[1])) ** 2
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
    def __init__(self, k=16, mu=0.45):
        self.ts_number = None
        self.templates_ = None
        self.time_series_ = None
        self.k, self.mu = k, mu

    def fit(self, time_series_list, template_length,
            max_template_spread):
        self.templates_ = Templates(template_length, max_template_spread)
        self.templates_.create_train_set(time_series_list)
        self.ts_number = len(time_series_list)

    def pull(self, time_series, window_index, test_size, eps):
        self.time_series_ = time_series
        self.time_series_.split_train_val_test(window_index, test_size)
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
        affiliation_continuation = np.full([x_dim, steps, z_dim], fill_value=0)
        train_vectors = np.hstack([self.templates_.train_set, vectors_continuation])
        # Мы создаем параллельно с обучающими векторами вектора,
        # которые имеют ту же форму и заполнены числами, равными номеру ряда,
        # по которому был получен обучающий вектор.
        # По ним можно определить принадлежность тому или иному ряду.
        affiliation_vectors = np.hstack([self.templates_.affiliation_matrix, affiliation_continuation])
        observation_indexes = self.templates_.observation_indexes
        affiliation_result = []
        for step in range(steps):
            test_vectors = values[:size_of_series + step][observation_indexes]
            distance_matrix = calc_distance_matrix(test_vectors, train_vectors)
            distance_mask = distance_matrix < eps
            points = train_vectors[distance_mask]
            affiliation_indexes = affiliation_vectors[distance_mask][:, -1]
            forecast_point, affiliation_step_result = self.freeze_point(points, 'cl', affiliation_indexes)
            affiliation_result.append(affiliation_step_result)
            forecast_trajectories[step, 0] = forecast_point
            values[size_of_series + step] = forecast_point
        changed_aff = np.array(affiliation_result)
        if len(changed_aff) == 0:
            return forecast_trajectories, values, np.full(self.ts_number, 0)
        return forecast_trajectories, values, changed_aff[-1]

    def freeze_point(self, points_pool, how, affiliation_indexes):
        result = None
        affiliation_result = np.full(self.ts_number, np.NaN)
        if points_pool.size == 0:
            result = np.nan
            return result, affiliation_result
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
            wishart.fit(points_pool)
            # print(np.unique(points_pool,return_counts=True))
            cluster_labels, cluster_sizes = np.unique(wishart.labels_[wishart.labels_ > -1], return_counts=True)
            if cluster_labels.size > 0 and (
                    np.count_nonzero(((cluster_sizes / cluster_sizes.max()).round(2) > 0.8)) == 1):
                biggest_cluster_center = points_pool[wishart.labels_ == cluster_labels[cluster_sizes.argmax()]][:,-1].mean()

                affiliation_result = count_elements_sorted(
                    affiliation_indexes[wishart.labels_ == cluster_labels[cluster_sizes.argmax()]],
                    range(self.ts_number))
                result = biggest_cluster_center
            else:
                result = np.nan
        return result, affiliation_result
