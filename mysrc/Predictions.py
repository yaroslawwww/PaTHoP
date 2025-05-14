# coding: utf-8
import os.path
import sys
from collections import defaultdict
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from tqdm import tqdm

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

    def create_train_set(self, time_series):
        all_train_sets = []  # Список для хранения train_set каждого временного ряда
        if time_series.train is not None:
            data = np.array(time_series.train)
        else:
            data = np.array(time_series.values)

        self.add_data_to_train_set(data, all_train_sets)
        if all_train_sets:
            self.train_set = np.concatenate(all_train_sets, axis=1)


def calc_distance_matrix(test_vectors, train_vectors):
    return np.squeeze(cdist(test_vectors, train_vectors, 'euclidean'), axis=0)


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
        self.motifs = None
        self.affiliation = None

    def fit(self, time_series_list, template_length,
            max_template_spread):
        print("Fitting\n")
        self.templates_ = Templates(template_length, max_template_spread)
        wishart = Wishart(k=self.k, mu=self.mu)
        self.motifs = dict()
        self.affiliation = dict()
        file_path = f"../labels/{max_template_spread}_{sys.argv[1]}_{sys.argv[2]}_{sys.argv[3]}_{sys.argv[4]}_{sys.argv[5]}.npz"
        if os.path.exists(file_path):
            save_labels = np.load(file_path)
            for i, ts in enumerate(time_series_list):
                self.templates_.create_train_set(ts)
                z_vectors = self.templates_.train_set
                for template in tqdm(range(z_vectors.shape[0])):
                    inf_mask = ~np.isinf(z_vectors[template]).any(axis=1)
                    temp_z_v = z_vectors[template][inf_mask]
                    wishart.labels_ = save_labels[f"arr_{template}"]
                    cluster_labels, cluster_sizes = np.unique(wishart.labels_[wishart.labels_ > -1], return_counts=True)
                    motifs = [temp_z_v[wishart.labels_ == i].mean(axis=0) for i in cluster_labels]
                    if template in self.motifs:
                        self.motifs[template] += list(np.array(motifs).reshape(-1, len(motifs[0])))
                    else:
                        self.motifs[template] = list(np.array(motifs).reshape(-1, len(motifs[0])))
                    if template in self.affiliation:
                        self.affiliation[template] += list(
                            np.full(fill_value=i, shape=len(np.array(motifs).reshape(-1, len(motifs[0])))))
                    else:
                        self.affiliation[template] = list(
                            np.full(fill_value=i, shape=len(np.array(motifs).reshape(-1, len(motifs[0])))))
        else:
            save_labels = []
            for i, ts in enumerate(time_series_list):
                self.templates_.create_train_set(ts)
                z_vectors = self.templates_.train_set
                for template in tqdm(range(z_vectors.shape[0])):
                    inf_mask = ~np.isinf(z_vectors[template]).any(axis=1)
                    temp_z_v = z_vectors[template][inf_mask]
                    wishart.fit(temp_z_v)
                    cluster_labels, cluster_sizes = np.unique(wishart.labels_[wishart.labels_ > -1], return_counts=True)
                    save_labels.append(wishart.labels_)
                    motifs = [temp_z_v[wishart.labels_ == i].mean(axis=0) for i in cluster_labels]
                    if template in self.motifs:
                        self.motifs[template] += list(np.array(motifs).reshape(-1, len(motifs[0])))
                    else:
                        self.motifs[template] = list(np.array(motifs).reshape(-1, len(motifs[0])))
                    if template in self.affiliation:
                        self.affiliation[template] += list(
                            np.full(fill_value=i, shape=len(np.array(motifs).reshape(-1, len(motifs[0])))))
                    else:
                        self.affiliation[template] = list(
                            np.full(fill_value=i, shape=len(np.array(motifs).reshape(-1, len(motifs[0])))))

            np.savez(file_path, *save_labels)
        for template in self.motifs.keys():
            self.motifs[template] = np.array(self.motifs[template])
        for template in self.affiliation.keys():
            self.affiliation[template] = np.array(self.affiliation[template])

    def predict(self, time_series, window_index, test_size, eps):
        print("Predicting\n")
        self.time_series_ = time_series
        self.time_series_.split_train_val_test(window_index, test_size)
        steps = len(self.time_series_.test)
        values = self.time_series_.train
        values += self.time_series_.val
        size_of_series = len(values)
        values.extend([np.nan] * steps)
        values = np.array(values)
        forecast_trajectories = np.full((steps, 1), np.nan)
        observation_indexes = self.templates_.observation_indexes
        mean_affil = 0
        for step in range(steps):
            test_vectors = values[:size_of_series + step][observation_indexes]
            motifs_pool = []
            motifs_affil = []
            for template in self.motifs.keys():
                train_truncated_vectors_template = np.array(self.motifs[template])[:, :-1]
                distance_matrix = calc_distance_matrix([test_vectors[template]], train_truncated_vectors_template)
                distance_mask = distance_matrix < eps
                best_motifs = self.motifs[template][distance_mask]
                motifs_pool.extend(best_motifs)
                # print(distance_mask.shape)
                # print(np.array(self.affiliation[template]).shape)
                # print(self.affiliation[template])
                motifs_affil.extend(self.affiliation[template][distance_mask])
            motifs_pool = np.array(motifs_pool)
            forecast_point = self.freeze_point(motifs_pool)
            forecast_trajectories[step, 0] = forecast_point
            values[size_of_series + step] = forecast_point
            # print(motifs_affil)
            mean_affil += np.mean(np.array(motifs_affil)) / steps if len(motifs_affil) > 0 else 0
        return mean_affil, values

    def freeze_point(self, motifs_pool):
        if motifs_pool.size == 0:
            result = np.nan
            return result
        points_pool = motifs_pool[:, -1].reshape(-1, 1)
        dbs = DBSCAN(0.01, min_samples=4)
        dbs.fit(points_pool)
        cluster_labels, cluster_sizes = np.unique(dbs.labels_[dbs.labels_ > -1], return_counts=True)
        if cluster_labels.size > 0 and (
                np.count_nonzero(((cluster_sizes / cluster_sizes.max()).round(2) > 0.3)) == 1):
            mask = (dbs.labels_ == cluster_labels[cluster_sizes.argmax()])
            biggest_cluster_center = points_pool[mask].mean()
            result = biggest_cluster_center
        else:
            result = np.nan
        return result
