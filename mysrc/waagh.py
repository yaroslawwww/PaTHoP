# coding: utf-8
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from tqdm import tqdm
from WishartClusterizationAlgorithm import Wishart
# Lorenz system class



def rmse(y_true, y_pred):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)

    y_true_masked = y_true[mask]
    y_pred_masked = y_pred[mask]

    if len(y_true_masked) == 0:
        return np.nan

    mse = np.mean((y_true_masked - y_pred_masked) ** 2)
    return np.sqrt(mse)


def mape(y_true, y_pred):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    # Маска для исключения NaN
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true_masked = y_true[mask]
    y_pred_masked = y_pred[mask]
    if len(y_true_masked) == 0:
        return 0
    # Проверка на наличие нулей в y_true
    zero_mask = y_true_masked != 0
    if not np.any(zero_mask):
        return np.nan  # Если все значения в y_true равны нулю после маски

    # Вычисление MAPE только для ненулевых значений
    y_true_non_zero = y_true_masked[zero_mask]
    y_pred_non_zero = y_pred_masked[zero_mask]

    # Расчет абсолютной процентной ошибки
    ape = np.abs((y_true_non_zero - y_pred_non_zero) / y_true_non_zero)
    return np.mean(ape)


def rmse(y_true, y_pred):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)

    y_true_masked = y_true[mask]
    y_pred_masked = y_pred[mask]

    if len(y_true_masked) == 0:
        return np.nan

    mse = np.mean((y_true_masked - y_pred_masked) ** 2)
    return np.sqrt(mse)

def mape(y_true, y_pred):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)

    y_true_masked = y_true[mask]
    y_pred_masked = y_pred[mask]

    if len(y_true_masked) == 0:
        return np.nan

    mape = np.mean(np.abs(y_true_masked - y_pred_masked))
    return mape

class Lorentz:
    def __init__(self, s=10, b=8 / 3):
        self.s = s
        self.b = b
        self.r = None

    def X(self, x, y, s):
        return s * (y - x)

    def Y(self, x, y, z, r):
        return (-x) * z + r * x - y

    def Z(self, x, y, z, b):
        return x * y - b * z

    def RK4(self, x, y, z, s, r, b, dt):
        k_1 = self.X(x, y, s)
        l_1 = self.Y(x, y, z, r)
        m_1 = self.Z(x, y, z, b)

        k_2 = self.X((x + k_1 * dt * 0.5), (y + l_1 * dt * 0.5), s)
        l_2 = self.Y((x + k_1 * dt * 0.5), (y + l_1 * dt * 0.5), (z + m_1 * dt * 0.5), r)
        m_2 = self.Z((x + k_1 * dt * 0.5), (y + l_1 * dt * 0.5), (z + m_1 * dt * 0.5), b)

        k_3 = self.X((x + k_2 * dt * 0.5), (y + l_2 * dt * 0.5), s)
        l_3 = self.Y((x + k_2 * dt * 0.5), (y + l_2 * dt * 0.5), (z + m_2 * dt * 0.5), r)
        m_3 = self.Z((x + k_2 * dt * 0.5), (y + l_2 * dt * 0.5), (z + m_2 * dt * 0.5), b)

        k_4 = self.X((x + k_3 * dt), (y + l_3 * dt), s)
        l_4 = self.Y((x + k_3 * dt), (y + l_3 * dt), (z + m_3 * dt), r)
        m_4 = self.Z((x + k_3 * dt), (y + l_3 * dt), (z + m_3 * dt), b)

        x += (k_1 + 2 * k_2 + 2 * k_3 + k_4) * dt * (1 / 6)
        y += (l_1 + 2 * l_2 + 2 * l_3 + l_4) * dt * (1 / 6)
        z += (m_1 + 2 * m_2 + 2 * m_3 + m_4) * dt * (1 / 6)

        return x, y, z

    def generate(self, dt, steps, r=28):
        x_0, y_0, z_0 = 1, 1, 1
        x_list = [x_0]
        y_list = [y_0]
        z_list = [z_0]
        self.r = r
        for i in range(steps):
            x, y, z = self.RK4(x_list[i], y_list[i], z_list[i], self.s, self.r, self.b, dt)
            x_list.append(x)
            y_list.append(y)
            z_list.append(z)
        return np.array(x_list), np.array(y_list), np.array(z_list)

# TimeSeries class
class TimeSeries:
    def __init__(self, series_type="Lorentz", size=0, r=28, dt=0.001):
        if series_type == "Lorentz":
            divisor = int(0.1 / dt)
            x, y, z = Lorentz().generate(dt=dt, steps=size * divisor, r=r)
            x = (x - x.min()) / (x.max() - x.min())  # Normalize
            self.values = list(x)[::divisor]
        else:
            raise ValueError("Unsupported series type")
        self.train = None
        self.val = None
        self.test = None
class Templates:
    def __init__(self, template_length, max_template_spread,last_step_spread = 0):
        self.train_set = None
        self.affiliation_matrix = None
        self.template_length = template_length
        self.max_template_spread = max_template_spread

        templates_quantity = max_template_spread ** (template_length - 1)
        templates = np.zeros((templates_quantity, template_length), dtype=int)

        for i in range(1, template_length):
            step_size = max_template_spread ** (template_length - i - 1)
            repeat_count = max_template_spread ** i
            if i != template_length - 1:
                block = np.repeat(np.arange(1, max_template_spread + 1), step_size)
            else:
                block = np.repeat(np.arange(1+last_step_spread, max_template_spread + 1 + last_step_spread), step_size)
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
    return np.squeeze(cdist(test_vectors, train_vectors, 'euclidean'),axis = 0)


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
    def fit(self, time_series_list, template_length,
            max_template_spread):
        print("Fisting\n")
        self.templates_ = Templates(template_length, max_template_spread,100)
        self.templates_.create_train_set(time_series_list)
        self.ts_number = len(time_series_list)
        wishart = Wishart(k=self.k, mu=self.mu)
        self.motifs = dict()
        z_vectors = self.templates_.train_set
        for template in tqdm(range(z_vectors.shape[0])):
            inf_mask = ~np.isinf(z_vectors[template]).any(axis=1)
            temp_z_v = z_vectors[template][inf_mask]
            self.motifs[template] = np.array(temp_z_v).reshape(-1, len(temp_z_v[0]))
    def predict(self, time_series, window_index, eps):
        print("Predicting\n")
        self.time_series_ = time_series
        steps = 5
        values = self.time_series_.values[:window_index]
        size_of_series = len(values)
        values.extend([np.nan] * steps)
        values = np.array(values)
        forecast_trajectories = np.full((steps, 1), np.nan)
        observation_indexes = self.templates_.observation_indexes
        for step in [0]:
            test_vectors = values[:size_of_series + step][observation_indexes]

            motifs_pool = []

            for template in self.motifs.keys():
                train_truncated_vectors_template = self.motifs[template][:,:-1]
                distance_matrix = calc_distance_matrix([test_vectors[template]], train_truncated_vectors_template)
                # distance_mask = distance_matrix < eps
                # best_motifs = self.motifs[template][distance_mask]
                min_value = min(distance_matrix)
                min_index = list(distance_matrix).index(min_value)
                indices_smallest = np.argpartition(distance_matrix, 1)[:1]

                # Создаем маску
                mask = np.zeros_like(distance_matrix, dtype=bool)
                mask[indices_smallest] = True
                # mask = np.zeros(len(distance_matrix), dtype=bool)
                # mask[min_index] = True
                best_motifs = self.motifs[template][mask]
                motifs_pool.extend(best_motifs)

            motifs_pool = np.array(motifs_pool)
            print(motifs_pool.shape)
            forecast_point = self.freeze_point(motifs_pool)
            forecast_trajectories[step, 0] = forecast_point
            values[size_of_series + step] = forecast_point
        return forecast_trajectories, values

    def freeze_point(self, motifs_pool):
        if motifs_pool.size == 0:
            result = np.nan
            return result
        points_pool = motifs_pool[:, -1].reshape(-1, 1)
        dbs = DBSCAN(eps=0.01,min_samples=10)
        dbs.fit(points_pool.reshape(-1,1))
        cluster_labels, cluster_sizes = np.unique(dbs.labels_[dbs.labels_ > -1], return_counts=True)
        # print(cluster_sizes)
        cluster_centers = []
        for i in cluster_labels:
            cluster_centers.append(float(points_pool[dbs.labels_ == i].mean()))
        # print(cluster_centers)
        indices_smallest = np.argpartition(cluster_sizes, -3)[-3:]
        mask = np.zeros_like(cluster_sizes, dtype=bool)
        mask[indices_smallest] = True
        print(np.array(cluster_sizes)[mask],np.array(cluster_centers)[mask])
        if cluster_labels.size > 0 and (
                np.count_nonzero(((cluster_sizes / cluster_sizes.max()).round(2) > 0.5)) == 1):
            mask = (dbs.labels_ == cluster_labels[cluster_sizes.argmax()])
            biggest_cluster_center = points_pool[mask].mean()
            result = biggest_cluster_center
        else:
            result = np.nan
        return result

def predict_handler(gap_number, window_size,
                    epsilon, ts, ts_processor: TSProcessor):
    ts_size = len(ts.values)
    window_index = ts_size - (gap_number + 1) - window_size
    if window_index > len(ts.values) or window_index < 0:
        raise ValueError("Window index out of range")
    fort, values = ts_processor.predict(ts, window_index, epsilon)
    real_values = np.array(ts.values[window_index:window_index + window_size])
    pred_values = np.array(list(values)[window_index:window_index + window_size])
    is_np_point = 1 if np.isnan(pred_values[-1]) else 0
    print(real_values[-1], pred_values[-1])
    mask = ~np.isnan(real_values[-1]) & ~np.isnan(pred_values[-1])
    print("res:",abs(real_values[-1][mask]-pred_values[-1][mask]).round(2))
    return pred_values[-1], is_np_point, real_values[-1]


def main():
    r = 28
    dt = 0.001
    total_steps = 30000
    train_size = 10000
    test_size = 10000
    template_length = 4
    max_template_spread = 15
    val_size = 0
    eps = 0.002

    ts = TimeSeries("Lorentz", size=total_steps, r=r, dt=dt)
    ts.train = ts.values[:train_size]
    ts.val = ts.values[train_size:train_size + val_size]
    ts.test = ts.values[train_size + val_size:train_size + val_size + test_size]
    list_ts = [ts]
    tsproc = TSProcessor()
    tsproc.fit(list_ts, template_length, max_template_spread)
    pred_points_values = []
    is_np_points = []
    real_points_values = []
    with ProcessPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(predict_handler, gap, 1, eps, list_ts[0], tsproc)
                   for gap in range(1000)]
        for future in futures:
            result = future.result()
            if result is not None and len(result) > 0:
                pred_points_values.append(result[0])
                is_np_points.append(result[1])
                real_points_values.append(result[2])
    print( rmse(pred_points_values, real_points_values), np.mean(is_np_points),mape(pred_points_values, real_points_values))

if __name__ == '__main__':
    main()