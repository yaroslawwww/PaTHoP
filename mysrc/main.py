import sys
import numpy as np
from tqdm import tqdm
from WishartClusterizationAlgorithm import Wishart
import os.path
import multiprocessing
from clusterization_metrics import compute_metrics_sequential
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

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
        i = 0
        self.r = r
        while i < steps:
            x = x_list[i]
            y = y_list[i]
            z = z_list[i]
            position = self.RK4(x, y, z, self.s, self.r, self.b, dt)
            x_list.append(position[0])
            y_list.append(position[1])
            z_list.append(position[2])
            i += 1
        x_array = np.array(x_list)
        y_array = np.array(y_list)
        z_array = np.array(z_list)
        return x_array, y_array, z_array

class TimeSeries:
    def __init__(self, series_type="Lorentz", size=0, r=28, dt=0.01, array=None):
        if series_type == "Lorentz":
            divisor = int(0.1 / dt)
            x, y, z = Lorentz().generate(dt=dt, steps=size * divisor, r=r)
            x = (x - x.min()) / (x.max() - x.min())
            self.values = list(x)[::divisor]
        else:
            x = np.array(array)
            x = (x - x.min()) / (x.max() - x.min())
            self.values = list(x)
        self.train = None
        self.after_test_train = None
        self.test = None
        self.val = []
        self.time = [i for i in range(len(self.values))]

    def split_train_val_test(self, window_index, test_size=100):
        if window_index + test_size > len(self.values):
            raise ValueError("test index out of range")
        self.train = self.values[:window_index]
        self.test = self.values[window_index:window_index + test_size]

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
            return
        x_dim = self.templates.shape[0]
        y_dim = 0
        for i in range(x_dim):
            template_window = self.templates[i][-1]
            y_dim = max(len(data) - template_window, y_dim)
        z_dim = self.templates.shape[1]
        if y_dim <= 0:
            return
        individual_train_set = np.full((x_dim, y_dim, z_dim), np.inf, dtype=float)
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
        all_train_sets = []
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



class TSProcessor:
    def __init__(self, k=16, mu=0.45):
        self.ts_number = None
        self.templates_ = None
        self.time_series_ = None
        self.k, self.mu = k, mu
        self.motifs = None

    def fit(self, time_series_list, template_length, max_template_spread,p):
        print("Fitting\n")
        self.templates_ = Templates(template_length, max_template_spread)
        wishart = Wishart(k=self.k, mu=self.mu)
        self.motifs = dict()
        self.templates_.create_train_set(time_series_list)
        file_path = f"../labels/{max_template_spread}_share_{p}.npz"

        all_metrics = []  # Для хранения метрик по шаблонам

        if os.path.exists(file_path):
            save_labels = np.load(file_path)
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

                # Вычисление метрик
                assigned_mask = wishart.labels_ >= 0
                X_assigned = temp_z_v[assigned_mask]
                labels_assigned = wishart.labels_[assigned_mask]
                if len(X_assigned) > 0 and len(np.unique(labels_assigned)) > 1:
                    metrics = compute_metrics_parallel(X_assigned, labels_assigned)
                    all_metrics.append(metrics)
        else:
            save_labels = []
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

                # Вычисление метрик
                assigned_mask = wishart.labels_ >= 0
                X_assigned = temp_z_v[assigned_mask]
                labels_assigned = wishart.labels_[assigned_mask]
                if len(X_assigned) > 0 and len(np.unique(labels_assigned)) > 1:
                    metrics = compute_metrics_sequential(X_assigned, labels_assigned)
                    all_metrics.append(metrics)

            np.savez(file_path, *save_labels)

        for template in self.motifs.keys():
            self.motifs[template] = np.array(self.motifs[template])

        # Усреднение метрик
        if all_metrics:
            metric_names = all_metrics[0].keys()
            averaged_metrics = {}
            for name in metric_names:
                values = [metrics[name] for metrics in all_metrics if name in metrics]
                averaged_metrics[name] = np.mean(values) if values else np.nan
        else:
            averaged_metrics = {name: np.nan for name in
                                ['rmsstd', 'r2', 'modified_hubert', 'calinski_harabasz', 'i_index', 'dunn',
                                 'silhouette', 'davies_bouldin', 'xie_beni', 'sd', 'sdbw', 'cvnn2']}

        return averaged_metrics


def compute_for_p(p):
    general_size = 10000  # Фиксированный размер
    r_values = [28, 35]  # Значения r для двух рядов
    dt = 0.001  # Шаг времени
    shares = [p, 1 - p]
    ts_size = [int(p * general_size), int((1 - p) * general_size)]

    list_ts = []
    for i, r in enumerate(r_values):
        size = ts_size[i]
        ts = TimeSeries("Lorentz", size=size, r=r, dt=dt)
        list_ts.append(ts)

    template_length = 4
    max_template_spread = 10
    tsproc = TSProcessor(k=16, mu=0.45)
    metrics = tsproc.fit(list_ts, template_length, max_template_spread,p)
    print(metrics)
    return p, metrics


def main():
    proportions = np.arange(0.001, 1, 0.05)
    results = []

    # Распараллеливание вычислений
    with ProcessPoolExecutor() as executor:
        # Запускаем все задачи асинхронно
        future_to_p = {executor.submit(compute_for_p, p): p for p in proportions}

        # Обрабатываем результаты по мере их поступления
        for future in as_completed(future_to_p):
            p = future_to_p[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Ошибка для p={p}: {str(e)}")

    # Сортировка результатов по значению p (опционально)
    results.sort(key=lambda x: x[0])

    # Создаем директорию для результатов
    os.makedirs('/home/ikvasilev/PaTHoP/results', exist_ok=True)

    # Сохранение результатов в файл
    with open('/home/ikvasilev/PaTHoP/results/metrics.txt', 'w') as f:
        f.write(
            'p,rmsstd,r2,modified_hubert,calinski_harabasz,i_index,dunn,silhouette,davies_bouldin,xie_beni,sd,sdbw,cvnn2\n')
        for p, metrics in results:
            line = f'{p},'
            for name in ['rmsstd', 'r2', 'modified_hubert', 'calinski_harabasz',
                         'i_index', 'dunn', 'silhouette', 'davies_bouldin',
                         'xie_beni', 'sd', 'sdbw', 'cvnn2']:
                line += f'{metrics[name]},'
            line = line.rstrip(',')
            f.write(line + '\n')

if __name__ == '__main__':
    main()