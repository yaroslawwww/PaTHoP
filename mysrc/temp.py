# coding: utf-8
import os
import sys

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

# Templates class
class Templates:
    def __init__(self, template_length, max_template_spread,last_step_spread = 0):
        self.template_length = template_length
        self.max_template_spread = max_template_spread
        templates_quantity = max_template_spread ** (template_length - 1)
        templates = np.zeros((templates_quantity, template_length), dtype=int)
        for i in range(1, template_length):
            step_size = max_template_spread ** (template_length - i - 1)
            repeat_count = max_template_spread ** i
            if i != template_length - 1:
                block = np.repeat(np.arange(1, max_template_spread + 1), step_size)
                templates[:, i] = np.tile(block, repeat_count // max_template_spread) + templates[:, i - 1]
            else:
                block = np.repeat(np.arange(1 + last_step_spread, max_template_spread + 1 + last_step_spread), step_size)
                templates[:, i] = np.tile(block, repeat_count // max_template_spread) + templates[:, i - 1]
        self.templates = templates
        shapes = np.diff(templates, axis=1)
        self.observation_indexes = shapes[:, ::-1].cumsum(axis=1)[:, ::-1] * -1

    def create_train_set(self, time_series_list):
        all_train_sets = []
        for time_series in time_series_list:
            data = np.array(time_series.train)
            x_dim = self.templates.shape[0]
            y_dim = max(len(data) - self.templates[i][-1] for i in range(x_dim))
            z_dim = self.templates.shape[1]
            individual_train_set = np.full((x_dim, y_dim, z_dim), np.inf, dtype=float)
            for i in range(x_dim):
                template_window = self.templates[i][-1]
                n_windows = len(data) - template_window
                if n_windows > 0:
                    time_series_indexes = self.templates[i] + np.arange(n_windows)[:, None]
                    time_series_vectors = data[time_series_indexes]
                    individual_train_set[i, :n_windows] = time_series_vectors
            all_train_sets.append(individual_train_set)
        self.train_set = np.concatenate(all_train_sets, axis=1)

# TSProcessor class
class TSProcessor:
    def __init__(self, k=16, mu=0.45):
        self.k, self.mu = k, mu
        self.motifs = None
        self.time_series_ = None

    def fit(self, time_series_list, template_length,
            max_template_spread):
        print("Fitting\n")
        self.templates_ = Templates(template_length, max_template_spread,100)
        wishart = Wishart(k=self.k, mu=self.mu)
        self.motifs = dict()
        self.templates_.create_train_set(time_series_list)
        file_path = f"../labels/{max_template_spread}.npz"
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

            np.savez(file_path, *save_labels)
        for template in self.motifs.keys():
            self.motifs[template] = np.array(self.motifs[template])

    def _process_template(self, template, test_vector, eps):
        """Обработка одного шаблона (вынесено в отдельный метод)"""
        train_truncated = self.motifs[template][:, :-1]
        distance_matrix = cdist([test_vector], train_truncated, 'euclidean')
        distance_mask = distance_matrix[0] < eps
        return (
            self.motifs[template][distance_mask],
            [(template, idx) for idx in np.where(distance_mask)[0]]
        )

    def motifs_validation(self, validation_set, prediction_size, eps = 0.02, beta=0.05, filter_threshold=0.2):
        beta = 0.05
        values = self.time_series_.train.copy()
        size_of_series = len(values)
        steps = min(prediction_size, len(validation_set))
        values.extend(validation_set[:steps])
        values.extend([np.nan] * steps)
        values = np.array(values)

        predicted_val = []
        actual_val = []
        is_np_points = []
        motif_errors = {t: {i: [] for i in range(len(self.motifs[t]))} for t in self.motifs}
        motif_counts = {t: {i: 0 for i in range(len(self.motifs[t]))} for t in self.motifs}

        for step in tqdm(range(steps), desc="Predicting"):
            p = size_of_series + step
            test_vectors = values[:p][self.templates_.observation_indexes]
            motifs_pool = []
            motifs_indices = []
            for template in self.motifs.keys():
                train_truncated_vectors_template = self.motifs[template][:, :-1]
                distance_matrix = cdist([test_vectors[template]], train_truncated_vectors_template, 'euclidean')
                distance_mask = distance_matrix[0] < eps
                best_motifs = self.motifs[template][distance_mask]
                best_indices = np.where(distance_mask)[0]
                motifs_pool.extend(best_motifs)
                motifs_indices.extend([(template, idx) for idx in best_indices])
            motifs_pool = np.array(motifs_pool)
            forecast_point = self.freeze_point(motifs_pool)
            actual_point = validation_set[step]
            predicted_val.append(forecast_point)
            actual_val.append(actual_point)
            is_np_points.append(1 if np.isnan(forecast_point) else 0)
            if motifs_pool.size > 0:
                pred_points = motifs_pool[:, -1]
                errors = np.abs(pred_points - actual_point)
                for (template, motif_idx), error in zip(motifs_indices, errors):
                    motif_errors[template][motif_idx].append((step, error))
                    if error < beta:
                        motif_counts[template][motif_idx] += 1

        min_counts = 10
        active_templates = []
        active_motifs = {}
        for t in self.motifs.keys():
            motif_counts_array = np.array([motif_counts[t][i] for i in range(len(self.motifs[t]))])
            active_indices = np.where(motif_counts_array >= min_counts)[0]
            if len(active_indices) > 0:
                active_templates.append(t)
                active_motifs[t] = active_indices.tolist()

        if not active_templates:
            metrics_before = {
                "rmse": rmse(actual_val,predicted_val),
                "mape": mape(actual_val,predicted_val),
                "mean_is_np": np.mean(is_np_points),
                "motif_count": sum(len(self.motifs[t]) for t in self.motifs)
            }
            return {
                "before_filtering": metrics_before,
                "after_filtering": metrics_before,
                "prognostic_values": {}
            }

        errors_array = []
        for t in active_templates:
            for m_idx in active_motifs[t]:
                for obs_idx, error in motif_errors[t][m_idx]:
                    errors_array.append([t, m_idx, obs_idx, error])
        errors_array = np.array(errors_array, dtype=np.float64)

        prognostic_values = {}
        # max_observations = 5000
        # sampled_obs = np.random.choice(steps, size=min(steps, max_observations), replace=False)

        for t in tqdm(active_templates, desc="Computing Q_k"):
            prognostic_values[t] = []
            for m_idx in active_motifs[t]:
                total_uses = len([e for o, e in motif_errors[t][m_idx]])
                if total_uses == 0:
                    Q_k = 0.0
                else:
                    Q_k = motif_counts[t][m_idx] / total_uses
                prognostic_values[t].append(Q_k)
            prognostic_values[t] = np.array(prognostic_values[t])

        metrics_before = {
            "rmse": rmse(actual_val,predicted_val),
            "mape": mape(actual_val,predicted_val),
            "mean_is_np": np.mean(is_np_points),
            "motif_count": sum(len(self.motifs[t]) for t in self.motifs)
        }

        original_motifs = self.motifs.copy()
        for t in active_templates:
            if len(prognostic_values[t]) == 0:
                continue
            threshold = np.percentile(prognostic_values[t], filter_threshold * 100)
            keep_mask = prognostic_values[t] >= threshold
            self.motifs[t] = self.motifs[t][active_motifs[t]][keep_mask]
            if len(self.motifs[t]) == 0:
                del self.motifs[t]

        values = self.time_series_.train.copy()
        size_of_series = len(values)
        values.extend(validation_set[:steps])
        values.extend([np.nan] * steps)
        values = np.array(values)
        predicted_val_filtered = []
        is_np_points_filtered = []

        for step in tqdm(range(steps), desc="Predicting (filtered)"):
            p = size_of_series + step
            test_vectors = values[:p][self.templates_.observation_indexes]
            motifs_pool = []
            for template in self.motifs.keys():
                train_truncated_vectors_template = self.motifs[template][:, :-1]
                distance_matrix = cdist([test_vectors[template]], train_truncated_vectors_template, 'euclidean')
                distance_mask = distance_matrix[0] < eps
                best_motifs = self.motifs[template][distance_mask]
                motifs_pool.extend(best_motifs)
            motifs_pool = np.array(motifs_pool)
            forecast_point = self.freeze_point(motifs_pool)
            predicted_val_filtered.append(forecast_point)
            is_np_points_filtered.append(1 if np.isnan(forecast_point) else 0)

        metrics_after = {
            "rmse": rmse(actual_val,predicted_val_filtered),
            "mape": mape(actual_val,predicted_val_filtered),
            "mean_is_np": np.mean(is_np_points_filtered),
            "motif_count": sum(len(self.motifs[t]) for t in self.motifs)
        }

        self.motifs = original_motifs

        return {
            "before_filtering": metrics_before,
            "after_filtering": metrics_after,
            "prognostic_values": prognostic_values
        }

    def freeze_point(self, motifs_pool):
        if motifs_pool.size == 0:
            return np.nan
        points_pool = motifs_pool[:, -1].reshape(-1, 1)
        dbs = DBSCAN(0.01, min_samples=4)
        dbs.fit(points_pool)
        cluster_labels, cluster_sizes = np.unique(dbs.labels_[dbs.labels_ > -1], return_counts=True)
        if cluster_labels.size > 0 and (np.count_nonzero(((cluster_sizes / cluster_sizes.max()).round(2) > 0.3)) == 1):
            mask = (dbs.labels_ == cluster_labels[cluster_sizes.argmax()])
            return points_pool[mask].mean()
        return np.nan

# Main function
def main():
    r = 28
    dt = 0.05
    total_steps = 200000
    train_size = 10000
    val_size = 50000
    test_size = 50000
    template_length = 4
    max_template_spread = 14
    eps = 0.005
    beta = None
    filter_threshold = 0.1

    ts = TimeSeries("Lorentz", size=total_steps, r=r, dt=dt)
    ts.train = ts.values[:train_size]
    ts.val = ts.values[train_size:train_size + val_size]
    ts.test = ts.values[train_size + val_size:train_size + val_size + test_size]

    ts_processor = TSProcessor(k=16, mu=0.45)
    ts_processor.fit(time_series_list=[ts], template_length=template_length, max_template_spread=max_template_spread)
    ts_processor.time_series_ = ts
    results = ts_processor.motifs_validation(validation_set=np.array(ts.val), prediction_size=len(ts.val), eps=eps, beta=beta, filter_threshold=filter_threshold)

    print("=== Validation Results ===")

    print("\nPrognostic Values Summary:")
    for template in results['prognostic_values']:
        values = results['prognostic_values'][template]
        if len(values) > 0:
            print(f"Template {template}:")
            print(f"  Mean Q_k: {np.mean(values):.6f}")
            print(f"  Min Q_k: {np.min(values):.6f}")
            print(f"  Max Q_k: {np.max(values):.6f}")
            print(f"  Motifs: {len(values)}")

    print("\nBefore Filtering:")
    print(f"RMSE: {results['before_filtering']['rmse']:.6f}")
    print(f"MAPE: {results['before_filtering']['mape']:.6f}")
    print(f"Mean Non-Predictable Points: {results['before_filtering']['mean_is_np']:.6f}")
    print(f"Motif Count: {results['before_filtering']['motif_count']}")
    print("\nAfter Filtering:")
    print(f"RMSE: {results['after_filtering']['rmse']:.6f}")
    print(f"MAPE: {results['after_filtering']['mape']:.6f}")
    print(f"Mean Non-Predictable Points: {results['after_filtering']['mean_is_np']:.6f}")
    print(f"Motif Count: {results['after_filtering']['motif_count']}")


if __name__ == '__main__':
    main()