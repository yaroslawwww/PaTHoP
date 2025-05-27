# coding: utf-8
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import shared_memory
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from tqdm import tqdm
from WishartClusterizationAlgorithm import Wishart
import math

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
    zero_mask = y_true_masked != 0
    if not np.any(zero_mask):
        return np.nan
    y_true_non_zero = y_true_masked[zero_mask]
    y_pred_non_zero = y_pred_masked[zero_mask]
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

class TimeSeries:
    def __init__(self, series_type="Lorentz", size=0, r=28, dt=0.001):
        if series_type == "Lorentz":
            divisor = int(0.1 / dt)
            x, y, z = Lorentz().generate(dt=dt, steps=size * divisor, r=r)
            x = (x - x.min()) / (x.max() - x.min())
            self.values = list(x)[::divisor]
        else:
            raise ValueError("Unsupported series type")
        self.train = None
        self.val = None
        self.test = None

class Templates:
    def __init__(self, template_length, max_template_spread, last_step_spread=0):
        self.train_set = None
        self.train_set_shm = None
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
                block = np.repeat(np.arange(1 + last_step_spread, max_template_spread + 1 + last_step_spread), step_size)
            templates[:, i] = np.tile(block, repeat_count // max_template_spread) + templates[:, i - 1]
        self.templates = templates
        shapes = np.diff(templates, axis=1)
        self.observation_indexes = shapes[:, ::-1].cumsum(axis=1)[:, ::-1] * -1

    def add_data_to_train_set(self, data, train_set, y_offset):
        if len(data) == 0:
            return
        x_dim = self.templates.shape[0]
        z_dim = self.templates.shape[1]
        for i in range(x_dim):
            template_window = self.templates[i][-1]
            n_windows = len(data) - template_window
            if n_windows > 0:
                time_series_indexes = self.templates[i] + np.arange(n_windows)[:, None]
                time_series_vectors = data[time_series_indexes]
                train_set[i, y_offset:y_offset + n_windows] = time_series_vectors

    def add_data_to_affiliation_matrix(self, data, affiliation_matrix, index, y_offset):
        x_dim = self.templates.shape[0]
        y_dim = 0
        for i in range(x_dim):
            template_window = self.templates[i][-1]
            y_dim = max(len(data) - template_window, y_dim)
        z_dim = self.templates.shape[1]
        affiliation_matrix.append(np.full((x_dim, y_dim, z_dim), index, dtype=int))

    def create_train_set(self, time_series_list):
        self.affiliation_matrix = []
        y_max = 0
        for time_series in time_series_list:
            data = np.array(time_series.train if time_series.train is not None else time_series.values)
            for i in range(self.templates.shape[0]):
                template_window = self.templates[i][-1]
                n_windows = len(data) - template_window if len(data) > template_window else 0
                y_max = max(y_max, n_windows)
        x_dim = self.templates.shape[0]
        z_dim = self.templates.shape[1]
        self.train_set_shm = shared_memory.SharedMemory(create=True, size=x_dim * y_max * z_dim * 8)
        self.train_set = np.ndarray((x_dim, y_max, z_dim), dtype=np.float64, buffer=self.train_set_shm.buf)
        self.train_set.fill(np.inf)
        y_offset = 0
        for i, time_series in enumerate(time_series_list):
            data = np.array(time_series.train if time_series.train is not None else time_series.values)
            self.add_data_to_train_set(data, self.train_set, y_offset)
            self.add_data_to_affiliation_matrix(data, self.affiliation_matrix, i, y_offset)
            y_offset += len(data) - min(self.templates[:, -1])
        if self.affiliation_matrix:
            self.affiliation_matrix = np.concatenate(self.affiliation_matrix, axis=1)

class TSProcessor:
    def __init__(self, k=16, mu=0.45):
        self.ts_number = None
        self.templates_ = None
        self.time_series_ = None
        self.k, self.mu = k, mu

    def fit(self, time_series_list, template_length, max_template_spread, last_step_spread):
        print(f"Fitting for lts={last_step_spread}", flush=True)
        self.templates_ = Templates(template_length, max_template_spread, last_step_spread)
        self.templates_.create_train_set(time_series_list)
        wishart = Wishart(k=self.k, mu=self.mu)
        self.motifs = dict()
        file_path = f"../labels/elongated_{max_template_spread}_{last_step_spread}.npz"
        if os.path.exists(file_path):
            save_labels = np.load(file_path)
            z_vectors = self.templates_.train_set
            for template in tqdm(range(z_vectors.shape[0]), desc=f"Processing templates (lts={last_step_spread})"):
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
            for template in tqdm(range(z_vectors.shape[0]), desc=f"Processing templates (lts={last_step_spread})"):
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

    def predict(self, time_series, window_index, eps):
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
            for template in range(self.templates_.train_set.shape[0]):
                inf_mask = ~np.isinf(self.motifs[template]).any(axis=1)
                train_vectors = self.motifs[template][inf_mask]
                if train_vectors.size == 0:
                    continue
                train_truncated_vectors = train_vectors[:, :-1]
                distance_matrix = cdist([test_vectors[template]], train_truncated_vectors, 'euclidean')
                distance_mask = distance_matrix[0] < eps
                best_motifs = train_vectors[distance_mask]
                motifs_pool.extend(best_motifs)
            motifs_pool = np.array(motifs_pool)
            forecast_point = self.freeze_point(motifs_pool)
            forecast_trajectories[step, 0] = forecast_point
            values[size_of_series + step] = forecast_point
        return forecast_trajectories, values

    def freeze_point(self, motifs_pool):
        if motifs_pool.size == 0:
            return np.nan
        points_pool = motifs_pool[:, -1].reshape(-1, 1)
        dbs = DBSCAN(eps=0.01, min_samples=10)
        dbs.fit(points_pool)
        cluster_labels, cluster_sizes = np.unique(dbs.labels_[dbs.labels_ > -1], return_counts=True)
        cluster_centers = [float(points_pool[dbs.labels_ == i].mean()) for i in cluster_labels]
        if len(cluster_sizes) > 3:
            indices_smallest = np.argpartition(cluster_sizes, -3)[-3:]
            mask = np.zeros_like(cluster_sizes, dtype=bool)
            mask[indices_smallest] = True
        if cluster_labels.size > 0 and np.count_nonzero((cluster_sizes / cluster_sizes.max()).round(2) > 0.3) == 1:
            mask = (dbs.labels_ == cluster_labels[cluster_sizes.argmax()])
            biggest_cluster_center = points_pool[mask].mean()
            return biggest_cluster_center
        return np.nan

def predict_handler(gap_number, window_size, epsilon, ts, ts_processor):
    ts_size = len(ts.values)
    window_index = ts_size - (gap_number + 1) - window_size
    if window_index > len(ts.values) or window_index < 0:
        raise ValueError("Window index out of range")
    fort, values = ts_processor.predict(ts, window_index, epsilon)
    real_values = np.array(ts.values[window_index:window_index + window_size])
    pred_values = np.array(list(values)[window_index:window_index + window_size])
    is_np_point = 1 if np.isnan(pred_values[-1]) else 0
    return pred_values[-1], is_np_point, real_values[-1]

def batch_tasks(total_tasks, cpu_cores=1):
    """Split tasks into batches based on number of cores"""
    batch_size = math.ceil(total_tasks / cpu_cores)
    return [range(i, min(i + batch_size, total_tasks))
            for i in range(0, total_tasks, batch_size)]

def process_batch(batch, ts, ts_processor, eps):
    """Process a batch of prediction tasks"""
    return [predict_handler(gap, 1, eps, ts, ts_processor) for gap in batch]

def process_lts(lts):
    """Process a single lts iteration"""
    # Initialize parameters
    r = 28
    dt = 0.001
    total_steps = 25500
    train_size = 10000
    test_size = 25500
    template_length = 4
    max_template_spread = 10
    val_size = 0
    total_predictions = 3000
    eps = 0.01

    # Initialize time series
    ts = TimeSeries("Lorentz", size=total_steps, r=r, dt=dt)
    ts.train = ts.values[:train_size]
    ts.test = ts.values[train_size + val_size:train_size + val_size + test_size]
    list_ts = [ts]

    # Initialize processor and fit
    tsproc = TSProcessor()
    print(f"Starting model fitting for lts={lts}...", flush=True)
    tsproc.fit(list_ts, template_length, max_template_spread, lts)
    print(f"Model fitting completed for lts={lts}.", flush=True)

    # Prepare batches for prediction
    batches = batch_tasks(total_predictions, cpu_cores=1)
    print(f"Using {len(batches)} batches for lts={lts}", flush=True)

    pred_points_values = []
    is_np_points = []
    real_points_values = []

    # Process predictions sequentially within this lts
    with ProcessPoolExecutor(max_workers=1) as executor:
        futures = {
            executor.submit(
                process_batch,
                batch,
                list_ts[0],
                tsproc,
                eps
            ): batch for batch in batches
        }
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"Processing batches (lts={lts})",
            smoothing=0,
            file=sys.stdout
        ):
            batch_results = future.result()
            for result in batch_results:
                if result:
                    pred, is_np, real = result
                    pred_points_values.append(pred)
                    is_np_points.append(is_np)
                    real_points_values.append(real)

    # Compute metrics
    rmse_val = rmse(real_points_values, pred_points_values)
    nan_ratio = np.mean(is_np_points)
    mape_val = mape(real_points_values, pred_points_values)
    print(f"\nMetrics for lts={lts}:", flush=True)
    print(f"RMSE: {rmse_val:.4f}", flush=True)
    print(f"NaN ratio: {nan_ratio:.2%}", flush=True)
    print(f"MAPE: {mape_val:.2%}", flush=True)

    return rmse_val, nan_ratio, mape_val, lts

def main():
    # Initialize parameters
    lts_range = range(0, 200)
    predictions = []

    # Parallelize over lts
    with ProcessPoolExecutor(max_workers=32) as executor:
        futures = {executor.submit(process_lts, lts): lts for lts in lts_range}
        for future in tqdm(
            as_completed(futures),
            total=len(lts_range),
            desc="Processing lts iterations",
            smoothing=0,
            file=sys.stdout
        ):
            result = future.result()
            if result:
                predictions.append(result)

    # Sort predictions by lts for consistent output
    predictions.sort(key=lambda x: x[3])

    # Write results to file
    with open("/home/ikvasilev/PaTHoP/results/stretched", "a") as f:
        for item in predictions:
            line = ",".join(map(str, item))
            f.write(line + "\n")

if __name__ == '__main__':
    sys.stdout.reconfigure(line_buffering=True)
    main()