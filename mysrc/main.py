# coding: utf-8
import sys
import numpy as np
import os
from WishartClusterizationAlgorithm import Wishart
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
from scipy.stats import entropy
from tqdm import tqdm


def rmse(y_true, y_pred):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true_masked = y_true[mask]
    y_pred_masked = y_pred[mask]
    return np.sqrt(np.mean((y_true_masked - y_pred_masked) ** 2)) if len(y_true_masked) > 0 else np.nan


def mape(y_true, y_pred):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true_masked = y_true[mask]
    y_pred_masked = y_pred[mask]
    if len(y_true_masked) == 0:
        return 0
    zero_mask = y_true_masked != 0
    if not np.any(zero_mask):
        return np.nan
    y_true_non_zero = y_true_masked[zero_mask]
    y_pred_non_zero = y_pred_masked[zero_mask]
    return np.mean(np.abs((y_true_non_zero - y_pred_non_zero) / y_true_non_zero))


class Daemon:
    @staticmethod
    def is_np_basic_dbscan(points_pool, dbscan_eps=0.01, min_samples=4, dominance_threshold=0.3):
        dbs = DBSCAN(eps=dbscan_eps, min_samples=min_samples)
        dbs.fit(points_pool)
        cluster_labels, cluster_sizes = np.unique(dbs.labels_[dbs.labels_ > -1], return_counts=True)
        if cluster_labels.size == 0:
            return True
        is_multimodal = np.count_nonzero((cluster_sizes / cluster_sizes.max()).round(2) > dominance_threshold) > 1
        return is_multimodal

    @staticmethod
    def is_np_spread(points_pool, threshold=0.12):
        if points_pool.size < 2: return False
        return np.std(points_pool) > threshold

    # @staticmethod
    # def is_np_iqr(points_pool, threshold=0.15):
    #     if points_pool.size < 4: return False
    #     q75, q25 = np.percentile(points_pool, [75, 25])
    #     return (q75 - q25) > threshold

    @staticmethod
    def is_np_entropy(points_pool, threshold=2.8, bins=15):
        if points_pool.size < bins: return False
        hist, bin_edges = np.histogram(points_pool, bins=bins, density=True)
        probabilities = hist[hist > 0] * np.diff(bin_edges)[0]
        return entropy(probabilities, base=2) > threshold

    @staticmethod
    def is_np_apriori(final_prediction, true_value, threshold=0.05):
        if np.isnan(final_prediction) or true_value is None or np.isnan(true_value):
            return False
        return np.abs(final_prediction - true_value) > threshold


# --- FIX: THE DISPATCHER DICTIONARY IS DEFINED *AFTER* THE CLASS BLOCK ---
Daemon.DAEMON_DISPATCHER = {
    'basic_dbscan': Daemon.is_np_basic_dbscan,
    'spread': Daemon.is_np_spread,
    'entropy': Daemon.is_np_entropy,
    'apriori': Daemon.is_np_apriori,
}


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
        x_list, y_list, z_list = [x_0], [y_0], [z_0]
        self.r = r
        for _ in range(steps):
            x, y, z = self.RK4(x_list[-1], y_list[-1], z_list[-1], self.s, self.r, self.b, dt)
            x_list.append(x)
            y_list.append(y)
            z_list.append(z)
        return np.array(x_list), np.array(y_list), np.array(z_list)


class TimeSeries:
    def __init__(self, series_type="Lorentz", size=0, r=28, dt=0.01, array=None):
        if series_type == "Lorentz":
            divisor = int(0.1 / dt)
            x, _, _ = Lorentz().generate(dt=dt, steps=size * divisor, r=r)
            x = (x - x.min()) / (x.max() - x.min())
            self.values = x[::divisor]
        else:
            self.values = np.array(array)
            self.values = (self.values - self.values.min()) / (self.values.max() - self.values.min())
        self.train = None
        self.test = None

    def split_train_val_test(self, window_index, test_size=100):
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
        if len(data) == 0: return
        x_dim = self.templates.shape[0]
        y_dim = max(len(data) - self.templates[i][-1] for i in range(x_dim))
        if y_dim <= 0: return
        z_dim = self.templates.shape[1]
        individual_train_set = np.full((x_dim, y_dim, z_dim), np.inf, dtype=float)
        for i in range(len(self.templates)):
            template_window = self.templates[i][-1]
            n_windows = len(data) - template_window
            if n_windows > 0:
                time_series_indexes = self.templates[i] + np.arange(n_windows)[:, None]
                time_series_vectors = data[time_series_indexes]
                individual_train_set[i, :n_windows] = time_series_vectors
        all_train_sets.append(individual_train_set)

    def add_data_to_affiliation_matrix(self, data, affiliation_matrix, index):
        x_dim = self.templates.shape[0]
        y_dim = max(len(data) - self.templates[i][-1] for i in range(x_dim))
        z_dim = self.templates.shape[1]
        affiliation_matrix.append(np.full((x_dim, y_dim, z_dim), index, dtype=int))

    def create_train_set(self, time_series_list):
        all_train_sets, affiliation_matrix = [], []
        for i, time_series in enumerate(time_series_list):
            data = np.array(time_series.train if time_series.train is not None else time_series.values)
            self.add_data_to_train_set(data, all_train_sets)
            self.add_data_to_affiliation_matrix(data, affiliation_matrix, i)
        if all_train_sets:
            self.train_set = np.concatenate(all_train_sets, axis=1)
            self.affiliation_matrix = np.concatenate(affiliation_matrix, axis=1)


class TSProcessor:
    def __init__(self, k=16, mu=0.45):
        self.templates_ = None
        self.time_series_ = None
        self.k, self.mu = k, mu
        self.motifs = None
        self.daemon_func = None
        self.daemon_name = None

    def set_daemon(self, daemon_name):
        self.daemon_name = daemon_name
        self.daemon_func = Daemon.DAEMON_DISPATCHER[daemon_name]

    def fit(self, time_series_list, template_length, max_template_spread):
        print("fitting")
        self.templates_ = Templates(template_length, max_template_spread)
        self.templates_.create_train_set(time_series_list)
        wishart = Wishart(k=self.k, mu=self.mu)
        self.motifs = dict()
        file_path = f"../assets/labels/{int(float(sys.argv[3]))}_{int(sys.argv[5])}_{float(sys.argv[1])}.npz"
        if os.path.exists(file_path):
            save_labels = np.load(file_path)
            z_vectors = self.templates_.train_set
            for template in tqdm(range(z_vectors.shape[0])):
                inf_mask = ~np.isinf(z_vectors[template]).any(axis=1)
                temp_z_v = z_vectors[template][inf_mask]
                wishart.labels_ = save_labels[f"arr_{template}"]
                cluster_labels, _ = np.unique(wishart.labels_[wishart.labels_ > -1], return_counts=True)
                motifs = [temp_z_v[wishart.labels_ == i].mean(axis=0) for i in cluster_labels]
                self.motifs.setdefault(template, []).extend(list(np.array(motifs).reshape(-1, len(motifs[0]))))
        else:
            save_labels = []
            z_vectors = self.templates_.train_set
            for template in tqdm(range(z_vectors.shape[0])):
                inf_mask = ~np.isinf(z_vectors[template]).any(axis=1)
                temp_z_v = z_vectors[template][inf_mask]
                wishart.fit(temp_z_v)
                cluster_labels, _ = np.unique(wishart.labels_[wishart.labels_ > -1], return_counts=True)
                save_labels.append(wishart.labels_)
                motifs = [temp_z_v[wishart.labels_ == i].mean(axis=0) for i in cluster_labels]
                self.motifs.setdefault(template, []).extend(list(np.array(motifs).reshape(-1, len(motifs[0]))))
            np.savez(file_path, *save_labels)
        for template in self.motifs.keys():
            self.motifs[template] = np.array(self.motifs[template])

    def predict(self, time_series, window_index, test_size, eps):
        self.time_series_ = time_series
        self.time_series_.split_train_val_test(window_index, test_size)
        steps = len(self.time_series_.test)
        values = np.array(list(self.time_series_.train) + [np.nan] * steps)
        observation_indexes = self.templates_.observation_indexes
        for step in range(steps):
            test_vectors = values[:len(self.time_series_.train) + step][observation_indexes]
            all_motifs = []
            for template in self.motifs.keys():
                train_truncated = self.motifs[template][:, :-1]
                distance_matrix = calc_distance_matrix([test_vectors[template]], train_truncated)
                distance_mask = distance_matrix < eps
                matched_motifs = self.motifs[template][distance_mask.ravel()]
                if matched_motifs.size > 0:
                    all_motifs.append(matched_motifs)
            motifs_pool = np.vstack(all_motifs) if all_motifs else np.empty((0, 4))

            true_value = self.time_series_.test[step]
            forecast_point = self.freeze_point(motifs_pool, true_value_for_apriori=true_value)

            values[len(self.time_series_.train) + step] = forecast_point
        return values

    def freeze_point(self, motifs_pool, true_value_for_apriori=None):
        if motifs_pool.size == 0:
            return np.nan
        points_pool = motifs_pool[:, -1].reshape(-1, 1)

        if self.daemon_name == 'apriori':
            dbs = DBSCAN(0.01, min_samples=4)
            dbs.fit(points_pool)
            cluster_labels, cluster_sizes = np.unique(dbs.labels_[dbs.labels_ > -1], return_counts=True)

            potential_prediction = np.nan
            if cluster_labels.size > 0:
                mask = (dbs.labels_ == cluster_labels[cluster_sizes.argmax()])
                potential_prediction = points_pool[mask].mean()

            if self.daemon_func(potential_prediction, true_value_for_apriori):
                return np.nan
            else:
                return potential_prediction

        else:
            if self.daemon_func(points_pool):
                return np.nan

            dbs = DBSCAN(0.01, min_samples=4)
            dbs.fit(points_pool)
            cluster_labels, cluster_sizes = np.unique(dbs.labels_[dbs.labels_ > -1], return_counts=True)
            if cluster_labels.size > 0:
                mask = (dbs.labels_ == cluster_labels[cluster_sizes.argmax()])
                return points_pool[mask].mean()
            return np.nan


def calc_distance_matrix(test_vectors, train_vectors):
    return np.squeeze(cdist(test_vectors, train_vectors, 'euclidean'), axis=0)


def predict_handler(gap, test_size_constant, epsilon, ts, tsproc):
    ts_size = len(ts.values)
    window_index = ts_size - (gap + 1) - test_size_constant
    if window_index < 0 or window_index >= ts_size:
        sys.exit(1)
    values = tsproc.predict(ts, window_index, test_size_constant, epsilon)
    real_values = np.array(ts.values[window_index:window_index + test_size_constant])
    pred_values = np.array(values[-test_size_constant:])
    is_np_point = 1 if np.isnan(pred_values[-1]) else 0
    return pred_values[-1], is_np_point, real_values[-1]


def research(r_values, ts_size, how_many_gaps, test_size_constant, dt=0.001, epsilon=0.01,
             template_length_constant=4, template_spread_constant=10):
    list_ts = [TimeSeries("Lorentz", size=size, r=r, dt=dt) for size, r in zip(ts_size, r_values) if size > 0]

    tsproc = TSProcessor()
    tsproc.fit(list_ts[1:], template_length_constant, template_spread_constant)

    all_results = {}
    ts = list_ts[0]

    for daemon_name in Daemon.DAEMON_DISPATCHER.keys():
        print(f"Running predictions for daemon: {daemon_name}")
        tsproc.set_daemon(daemon_name)

        pred_points_values, is_np_points, real_points_values = [], [], []
        for gap in tqdm(range(how_many_gaps), desc=f"Predicting with {daemon_name}"):
            pred_point, is_np_point, real_point = predict_handler(
                gap, test_size_constant, epsilon, ts, tsproc
            )
            if pred_point is not None:
                pred_points_values.append(pred_point)
                is_np_points.append(is_np_point)
                real_points_values.append(real_point)

        rmses = rmse(pred_points_values, real_points_values)
        np_rate = np.mean(is_np_points)
        mapes = mape(pred_points_values, real_points_values)

        all_results[daemon_name] = (rmses, np_rate, mapes)

    return all_results


def main():
    base_size = int(sys.argv[5])
    deviation = float(sys.argv[1])
    prediction_size = int(sys.argv[2])
    added_size = int(float(sys.argv[3]))
    sizes = [base_size, added_size]
    general_size = base_size + added_size
    experiment = sys.argv[4]
    how_many_gaps = 1500

    all_daemon_results = research(
        r_values=[28, 28, 28 + deviation],
        ts_size=np.array([how_many_gaps + 100 + sizes[0]] + list(sizes)),
        how_many_gaps=how_many_gaps,
        test_size_constant=prediction_size
    )

    output_dir = f"/home/ikvasilev/PaTHoP/assets/results/{experiment}"
    os.makedirs(output_dir, exist_ok=True)

    for daemon_name, metrics in all_daemon_results.items():
        rmses, np_points, mapes = metrics
        output_filename = f"{output_dir}/daemons_size_experiment_{daemon_name}.txt"
        with open(output_filename, "a") as f:
            f.write(f"{deviation},{added_size},{prediction_size},{rmses},{np_points},{mapes},{general_size}\n")
        print(f"Results for '{daemon_name}' saved to {output_filename}")


if __name__ == '__main__':
    main()