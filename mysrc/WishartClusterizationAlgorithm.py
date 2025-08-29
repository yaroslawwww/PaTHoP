import numpy as np
from sklearn.neighbors import KDTree
from collections import defaultdict
from math import gamma, pi


def volume(radius, dim):
    return pi ** (dim / 2) * radius ** dim / gamma(dim / 2 + 1)


class Wishart:
    def __init__(self, k: int, mu: float):
        self.k, self.mu = k, mu
        self.labels_ = None

    def significant_batch(self, clusters_points, p_values):
        """Векторизованная проверка значимости для нескольких кластеров"""
        significant_mask = np.zeros(len(clusters_points), dtype=bool)
        for i, (cluster_id, points) in enumerate(clusters_points.items()):
            if len(points) < 2:
                continue
            cluster_p = p_values[points]
            significant_mask[i] = (np.max(cluster_p) - np.min(cluster_p)) >= self.mu
        return significant_mask

    def fit(self, x):
        n = len(x)
        x = np.array(x)
        dim = x.shape[1] if x.ndim > 1 else 1

        # 1. Используем KDTree для быстрого вычисления расстояний
        tree = KDTree(x)
        dists, _ = tree.query(x, k=self.k + 1)
        distances_to_k_nearest = dists[:, self.k]

        # 2. Векторизованное вычисление p-values
        volumes = np.array([max(volume(r, dim), 1e-10) for r in distances_to_k_nearest])
        p_values = self.k / (volumes * n)

        # 3. Оптимизированные структуры данных
        labels = np.zeros(n, dtype=int)
        completed = np.zeros(n + 1, dtype=bool)  # Массив вместо словаря
        completed[0] = True  # Шум всегда завершен

        # Массивы для хранения информации о кластерах
        cluster_points = [np.array([], dtype=int) for _ in range(n + 1)]
        cluster_active = np.zeros(n + 1, dtype=bool)

        next_cluster_id = 1

        # Сортируем точки по расстоянию до k-го соседа
        sorted_indices = np.argsort(distances_to_k_nearest)

        # Массивы для хранения обработанных точек
        processed_points = np.array([], dtype=int)
        processed_coords = np.empty((0, dim)) if dim > 1 else np.array([], dtype=float)

        for i in sorted_indices:
            current_point = x[i]

            # Поиск соседей среди обработанных точек
            if len(processed_points) > 0:
                processed_tree = KDTree(processed_coords)
                neighbor_indices = processed_tree.query_radius([current_point],
                                                               r=distances_to_k_nearest[i])[0]
                neighbors = processed_points[neighbor_indices]
            else:
                neighbors = np.array([], dtype=int)

            # Добавляем текущую точку в обработанные
            processed_points = np.append(processed_points, i)
            if dim > 1:
                processed_coords = np.vstack([processed_coords, current_point])
            else:
                processed_coords = np.append(processed_coords, current_point)

            if len(neighbors) == 0:
                # Создаем новый кластер
                labels[i] = next_cluster_id
                cluster_points[next_cluster_id] = np.array([i])
                cluster_active[next_cluster_id] = True
                next_cluster_id += 1
                continue

            # Находим уникальные активные кластеры среди соседей
            neighbor_labels = labels[neighbors]
            unique_labels = np.unique(neighbor_labels)
            active_clusters = [label for label in unique_labels
                               if label != 0 and not completed[label] and cluster_active[label]]

            if len(active_clusters) == 0:
                labels[i] = 0
                continue

            if len(active_clusters) == 1:
                labels[i] = active_clusters[0]
                cluster_points[active_clusters[0]] = np.append(cluster_points[active_clusters[0]], i)
                continue

            # Проверка значимости для нескольких кластеров
            clusters_to_check = {}
            for cluster_id in active_clusters:
                # Находим точки этого кластера среди соседей
                cluster_mask = (neighbor_labels == cluster_id)
                clusters_to_check[cluster_id] = neighbors[cluster_mask]

            # Векторизованная проверка значимости
            significant_flags = self.significant_batch(clusters_to_check, p_values)
            significant_clusters = [cluster_id for cluster_id, is_sig in
                                    zip(active_clusters, significant_flags) if is_sig]

            if len(significant_clusters) > 1:
                labels[i] = 0
                for cluster_id in significant_clusters:
                    completed[cluster_id] = True
                    cluster_active[cluster_id] = False
            else:
                target_cluster = significant_clusters[0] if significant_clusters else active_clusters[0]
                labels[i] = target_cluster
                cluster_points[target_cluster] = np.append(cluster_points[target_cluster], i)

                # Объединение кластеров
                for cluster_id in active_clusters:
                    if cluster_id != target_cluster:
                        # Обновляем метки точек
                        points_to_move = cluster_points[cluster_id]
                        labels[points_to_move] = target_cluster

                        # Объединяем точки кластеров
                        cluster_points[target_cluster] = np.concatenate([
                            cluster_points[target_cluster],
                            points_to_move
                        ])

                        # Деактивируем старый кластер
                        cluster_active[cluster_id] = False
                        completed[cluster_id] = True
                        cluster_points[cluster_id] = np.array([], dtype=int)

        self.labels_ = labels
        return self

