import math
import numpy as np
from scipy.special import gamma
from sklearn.neighbors import NearestNeighbors, BallTree
from tqdm import tqdm


class UnionFind:
    def __init__(self):
        self.parent = {}
        # 0 - специальный корень для шума/границ/удаленных кластеров
        self.parent[0] = 0

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
        if self.parent[x] != x:
            # Сжатие путей для гарантии амортизированного O(1)
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]


class Wishart:
    def __init__(self, k, mu):
        self.k = k
        self.mu = mu
        self.labels_ = None
        self.clusters_centers_ = None
        self.center = None

    def fit(self, z_vectors, tqdms=False):
        z_vectors = np.asarray(z_vectors)
        n, dim = z_vectors.shape

        # Инициализация -1 для явного обозначения необработанных точек
        labels = np.full(n, -1, dtype=int)

        uf = UnionFind()
        completed = {}
        cluster_min_p = {}
        cluster_max_p = {}
        cluster_counter = 1

        # 1-2. Находим k-расстояния и сортируем
        # k-расстояние требует k+1 соседей (точка + k соседей)
        knn = NearestNeighbors(n_neighbors=self.k + 1)
        knn.fit(z_vectors)
        k_distances = knn.kneighbors(z_vectors, return_distance=True)[0][:, self.k]

        # Избегаем деления на ноль для совпадающих точек
        r_dist = np.maximum(k_distances, 1e-10)

        # Объем d-мерной сферы: V = π^(d/2) * r^d / Γ(d/2 + 1)
        volumes = (np.pi ** (dim / 2) * r_dist ** dim) / gamma(dim / 2 + 1)
        p_values = self.k / (volumes * n)

        # Сортировка по возрастанию радиуса = по убыванию плотности p(x)
        processed_order = np.argsort(r_dist)
        tree = BallTree(z_vectors)

        if tqdms:
            processed_order = tqdm(processed_order)

        for i in processed_order:
            xi = z_vectors[i:i + 1]

            # Строим подграф: ищем соседей в пределах d_k(x_q)
            neighbors = tree.query_radius(xi, r=r_dist[i])[0]

            active_clusters = set()
            completed_clusters = set()
            has_zero = False  # Добавляем флаг касания фона

            for n_idx in neighbors:
                if n_idx == i:
                    continue

                if labels[n_idx] == -1:
                    continue

                root = uf.find(labels[n_idx])

                if root == 0:
                    has_zero = True  # Фиксируем касание, но не добавляем 0 в active_clusters
                    continue

                if completed.get(root, False):
                    completed_clusters.add(root)
                else:
                    active_clusters.add(root)

            active_clusters = list(active_clusters)

            # Если соседей нет, проверяем, не коснулись ли мы чистого фона
            if len(active_clusters) == 0 and len(completed_clusters) == 0:
                if has_zero:
                    labels[i] = 0
                else:
                    new_label = cluster_counter
                    cluster_counter += 1

                    labels[i] = new_label
                    uf.parent[new_label] = new_label
                    completed[new_label] = False

                    cluster_max_p[new_label] = p_values[i]
                    cluster_min_p[new_label] = p_values[i]
                continue

            if len(active_clusters) == 0 and len(completed_clusters) > 0:
                labels[i] = 0
                continue

            significant_clusters = []
            insignificant_clusters = []
            for root in active_clusters:
                if (cluster_max_p[root] - cluster_min_p[root]) >= self.mu:
                    significant_clusters.append(root)
                else:
                    insignificant_clusters.append(root)

            # Строка 15-18 из статьи: если значимых > 1 ИЛИ есть касание нулевого кластера
            if len(significant_clusters) > 1 or has_zero:
                labels[i] = 0  # Точка становится границей/фоном

                for r in significant_clusters:
                    completed[r] = True

                for r in insignificant_clusters:
                    uf.parent[r] = 0

            else:
                # Обычное слияние (когда significant_clusters <= 1 и has_zero == False)
                if len(significant_clusters) == 1:
                    primary_root = significant_clusters[0]
                else:
                    primary_root = max(active_clusters, key=lambda r: cluster_max_p[r])

                labels[i] = primary_root

                for r in active_clusters:
                    if r != primary_root:
                        uf.parent[r] = primary_root
                        cluster_max_p[primary_root] = max(cluster_max_p[primary_root], cluster_max_p[r])
                        cluster_min_p[primary_root] = min(cluster_min_p[primary_root], cluster_min_p[r])

                cluster_max_p[primary_root] = max(cluster_max_p[primary_root], p_values[i])
                cluster_min_p[primary_root] = min(cluster_min_p[primary_root], p_values[i])

        # Финальное разрешение меток (Flattening)
        for i in range(n):
            if labels[i] > 0:
                labels[i] = uf.find(labels[i])
            elif labels[i] == -1:
                labels[i] = 0  # На всякий случай подчищаем, если вдруг какие-то зависли

        # Извлечение центров (мотивов)
        unique_labels = np.unique(labels)
        self.clusters_centers_ = {}
        for l in unique_labels:
            if l == 0:
                continue
            mask = (labels == l)
            self.clusters_centers_[l] = z_vectors[mask].mean(axis=0)

        self.labels_ = labels
        self.center = z_vectors.mean(axis=0)
        return self