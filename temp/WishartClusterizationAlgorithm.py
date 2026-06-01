import numpy as np
from scipy.special import gamma
from sklearn.neighbors import BallTree


class Wishart:
    def __init__(self, k=11, mu=0.2):
        self.r = k
        self.mu = mu
        self.labels_ = None
        self.clusters_centers_ = None

    def fit(self, z_vectors):
        z_vectors = np.asarray(z_vectors)
        n, dim = z_vectors.shape

        # determine dr(xq) = distance to the sample's r-nearest neighbor;
        tree = BallTree(z_vectors)
        dist, ind = tree.query(z_vectors, k=self.r + 1)
        dr = dist[:, -1]

        # Вычисляем плотности p(x) согласно тексту статьи
        dr_safe = np.maximum(dr, 1e-10)
        volumes = (np.pi ** (dim / 2) * dr_safe ** dim) / gamma(dim / 2 + 1)
        p = self.r / (volumes * n)

        # sort dr(xq) in ascending order;
        sorted_indices = np.argsort(dr)

        w = np.zeros(n, dtype=int)
        completed = np.zeros(n + 1, dtype=bool)

        # Словари для поддержания скорости O(1) и векторизации при слияниях
        cluster_points = {}
        cluster_max_p = {}
        cluster_counter = 1

        in_subgraph = np.zeros(n, dtype=bool)

        # q = 1;
        q = 1

        # for each subgraph G(Zq, Uq):
        # (Итеративное построение графа по мере роста q)
        while q <= n:

            # xq = newly added vertex of the subgraph;
            idx_q = sorted_indices[q - 1]
            in_subgraph[idx_q] = True

            # Получаем соседей, которые уже были добавлены в подграф
            neighbors = ind[idx_q]
            valid_neighbors = neighbors[in_subgraph[neighbors]]
            valid_neighbors = valid_neighbors[valid_neighbors != idx_q]

            connected_labels = w[valid_neighbors]
            unique_labels = np.unique(connected_labels)

            # if xq is not connected to any clusters:
            if len(unique_labels) == 0:
                # start new cluster;
                c1 = cluster_counter
                cluster_counter += 1
                w[idx_q] = c1
                cluster_points[c1] = [idx_q]
                cluster_max_p[c1] = p[idx_q]

            else:
                # if xq connected to the vertices of clusters c1, c2, ..., cl, l >= 1:
                # Сортируем так, чтобы кластер 0 (шум) был первым (c1), как требует логика алгоритма,
                # либо сортируем по плотности для детерминированности.
                c_list = list(unique_labels)
                c_list.sort(key=lambda c: float('inf') if c == 0 else cluster_max_p.get(c, 0), reverse=True)

                c1 = c_list[0]

                # if all clusters are completed:
                if all(c != 0 and completed[c] for c in c_list):
                    # w(xq) = 0;
                    w[idx_q] = 0

                else:
                    # k(μ) = number of significant clusters;
                    k_mu = sum(1 for c in c_list if c != 0 and (cluster_max_p[c] - p[idx_q]) >= self.mu)

                    # if k(μ) > 1 or c1 == 0:
                    if k_mu > 1 or c1 == 0:
                        # w(xq) = 0;
                        w[idx_q] = 0

                        # label significant clusters as completed;
                        for c in c_list:
                            if c != 0 and (cluster_max_p[c] - p[idx_q]) >= self.mu:
                                completed[c] = True

                        # delete labels of insignificant clusters;
                        for c in c_list:
                            if c != 0 and (cluster_max_p[c] - p[idx_q]) < self.mu:
                                if c in cluster_points:
                                    pts = cluster_points[c]
                                    w[pts] = 0  # Мгновенное удаление меток массивом
                                    cluster_points[c] = []

                    else:
                        # merge clusters c2, ..., cn into c1;
                        # w(xq) = c1;
                        w[idx_q] = c1
                        cluster_points[c1].append(idx_q)

                        # set w(xi) = c1 for samples in c2, ..., cn;
                        for c in c_list[1:]:
                            if c != 0 and c in cluster_points:
                                pts = cluster_points[c]
                                if len(pts) > 0:
                                    w[pts] = c1  # Мгновенное обновление меток массивом
                                    cluster_points[c1].extend(pts)
                                    cluster_max_p[c1] = max(cluster_max_p[c1], cluster_max_p[c])
                                    cluster_points[c] = []

            # q = q + 1;
            q = q + 1

        self.labels_ = w

        # Завершающий расчет центров кластеров
        unique_final_labels = np.unique(w)
        self.clusters_centers_ = {}
        for label in unique_final_labels:
            if label == 0:
                continue
            mask = (w == label)
            self.clusters_centers_[label] = z_vectors[mask].mean(axis=0)

        return self