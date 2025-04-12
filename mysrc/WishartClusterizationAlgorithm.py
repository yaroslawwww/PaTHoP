# coding: utf-8
import copy
from collections import defaultdict
from itertools import product
from math import gamma

import numpy as np
from scipy.spatial.distance import squareform, pdist
from QuickSelect import QuickSelect
import cupy as cp
from cupyx.scipy.spatial.distance import cdist
from itertools import product
from collections import defaultdict
import math

def volume(radius, dim):
    return np.pi ** (dim / 2) * radius ** dim / gamma(dim / 2 + 1)


class WishartGPU:
    def __init__(self, k, mu):
        self.k, self.mu = k, mu
        self.labels_ = None
        self.clusters_centers_ = None
        self.center = None

    def significant(self, cluster_significances):
        return cluster_significances.max() - cluster_significances.min() >= self.mu

    def merge(self, target_label, source_label, labels):
        mask = (labels == source_label)
        labels[mask] = target_label

    def fit(self, x):
        x_gpu = cp.asarray(x)  # Перенос данных на GPU
        n, dim = x_gpu.shape
        labels = cp.zeros(n, dtype=int)
        completed = {0: False}
        cluster_counter = 1

        # 1. Вычисление k-расстояний на GPU
        pairwise_dist = cdist(x_gpu, x_gpu, 'euclidean')
        cp.fill_diagonal(pairwise_dist, cp.inf)
        k_distances = cp.partition(pairwise_dist, self.k, axis=1)[:, self.k]

        # 2. Вычисление значимостей
        significance_values = self.k / (cp.maximum(
            cp.array([volume(r, dim) for r in k_distances.get()]), 1e-10) * n)

        # 3. Сортировка индексов по k-distance
        processed_order = cp.argsort(k_distances).get()

        # 4. Основной цикл обработки (частично на CPU)
        for i in processed_order:
            neighbors = cp.where(pairwise_dist[i] <= k_distances[i])[0].get()
            neighbors = neighbors[neighbors != i]

            if len(neighbors) == 0:
                labels[i] = cluster_counter
                completed[cluster_counter] = False
                cluster_counter += 1
                continue

            neighbor_labels = cp.unique(labels[neighbors]).get().tolist()
            neighbor_labels = [l for l in neighbor_labels if l != 0]

            if not neighbor_labels:
                continue

            if all(completed.get(l, False) for l in neighbor_labels):
                labels[i] = 0
                continue

            # 5. Проверка значимости кластеров
            significant_clusters = []
            for l in neighbor_labels:
                mask = (labels == l) & (pairwise_dist[i] <= k_distances[i])
                if cp.sum(mask) > 0:
                    cluster_sig = significance_values[cp.where(mask)]
                    if self.significant(cluster_sig):
                        significant_clusters.append(l)

            if len(significant_clusters) > 1:
                labels[i] = 0
                for l in neighbor_labels:
                    completed[l] = (l in significant_clusters)
                continue

            target_label = min(neighbor_labels) if not significant_clusters else significant_clusters[0]
            labels[i] = target_label

            for l in neighbor_labels:
                if l != target_label:
                    self.merge(target_label, l, labels)

        # Перенос результатов обратно на CPU
        self.labels_ = labels.get()
        unique_labels = cp.unique(labels).get()

        self.clusters_centers_ = {}
        x_cpu = x_gpu.get()
        for l in unique_labels:
            self.clusters_centers_[int(l)] = x_cpu[self.labels_ == l].mean(axis=0)

        self.center = x_cpu.mean(axis=0)
        return self

#Test Wishart
# from sklearn.datasets import  make_blobs
# # import matplotlib as plt
# import seaborn as sns
# X,y,centers = make_blobs(n_samples = 100,centers= 10,n_features=2,center_box=(2,100),shuffle=True,return_centers=True,random_state=666)
# X = pd.DataFrame(data = X,columns=["x","y"])
# X.head()
# # plt.figure(figsize=(10,6))
#
# plt.rc('axes',labelsize = 20)
# sns.scatterplot(data = X,x = 'x',y = 'y',hue = y,palette='rocket')
# plt.scatter(centers[:,0],centers[:,1],s=50,c='g',marker = 'o')
# plt.title('Clusters Visualization')
# plt.xlabel('X axis')
# plt.ylabel('Y axis')
# plt.legend()
# plt.show()
# wishart = Wishart(k = 4,mu = 0.2)
# x_list = X.values.tolist()
# # print(x_list)
# wishart.fit(x_list)
# print(wishart.labels_)
# clusters = wishart.labels_
# cluster_centers = wishart.clusters_centers_
# print(cluster_centers)
# from matplotlib import colormaps
# unique_clusters = set(clusters)
# colors = plt.get_cmap('plasma', len(unique_clusters))
# for cluster in unique_clusters:
#     cluster_points = X[clusters == cluster]
#     plt.scatter(cluster_points['x'], cluster_points['y'],
#                 color=colors(cluster), label=f'Cluster {cluster}')
#     plt.scatter(cluster_centers[cluster][0],cluster_centers[cluster][1],color = 'green')
# plt.title('Clusters Visualization')
# plt.xlabel('X axis')
# plt.ylabel('Y axis')
# plt.legend()
# plt.show()
