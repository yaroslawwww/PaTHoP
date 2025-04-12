# coding: utf-8
import copy
from collections import defaultdict
from itertools import product
from math import gamma

import numpy as np
from scipy.spatial.distance import squareform, pdist
from sklearn.neighbors import NearestNeighbors, BallTree
from tqdm import tqdm

from QuickSelect import QuickSelect
# import cupy as cp
# from cupyx.scipy.spatial.distance import cdist
# from itertools import product
from collections import defaultdict
import math

def volume(radius, dim):
    return np.pi ** (dim / 2) * radius ** dim / gamma(dim / 2 + 1)


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        fx = self.find(x)
        fy = self.find(y)
        if fx != fy:
            self.parent[fy] = fx


class Wishart:
    def __init__(self, k, mu):
        self.k, self.mu = k, mu
        self.labels_ = None
        self.clusters_centers_ = None
        self.center = None

    def significant(self, cluster_significances):
        return max(cluster_significances) - min(cluster_significances) >= self.mu

    def fit(self, z_vectors):
        z_vectors = np.asarray(z_vectors)
        n, dim = z_vectors.shape
        labels = np.zeros(n, dtype=int)
        completed = {0: False}
        cluster_counter = 1
        uf = UnionFind()

        # Compute k-distances using NearestNeighbors
        knn = NearestNeighbors(n_neighbors=self.k + 1)
        knn.fit(z_vectors)
        k_distances = knn.kneighbors(z_vectors, return_distance=True)[0][:, self.k]

        # Precompute significance values
        r = k_distances
        volumes = (np.pi ** (dim / 2) * r ** dim) / math.gamma(dim / 2 + 1)
        significance_values = self.k / (volumes * n)

        processed_order = np.argsort(k_distances)

        # Build BallTree for range queries
        tree = BallTree(z_vectors)

        for i in processed_order:
            xi = z_vectors[i:i + 1]
            neighbors = tree.query_radius(xi, r=k_distances[i])[0]
            neighbors = np.setdiff1d(neighbors, [i])  # Exclude self

            neighbor_roots = set()
            cluster_members = {}
            for n in neighbors:
                lbl = labels[n]
                if lbl == 0:
                    continue
                root = uf.find(lbl)
                neighbor_roots.add(root)
                if root not in cluster_members:
                    cluster_members[root] = []
                cluster_members[root].append(n)

            neighbor_roots = [r for r in neighbor_roots if not completed.get(r, False)]

            if len(neighbor_roots) == 0:
                new_label = cluster_counter
                labels[i] = new_label
                uf.union(new_label, new_label)  # Ensure parent exists
                completed[new_label] = False
                cluster_counter += 1
                continue

            if len(neighbor_roots) == 1:
                target = neighbor_roots[0]
                labels[i] = target
                continue

            cluster_significances = {}
            for root in neighbor_roots:
                members = cluster_members[root]
                if not members:
                    continue
                sig_min = np.min(significance_values[members])
                sig_max = np.max(significance_values[members])
                cluster_significances[root] = (sig_min, sig_max)

            significant_clusters = [
                r for r in cluster_significances
                if (cluster_significances[r][1] - cluster_significances[r][0]) >= self.mu
            ]

            if len(significant_clusters) > 1:
                labels[i] = 0
                for r in significant_clusters:
                    completed[r] = True
                continue

            if significant_clusters:
                target = significant_clusters[0]
            else:
                target = min(neighbor_roots, key=lambda r: np.mean(z_vectors[labels == r], axis=0).sum())

            labels[i] = target
            for r in neighbor_roots:
                if r != target:
                    uf.union(target, r)

        # Resolve final labels using union-find
        for i in range(n):
            if labels[i] != 0:
                labels[i] = uf.find(labels[i])

        # Update cluster centers and labels
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
