import copy
from collections import defaultdict
from itertools import product
from math import gamma

import numpy as np
from scipy.spatial.distance import squareform, pdist
from QuickSelect import QuickSelect


def volume(radius, dim):
    return np.pi ** (dim / 2) * radius ** dim / gamma(dim / 2 + 1)


class Wishart:
    def __init__(self, k: int, mu: float):
        self.k, self.mu = k, mu
        self.labels_ = None
        self.distances = None
        self.clusters_ = None
        self.clusters_centers_ = None
        self.center = None
    def significant(self, cluster, significance_values_of_point):
        max_diff = max(abs(significance_values_of_point[i] - significance_values_of_point[j]) for i, j in
                       product(cluster, cluster))
        return max_diff >= self.mu

    def merge(self, first_cl, second_cl, clusters: defaultdict, labels):
        for i in clusters[second_cl]:
            labels[i] = first_cl

    def fit(self, x):
        n = len(x)
        x = np.array(x)
        if isinstance(x[0], list):
            dim = len(x[0])
        else:
            dim = 1
        self.distances = squareform(pdist(x))
        distances = copy.deepcopy(self.distances)
        distances_to_k_nearest_neighbours = []
        if len(distances) < self.k:
            print("insufficient number of neighbors")
            return
        for i in range(n):
            distances_to_k_nearest_neighbours.append(QuickSelect(distances[i], self.k))
        #далее присутствует костыль без которого не работает решение(max)
        significance_of_each_point = [self.k / (max(volume(distances_to_k_nearest_neighbours[i], dim),0.000000001) * n) for i in
                                      range(n)]
        labels = [0] * n
        completed = {0: False}
        number_of_clusters = 1
        vertices = set()
        for d, i in sorted(zip(distances_to_k_nearest_neighbours, range(n))):
            neighbours = set()
            neighbour_labels = set()
            clusters = defaultdict(list)
            for j in vertices:
                if self.distances[i][j] <= distances_to_k_nearest_neighbours[i]:
                    neighbours.add(j)
                    neighbour_labels.add(labels[j])
                    clusters[labels[j]].append(j)
            vertices.add(i)
            if len(neighbours) == 0:
                labels[i] = number_of_clusters
                completed[number_of_clusters] = False
                number_of_clusters += 1
                continue
            if len(neighbour_labels) == 1:
                cluster = next(iter(neighbour_labels))
                if completed[cluster]:
                    labels[i] = 0
                else:
                    labels[i] = cluster
                continue
            if all(completed[wj] for wj in neighbour_labels):
                labels[i] = 0
                continue
            significant_clusters = set(cluster for cluster in neighbour_labels if
                                       self.significant(clusters[cluster], significance_of_each_point))
            if len(significant_clusters) > 1:
                labels[i] = 0
                for cluster in neighbour_labels:
                    if cluster in significant_clusters:
                        completed[cluster] = (cluster != 0)
                    else:
                        self.merge(0, cluster, clusters, labels)
                continue
            if len(significant_clusters) == 0:
                most_suitable_cluster = next(iter(neighbour_labels))
            else:
                most_suitable_cluster = next(iter(significant_clusters))
            labels[i] = most_suitable_cluster
            for cluster in neighbour_labels:
                self.merge(most_suitable_cluster, cluster, clusters, labels)
        self.labels_ = np.array(labels)
        self.clusters_ = defaultdict(list)
        self.clusters_centers_ = defaultdict(list)
        for i, label in enumerate(self.labels_):
            self.clusters_[label].append(x[i])
        for cl in self.clusters_.keys():
            t = x[self.labels_ == cl]
            self.clusters_centers_[cl] = x[self.labels_ == cl].mean(axis = 0)
        self.center = x.mean(axis = 0)
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
