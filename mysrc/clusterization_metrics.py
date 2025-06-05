import numpy as np
from scipy.spatial.distance import pdist, cdist, euclidean
from multiprocessing import Pool
from sklearn.neighbors import NearestNeighbors
import warnings


def compute_rmsstd(X, labels):
    """
    Вычисляет RMSSTD (Root Mean Square Standard Deviation)
    Args:
        X: Матрица признаков (n_samples, n_features)
        labels: Метки кластеров (n_samples,)
    Returns:
        RMSSTD: Скалярное значение
    """
    n = X.shape[0]
    d = X.shape[1]
    K = len(np.unique(labels))
    SSW = 0.0

    for k in np.unique(labels):
        cluster_points = X[labels == k]
        if len(cluster_points) > 1:
            cluster_var = np.sum((cluster_points - cluster_points.mean(axis=0)) ** 2)
            SSW += cluster_var

    return np.sqrt(SSW / ((n - K) * d))


def compute_r2(X, labels):
    """
    Вычисляет R-squared (коэффициент детерминации)
    Args:
        X: Матрица признаков (n_samples, n_features)
        labels: Метки кластеров (n_samples,)
    Returns:
        R2: Скалярное значение [0, 1]
    """
    overall_mean = np.mean(X, axis=0)
    SST = np.sum((X - overall_mean) ** 2)

    SSW = 0.0
    for k in np.unique(labels):
        cluster_points = X[labels == k]
        if len(cluster_points) > 0:
            cluster_mean = np.mean(cluster_points, axis=0)
            SSW += np.sum((cluster_points - cluster_mean) ** 2)

    return (SST - SSW) / SST



def compute_calinski_harabasz(X, labels):
    n = len(labels)
    K = len(np.unique(labels))
    overall_mean = np.mean(X, axis=0)

    # SSW (Within-cluster dispersion)
    SSW = 0.0
    # SSB (Between-cluster dispersion)
    SSB = 0.0

    for k in np.unique(labels):
        cluster_points = X[labels == k]
        nk = cluster_points.shape[0]
        if nk > 0:
            cluster_mean = np.mean(cluster_points, axis=0)
            SSW += np.sum((cluster_points - cluster_mean) ** 2)
            SSB += nk * np.sum((cluster_mean - overall_mean) ** 2)

    return (SSB / (K - 1)) / (SSW / (n - K)) if K > 1 else 0


def compute_i_index(X, labels, p=2):
    K = len(np.unique(labels))
    overall_mean = np.mean(X, axis=0)
    E1 = np.sum(np.linalg.norm(X - overall_mean, axis=1) ** 2)

    EK = 0.0
    centroids = []
    for k in np.unique(labels):
        cluster_points = X[labels == k]
        if len(cluster_points) > 0:
            centroid = np.mean(cluster_points, axis=0)
            centroids.append(centroid)
            EK += np.sum(np.linalg.norm(cluster_points - centroid, axis=1) ** 2)

    # Максимальное расстояние между центроидами
    if len(centroids) > 1:
        DK = np.max(cdist(centroids, centroids, 'euclidean'))
    else:
        DK = 0

    return (E1 * DK / (EK * K)) ** p if EK > 0 else 0






def compute_davies_bouldin(X, labels):
    """
    Индекс Дэвиса-Боулдина (ниже = лучше)
    """
    K = len(np.unique(labels))
    if K <= 1:
        return 0

    centroids = []
    s_i = []
    for k in np.unique(labels):
        cluster_points = X[labels == k]
        centroid = np.mean(cluster_points, axis=0)
        centroids.append(centroid)
        dists = cdist(cluster_points, [centroid], 'euclidean')
        s_i.append(np.mean(dists))

    centroids = np.array(centroids)
    db = 0
    for i in range(K):
        r_ij = []
        for j in range(K):
            if i == j:
                continue
            d_ij = euclidean(centroids[i], centroids[j])
            r_ij.append((s_i[i] + s_i[j]) / d_ij)
        db += np.max(r_ij)

    return db / K


def compute_xie_beni(X, labels):
    """
    Индекс Xie-Beni (ниже = лучше)
    """
    n = X.shape[0]
    centroids = []
    SSW = 0.0

    for k in np.unique(labels):
        cluster_points = X[labels == k]
        centroid = np.mean(cluster_points, axis=0)
        centroids.append(centroid)
        SSW += np.sum(cdist(cluster_points, [centroid], 'sqeuclidean'))

    if len(centroids) < 2:
        return np.inf

    # Минимальное расстояние между центроидами
    min_centroid_dist = np.min(pdist(centroids, 'euclidean'))
    return SSW / (n * min_centroid_dist ** 2)


def compute_sd_index(X, labels, alpha=1.0):
    n_samples, n_features = X.shape
    unique_labels = np.unique(labels)
    K = len(unique_labels)

    # Обработка случая с одним кластером
    if K == 1:
        return np.inf  # SD не определен для K=1

    # Вычисление общей дисперсии
    overall_var = np.var(X, axis=0, ddof=1)
    norm_overall = np.linalg.norm(overall_var)

    # Вычисление Scat (компактность)
    scat = 0.0
    for k in unique_labels:
        cluster_points = X[labels == k]
        n_points = len(cluster_points)

        # Для кластера из 0 или 1 точки: дисперсия = 0
        if n_points <= 1:
            cluster_var = np.zeros(n_features)
        else:
            cluster_var = np.var(cluster_points, axis=0, ddof=1)

        # Защита от деления на ноль
        if norm_overall > 1e-10:  # Порог для избежания численных ошибок
            scat += np.linalg.norm(cluster_var) / norm_overall
        else:
            scat += 0  # Если все точки одинаковы

    scat /= K

    # Вычисление центроидов
    centroids = np.array([np.mean(X[labels == k], axis=0) for k in unique_labels])

    # Вычисление попарных расстояний между центроидами
    centroid_distances = pdist(centroids, 'euclidean')

    # Обработка случая с совпадающими центроидами
    if len(centroid_distances) == 0 or np.min(centroid_distances) < 1e-10:
        dis = np.inf
    else:
        D_max = np.max(centroid_distances)
        D_min = np.min(centroid_distances)
        dis = (D_max / D_min) * np.sum(centroid_distances)

    return alpha * scat + dis





# --------------------- Обертка и обработка без параллелизма ---------------------
def compute_metrics_sequential(X, labels):
    """
    Вычисляет все 12 метрик последовательно
    """
    metrics = [
        compute_rmsstd,
        compute_r2,
        compute_calinski_harabasz,
        compute_i_index,
        compute_davies_bouldin,
        compute_xie_beni,
        compute_sd_index
    ]

    results = {}
    for metric_func in metrics:
        metric_name = metric_func.__name__.replace('compute_', '')
        results[metric_name] = metric_func(X, labels)

    return results

