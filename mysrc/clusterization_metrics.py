import numpy as np
from scipy.spatial.distance import cdist, pdist
from sklearn.neighbors import NearestNeighbors


def compute_rmsstd(X, labels):
    """
    Вычисляет сумму квадратов внутрикластерных расстояний (WSS)
    """
    unique_labels = np.unique(labels)
    wss = 0.0
    for label in unique_labels:
        cluster_points = X[labels == label]
        if len(cluster_points) == 0:
            continue
        centroid = np.mean(cluster_points, axis=0)
        distances = cdist(cluster_points, [centroid], 'euclidean')
        wss += np.sum(distances ** 2)
    return wss


def compute_r2(X, labels):
    """
    Вычисляет R^2 как 1 - WSS/TSS
    """
    n_samples = X.shape[0]
    overall_mean = np.mean(X, axis=0)
    distances_to_mean = cdist(X, [overall_mean], 'euclidean')
    tss = np.sum(distances_to_mean ** 2)

    unique_labels = np.unique(labels)
    wss = 0.0
    for label in unique_labels:
        cluster_points = X[labels == label]
        if len(cluster_points) == 0:
            continue
        centroid = np.mean(cluster_points, axis=0)
        distances = cdist(cluster_points, [centroid], 'euclidean')
        wss += np.sum(distances ** 2)

    if tss == 0:
        return 1.0
    else:
        return 1 - wss / tss


def compute_modified_hubert(X, labels):
    """
    Предполагается, что это Silhouette score (требуется уточнение)
    """
    return compute_silhouette(X, labels)


def compute_calinski_harabasz(X, labels):
    """
    Вычисляет индекс Calinski-Harabasz (выше = лучше)
    """
    n_samples = X.shape[0]
    unique_labels = np.unique(labels)
    k = len(unique_labels)
    if k < 2:
        return 0

    overall_mean = np.mean(X, axis=0)

    bss = 0.0
    for label in unique_labels:
        cluster_points = X[labels == label]
        n_cluster = len(cluster_points)
        if n_cluster == 0:
            continue
        centroid = np.mean(cluster_points, axis=0)
        distance = np.linalg.norm(centroid - overall_mean)
        bss += n_cluster * distance ** 2

    wss = 0.0
    for label in unique_labels:
        cluster_points = X[labels == label]
        if len(cluster_points) == 0:
            continue
        centroid = np.mean(cluster_points, axis=0)
        distances = cdist(cluster_points, [centroid], 'euclidean')
        wss += np.sum(distances ** 2)

    if wss == 0:
        return float('inf')
    else:
        return (bss / wss) * (n_samples - k) / (k - 1)


def compute_i_index(X, labels, p=2):
    """
    Предполагается, что это Xie-Beni index (требуется уточнение)
    """
    return compute_xie_beni(X, labels)


def compute_dunn_index(X, labels):
    """
    Вычисляет индекс Dunn (выше = лучше)
    """
    unique_labels = np.unique(labels)
    k = len(unique_labels)
    if k < 2:
        return 0

    D = cdist(X, X, 'euclidean')

    min_inter = float('inf')
    for i in range(k):
        for j in range(i + 1, k):
            cluster_i = X[labels == unique_labels[i]]
            cluster_j = X[labels == unique_labels[j]]
            dists = cdist(cluster_i, cluster_j, 'euclidean')
            min_d = np.min(dists)
            if min_d < min_inter:
                min_inter = min_d

    if min_inter == 0:
        return 0

    max_intra = 0
    for label in unique_labels:
        cluster_points = X[labels == label]
        if len(cluster_points) < 2:
            continue
        dists = pdist(cluster_points, 'euclidean')
        max_d = np.max(dists)
        if max_d > max_intra:
            max_intra = max_d

    if max_intra == 0:
        return float('inf')
    else:
        return min_inter / max_intra


def compute_silhouette(X, labels):
    """
    Вычисляет средний силуэтный индекс для всех точек
    """
    n = len(labels)
    K = len(np.unique(labels))
    if K <= 1:
        return 0

    D = cdist(X, X, 'euclidean')

    silhouette_vals = np.zeros(n)
    for i in range(n):
        cluster_i = labels[i]

        mask_same = labels == cluster_i
        a_i = np.mean(D[i, mask_same]) if np.sum(mask_same) > 1 else 0

        b_vals = []
        for k in np.unique(labels):
            if k == cluster_i:
                continue
            mask_other = labels == k
            dist = np.mean(D[i, mask_other])
            b_vals.append(dist)

        b_i = np.min(b_vals) if b_vals else 0
        s_i = (b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) > 0 else 0
        silhouette_vals[i] = s_i

    return np.mean(silhouette_vals)


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
            d_ij = np.linalg.norm(centroids[i] - centroids[j])
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

    min_centroid_dist = np.min(pdist(centroids, 'euclidean'))
    return SSW / (n * min_centroid_dist ** 2)


def compute_sd_index(X, labels, alpha=1.0):
    """
    SD индекс (ниже = лучше)
    """
    n_samples, n_features = X.shape
    K = len(np.unique(labels))

    overall_var = np.var(X, axis=0, ddof=1)
    scat = 0.0
    for k in np.unique(labels):
        cluster_points = X[labels == k]
        cluster_var = np.var(cluster_points, axis=0, ddof=1)
        scat += np.linalg.norm(cluster_var) / np.linalg.norm(overall_var)
    scat /= K

    centroids = []
    for k in np.unique(labels):
        centroids.append(np.mean(X[labels == k], axis=0))
    centroids = np.array(centroids)

    D_max = np.max(pdist(centroids, 'euclidean'))
    D_min = np.min(pdist(centroids, 'euclidean'))
    dis = (D_max / D_min) * np.sum(pdist(centroids, 'euclidean')) if D_min > 0 else 0

    return alpha * scat + dis


def compute_sdbw_index(X, labels, radius_scale=0.1):
    """
    S_Dbw индекс (ниже = лучше)
    """
    n_samples, n_features = X.shape
    K = len(np.unique(labels))
    if K <= 1:
        return 0

    centroids = []
    variances = []
    for k in np.unique(labels):
        cluster_points = X[labels == k]
        centroid = np.mean(cluster_points, axis=0)
        centroids.append(centroid)
        variances.append(np.var(cluster_points, axis=0, ddof=1))

    centroids = np.array(centroids)
    variances = np.array(variances)

    overall_var = np.var(X, axis=0, ddof=1)
    scat = np.sum(np.linalg.norm(variances, axis=1)) / (K * np.linalg.norm(overall_var))

    std_dev = np.sqrt(np.mean(np.linalg.norm(variances, axis=1)))
    radius = radius_scale * std_dev

    dens_bw = 0.0
    for i in range(K):
        for j in range(i + 1, K):
            midpoint = (centroids[i] + centroids[j]) / 2
            dists = cdist(X, [midpoint], 'euclidean').flatten()
            density_ij = np.sum(dists <= radius)

            density_i = np.sum(cdist(X[labels == i], [centroids[i]], 'euclidean') <= radius)
            density_j = np.sum(cdist(X[labels == j], [centroids[j]], 'euclidean') <= radius)
            density_clusters = density_i + density_j

            if density_clusters > 0:
                dens_bw += density_ij / density_clusters

    dens_bw = dens_bw * 2 / (K * (K - 1)) if K > 1 else 0
    return scat + dens_bw


def compute_cvnn2_index(X, labels, k_neighbors=5):
    """
    CVNN2 индекс (ниже = лучше)
    """
    n_samples = X.shape[0]
    if n_samples < k_neighbors + 1:
        return 0

    knn = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(X)
    _, indices = knn.kneighbors(X)

    neighbor_indices = indices[:, 1:]
    same_label_count = 0

    for i in range(n_samples):
        neighbor_labels = labels[neighbor_indices[i]]
        same_label_count += np.sum(neighbor_labels == labels[i])

    return 1 - same_label_count / (n_samples * k_neighbors)

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

