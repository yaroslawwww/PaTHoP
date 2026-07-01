!pip install git+https://github.com/tslearn-team/tslearn.git

!pip install -U numba
os.environ['FORCE_NUMBA_DISABLE'] = '1'


import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree
from scipy.stats import pearsonr, spearmanr, entropy
import jax
import jax.numpy as jnp
import itertools
jax.config.update("jax_default_matmul_precision", "highest")
# --- ИСХОДНЫЕ ПАРАМЕТРЫ (БЕЗ ИЗМЕНЕНИЙ) ---
import numpy as np

from tslearn.metrics import dtw as tslearn_dtw

class Lorentz:
    def __init__(self, s=10, b=8 / 3):
        self.s = s
        self.b = b
        self.r = None
    def X(self, x, y, s):
        return s * (y - x)
    def Y(self, x, y, z, r):
        return -x * z + r * x - y
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
            self.values = x[::divisor]
        else:
            self.values = np.array(array)
import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import norm as sparse_norm

# ВНИМАНИЕ: убедитесь, что функция reconstruct_attractor(x, dim, delay) 
# находится в коде выше этой функции!

# --- МЕТРИКА 2: DTW (АЛГОРИТМ ИЗ ИНТЕРНЕТА) ---
def dtw_distance(s1, s2):
    n, m = len(s1), len(s2)
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(s1[i-1] - s2[j-1])
            last_min = min(dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1])
            dtw_matrix[i, j] = cost + last_min
    return dtw_matrix[n, m]
import numpy as np
from scipy.signal import argrelextrema
from tslearn.metrics import dtw as tslearn_dtw # Обычный DTW
def extract_local_extrema(series: np.ndarray):
    """
    Извлекает локальные экстремумы из временного ряда.
    Возвращает два массива значений: для минимумов и максимумов (в порядке индексов).
    """
    # Локальные максимумы
    max_idx = argrelextrema(series, np.greater)[0]
    maxima_values = series[max_idx] if len(max_idx) > 0 else np.array([])
    # Локальные минимумы
    min_idx = argrelextrema(series, np.less)[0]
    minima_values = series[min_idx] if len(min_idx) > 0 else np.array([])
    # Сортируем по индексам (на всякий случай, хотя argrelextrema уже возвращает отсортированные)
    maxima_values = maxima_values[np.argsort(max_idx)] if len(max_idx) > 0 else np.array([])
    minima_values = minima_values[np.argsort(min_idx)] if len(min_idx) > 0 else np.array([])
    return minima_values, maxima_values
def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Обёртка над tslearn DTW для удобства"""
    a_np = np.asarray(a, dtype=np.float64)
    b_np = np.asarray(b, dtype=np.float64)
    
    # Явно передаем дефолтные булевы аргументы, оборачивая их в bool(),
    # чтобы Numba не распознавала их как константные «Literal»
    return tslearn_dtw(
        a_np, 
        b_np, 
        global_constraint=None, 
        sakoe_chiba_radius=None, 
        itakura_max_slope=None
    )
def ledtw_distance(s1: np.ndarray, s2: np.ndarray) -> float:
    """
    LE-DTW расстояние между двумя временными рядами.
    """
    minima1, maxima1 = extract_local_extrema(s1)
    minima2, maxima2 = extract_local_extrema(s2)
    # Проверка на достаточное количество экстремумов
    if (len(minima1) < 2 or len(minima2) < 2 or
        len(maxima1) < 2 or len(maxima2) < 2):
        return dtw_distance(s1, s2)
    # DTW отдельно для минимумов и максимумов
    dtw_minima = dtw_distance(minima1, minima2)
    dtw_maxima = dtw_distance(maxima1, maxima2)
    return dtw_minima + dtw_maxima
from scipy.stats import gaussian_kde
from scipy.integrate import quad_vec  # или можно вручную по сетке

def kl_divergence(s1, s2, use_returns=False, grid_points=1000, eps=1e-12):
    """
    Вычисляет KL-дивергенцию между распределениями двух временных рядов
    с использованием KDE, следуя методологии Forouzan & Yazdi (2025).

    Параметры:
        s1, s2 : одномерные массивы
        use_returns : если True, то вместо исходных значений используются
                      разности (r_t = x_t - x_{t-1})
        grid_points : количество точек для численного интегрирования
        eps : малое число для избежания log(0)

    Возвращает:
        значение KL-дивергенции D_KL(P || Q)
    """
    if use_returns:
        # Вычисляем доходности (как в статье)
        r1 = np.diff(s1)
        r2 = np.diff(s2)
        data1, data2 = r1, r2
    else:
        data1, data2 = s1, s2

    # Убираем возможные NaN и inf
    data1 = data1[np.isfinite(data1)]
    data2 = data2[np.isfinite(data2)]

    # Построение KDE
    kde1 = gaussian_kde(data1)
    kde2 = gaussian_kde(data2)

    # Область интегрирования: объединение диапазонов с небольшим запасом
    low = min(data1.min(), data2.min()) - 0.5
    high = max(data1.max(), data2.max()) + 0.5
    x_grid = np.linspace(low, high, grid_points)

    # Вычисляем значения плотностей на сетке
    p = kde1.evaluate(x_grid)
    q = kde2.evaluate(x_grid)

    # Избегаем log(0)
    p = np.maximum(p, eps)
    q = np.maximum(q, eps)

    # Подынтегральное выражение
    integrand = p * np.log(p / q)

    # Численное интегрирование методом прямоугольников
    dx = x_grid[1] - x_grid[0]
    kl = np.sum(integrand) * dx

    # Код можно также использовать scipy.integrate.quad_vec для большей точности,
    # но для повторяемости оставим сетку.
    return kl

from sklearn.mixture import GaussianMixture

def mmd_distance(s1, s2, n_components=3, gamma=1.0, reg_covar=1e-6):
    """
    Вычисляет MMD между двумя временными рядами через GMM в фазовом пространстве.
    Следует методологии Sun (2020).
    
    Параметры:
        s1, s2: одномерные массивы
        n_components: число компонент GMM
        gamma: параметр ядра RBF
        reg_covar: регуляризация ковариации
    Возвращает:
        MMD расстояние
    """
    # Построение фазового пространства
    y1 = reconstruct_attractor(s1, DIM, DELAY)
    y2 = reconstruct_attractor(s2, DIM, DELAY)
    
    # Оценка GMM
    gmm1 = GaussianMixture(n_components=n_components, covariance_type='full', 
                            reg_covar=reg_covar, random_state=0).fit(y1)
    gmm2 = GaussianMixture(n_components=n_components, covariance_type='full', 
                            reg_covar=reg_covar, random_state=0).fit(y2)
    
    alpha = gmm1.weights_
    mu1 = gmm1.means_
    cov1 = gmm1.covariances_
    
    beta = gmm2.weights_
    mu2 = gmm2.means_
    cov2 = gmm2.covariances_
    
    d = y1.shape[1]  # размерность
    
    # Функция для вычисления ядра между двумя гауссианами
    def gaussian_kernel(mu_i, cov_i, mu_j, cov_j):
        # Вычисляем K(N_i, N_j) по формуле
        cov_sum = cov_i + cov_j
        # Регуляризация для обратимости
        I = np.eye(d)
        mat = I + gamma * cov_sum
        det = np.linalg.det(mat)
        if det <= 0:
            # Если определитель не положителен, возвращаем большое число
            return 1e12
        inv_mat = np.linalg.inv(mat)
        diff = mu_i - mu_j
        exp_arg = -0.5 * gamma * diff.T @ inv_mat @ diff
        return np.exp(exp_arg) / np.sqrt(det)
    
    # Вычисляем суммы
    K11 = 0.0
    for i in range(n_components):
        for k in range(n_components):
            K11 += alpha[i] * alpha[k] * gaussian_kernel(mu1[i], cov1[i], mu1[k], cov1[k])
    
    K22 = 0.0
    for j in range(n_components):
        for l in range(n_components):
            K22 += beta[j] * beta[l] * gaussian_kernel(mu2[j], cov2[j], mu2[l], cov2[l])
    
    K12 = 0.0
    for i in range(n_components):
        for j in range(n_components):
            K12 += alpha[i] * beta[j] * gaussian_kernel(mu1[i], cov1[i], mu2[j], cov2[j])
    
    mmd_sq = K11 + K22 - 2 * K12
    if mmd_sq < 0:
        mmd_sq = 0.0
    return np.sqrt(mmd_sq)# --- МЕТРИКИ 4, 5, 6: CORRELATIONS & RMSE & LE-DTW ---
def compute_all_metrics(r_ref, r_val, steps):
    ts_ref = TimeSeries(series_type="Lorentz", size=steps, r=r_ref)
    ts_test = TimeSeries(series_type="Lorentz", size=steps, r=r_val)
    x1, x2 = ts_ref.values, ts_test.values
    
   
    # 2. DTW (без изменений)
    m2 = dtw_distance(x1, x2)
   
    # 4. Pearson
    m4, _ = pearsonr(x1, x2)
   
    # 5. Spearman
    m5, _ = spearmanr(x1, x2)
   
    # 6. LE-DTW
    m6 = ledtw_distance(x1, x2)

    # 7. KL Divergence
    m7 = kl_divergence(x1, x2)

    # 8. MMD
    m8 = mmd_distance(x1, x2)
   
    return [0, m2, 0, m4, m5, m6, m7, m8]
if __name__ == '__main__':
    steps = 10000
    r_values = [
        28.0001, 28.01, 27.99, 27.9, 28.1, 30.0, 31.0, 32.0,
        33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0
    ]
   
    header = r"\begin{tabular}{|c|c|c|c|c|c|c|c|c|}"
    hline = r"\hline"
    columns = r"$r$ & DTW & Pearson & Spearman & LE-DTW & KL & MMD \\"
    rows = []
   
    for r in r_values:
        m = compute_all_metrics(28, r, steps)
        row = f"{r:8.4f} & {m[0]:8.3f} & {m[1]:9.2f} & {m[2]:8.3f} & {m[3]:8.3f} & {m[4]:8.3f} & {m[5]:9.2f} & {m[6]:8.3f} & {m[7]:8.3f} \\\\"
        rows.append(row)
   
    print(header)
    print(hline)
    print(columns)
    print(hline)
    for row in rows:
        print(row)
        print(hline)
    print(r"\end{tabular}")
