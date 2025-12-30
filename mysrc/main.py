# coding: utf-8
import os
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
from evaluation import *
from candidate_scorer import *
from scipy.spatial import KDTree

# ==================== Параметры ====================
R_TARGET = 28.0
SIZE_TARGET = 10000
SIZE_CANDIDATE = 20000
R_DISTANCE = 5.0
N_CANDIDATES = 10000
N_SELECT_PF = 480          # промежуточный отбор по Перрону-Фробениусу
N_FINAL = 48               # финальное количество в каждой группе
OUTPUT_DIR = "/home/ikvasilev/PaTHoP/assets/results/manna2"
SEED = 666
DIM = 3
DELAY = 10

# ==================== Метрики ====================
def reconstruct_attractor(x: np.ndarray, dim: int, delay: int) -> np.ndarray:
    n = len(x)
    max_idx = n - (dim - 1) * delay
    if max_idx <= 0:
        raise ValueError("Ряд слишком короткий.")
    attractor = np.column_stack([x[i:i + max_idx] for i in range(0, dim * delay, delay)])
    return attractor


def chamfer_distance_metric(x1: np.ndarray, x2: np.ndarray) -> float:
    y1 = reconstruct_attractor(x1, DIM, DELAY)
    y2 = reconstruct_attractor(x2, DIM, DELAY)
    tree1 = KDTree(y1)
    tree2 = KDTree(y2)
    d1, _ = tree2.query(y1, k=1, workers=-1)
    d2, _ = tree1.query(y2, k=1, workers=-1)
    return np.mean(d1 ** 2) + np.mean(d2 ** 2)


def perron_frobenius_score(ts1, ts2, n_bins=32):
    s1, s2 = ts1.values, ts2.values
    global_min = min(s1.min(), s2.min())
    global_max = max(s1.max(), s2.max())
    bins = np.linspace(global_min, global_max, n_bins + 1)
    d1 = np.clip(np.digitize(s1, bins) - 1, 0, n_bins - 1)
    d2 = np.clip(np.digitize(s2, bins) - 1, 0, n_bins - 1)

    def build_T(d):
        T = np.zeros((n_bins, n_bins))
        for i in range(len(d) - 1):
            T[d[i], d[i + 1]] += 1
        row_sums = T.sum(axis=1, keepdims=True)
        return np.where(row_sums > 0, T / row_sums, 0)

    return np.linalg.norm(build_T(d1) - build_T(d2), 'fro')


def calculate_best_predictability(main_attr, cand_attr, bins=8):
    def quantize(data, n_bins):
        q = np.zeros_like(data, dtype=int)
        for j in range(data.shape[1]):
            edges = np.histogram_bin_edges(data[:, j], bins=n_bins)
            q[:, j] = np.clip(np.digitize(data[:, j], edges) - 1, 0, n_bins - 1)
        return q

    src_q = quantize(main_attr, bins)
    trg_q = quantize(cand_attr, bins)
    n_states = bins ** main_attr.shape[1]
    src_states = np.ravel_multi_index(src_q.T, [bins] * main_attr.shape[1])
    trg_states = np.ravel_multi_index(trg_q.T, [bins] * cand_attr.shape[1])

    trans = np.zeros((n_states, n_states), dtype=int)
    for i in range(len(src_states) - 1):
        trans[src_states[i], src_states[i + 1]] += 1

    probs = np.where(trans.sum(axis=1, keepdims=True) > 0,
                     trans / trans.sum(axis=1, keepdims=True), 0)
    policy = np.argmax(probs, axis=1)

    correct = sum(policy[trg_states[i]] == trg_states[i + 1] for i in range(len(trg_states) - 1))
    return correct / (len(trg_states) - 1) if len(trg_states) > 1 else 0.0


# ==================== Обработка одного кандидата ====================
def process_candidate(r):
    main_ts = TimeSeries(series_type="Lorentz", size=SIZE_TARGET, r=R_TARGET)
    cand_ts = TimeSeries(series_type="Lorentz", size=SIZE_CANDIDATE, r=r)

    chamfer = chamfer_distance_metric(main_ts.values, cand_ts.values)
    pf = perron_frobenius_score(main_ts, cand_ts)
    best = calculate_best_predictability(
        reconstruct_attractor(main_ts.values, DIM, DELAY),
        reconstruct_attractor(cand_ts.values, DIM, DELAY)
    )
    return chamfer, pf, best


# ==================== Основная функция ====================
def main():
    np.random.seed(SEED)
    # candidate_r_values = np.random.uniform(R_TARGET - R_DISTANCE, R_TARGET + R_DISTANCE, N_CANDIDATES)
    #
    # print("Вычисление метрик для 10000 кандидатов...")
    # with Pool(48) as pool:
    #     results = list(tqdm(pool.imap(process_candidate, candidate_r_values),
    #                         total=N_CANDIDATES, desc="Оценка кандидатов"))
    #
    # chamfer_scores, pf_scores, best_scores = map(np.array, zip(*results))
    #
    # os.makedirs(OUTPUT_DIR, exist_ok=True)
    #
    # # 1. 48 лучших по BEST (чем выше — тем лучше)
    # idx_best = np.argsort(best_scores)[-N_FINAL:][::-1]
    # r_best = candidate_r_values[idx_best]
    #
    # # 2. Сначала топ-480 по PF (меньше — лучше) → потом из них топ-48 по Chamfer (меньше — лучше)
    # idx_pf_top480 = np.argsort(pf_scores)[:N_SELECT_PF]
    # chamfer_in_pf = chamfer_scores[idx_pf_top480]
    # idx_chamfer_best_in_pf = np.argsort(chamfer_in_pf)[:N_FINAL]
    # r_pf_chamfer = candidate_r_values[idx_pf_top480][idx_chamfer_best_in_pf]
    #
    # # 3. 48 случайных
    # idx_random = np.random.choice(N_CANDIDATES, N_FINAL, replace=False)
    # r_random = candidate_r_values[idx_random]

    # Сохранение только финальных 48 r-значений
    file_best = os.path.join(OUTPUT_DIR, "r_best_top48.txt")
    file_pf_chamfer = os.path.join(OUTPUT_DIR, "r_pf_chamfer_top48.txt")
    file_random = os.path.join(OUTPUT_DIR, "r_random_48.txt")

    # np.savetxt(file_best, r_best, fmt="%.12f")
    # np.savetxt(file_pf_chamfer, r_pf_chamfer, fmt="%.12f")
    # np.savetxt(file_random, r_random, fmt="%.12f")

    print("\nСохранены финальные группы (по 48 r):")
    print(f"   BEST:           {file_best}")
    print(f"   PF → Chamfer:   {file_pf_chamfer}")
    print(f"   Random:         {file_random}")

    # Запуск 9 задач (3 группы × 3 длины предсказания)
    prediction_lengths = [1, 10, 20]
    groups = [
        ("best_top48", file_best),
        ("pf_chamfer", file_pf_chamfer),
        ("random", file_random),
    ]

    print("\nЗапуск задач на кластере...")
    for group_name, r_file_path in groups:
        for pred_len in prediction_lengths:
            cmd = (
                f"sbatch -A proj_1716 "
                f"./subbash {R_TARGET} {SIZE_TARGET} {SIZE_CANDIDATE} "
                f"{group_name} {pred_len} \"{r_file_path}\""
            )
            print(f"   {cmd}")
            os.system(cmd)

    print("\nВсе 9 задач отправлены в очередь.")


if __name__ == "__main__":
    main()