#!/usr/bin/env python3
# coding: utf-8

import os
import sys
import time
import numpy as np
from datetime import datetime
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# Импорты из твоих файлов
from metrics import TimeSeries, chamfer_distance_metric
from candidate_scorer import candidate_score_ts_ts
from evaluation import evaluation

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

# --- ПАРАМЕТРЫ ЭКСПЕРИМЕНТА ---
TOTAL_SERIES = 10000  # Всего генерируем кандидатов
R_MIN = 23.0
R_MAX = 33.0
TARGET_R = 28.0
SERIES_SIZE = 10000  # Размер для вычисления метрик близости
PF_TOP_K = 500  # ПРАВКА НАУЧРУКА: Топ-500 по P-Frob
CHAMFER_TOP_K = 48  # ПРАВКА НАУЧРУКА: Топ-50 по Chamfer
RANDOM_K = 48  # Для сравнения берем 50 рандомных
BASE_SIZE = 10000  # Базовый размер реципиента
AUX_SIZE = 20000  # Размер добавляемого донора
HORIZONS = [10]  # ПРАВКА: Оставляем только h = 10
OUTPUT_DIR = os.path.join(CURRENT_DIR, "results")  # Сохраняем в папку results


def generate_r_values(n, r_min, r_max, seed=42):
    np.random.seed(seed)
    return np.random.uniform(r_min, r_max, n)


def compute_pf_distance_worker(args):
    r_candidate, ts_target = args
    ts_candidate = TimeSeries(series_type="Lorentz", size=SERIES_SIZE, r=r_candidate)

    # Считаем матрицу переходов через функцию из candidate_scorer.py
    # Чтобы не передавать сам объект (может не пиклиться), передаем только значения
    ts_target_dummy = TimeSeries(series_type="Lorentz", size=10, r=TARGET_R)
    ts_target_dummy.values = ts_target
    distance = candidate_score_ts_ts(ts_target_dummy, n_bins=32)
    # ВНИМАНИЕ: Если candidate_score_ts_ts ругается, можно использовать просто candidate_score

    return r_candidate, distance


def compute_chamfer_distance_worker(args):
    r_candidate, target_series_values = args
    ts_candidate = TimeSeries(series_type="Lorentz", size=SERIES_SIZE, r=r_candidate)
    distance = chamfer_distance_metric(target_series_values, ts_candidate.values)
    return r_candidate, distance


def evaluate_donor_worker(args):
    r_donor, main_size, aux_size, pred_len = args
    try:
        # evaluation() возвращает 6 значений: rmse1, np1, mape1 (для 0.1) и rmse2, np2, mape2 (для 0.3)
        rmse_01, np_01, mape_01, rmse_03, np_03, mape_03 = evaluation(
            r_values=[TARGET_R, r_donor],
            ts_sizes=[main_size, aux_size],
            prediction_size=pred_len
        )
        return {
            'r': r_donor,
            'rmse_01': rmse_01, 'np_01': np_01, 'mape_01': mape_01,
            'rmse_03': rmse_03, 'np_03': np_03, 'mape_03': mape_03,
            'success': True
        }
    except Exception as e:
        return {'r': r_donor, 'success': False, 'error': str(e)}


def evaluate_baseline_worker(args):
    main_size, pred_len = args
    try:
        rmse_01, np_01, mape_01, rmse_03, np_03, mape_03 = evaluation(
            r_values=[TARGET_R],
            ts_sizes=[main_size],
            prediction_size=pred_len
        )
        return {
            'size': main_size,
            'rmse_01': rmse_01, 'np_01': np_01, 'mape_01': mape_01,
            'rmse_03': rmse_03, 'np_03': np_03, 'mape_03': mape_03,
            'success': True
        }
    except Exception as e:
        return {'size': main_size, 'success': False, 'error': str(e)}


def run_stage1_pf_selection(r_values, target_values, n_workers):
    print(f"\n=== Stage 1: P-Frob Selection ===")
    args_list = [(r, target_values) for r in r_values]

    # Для ускорения можно заменить compute_pf_distance_worker на вызов candidate_score
    from candidate_scorer import candidate_score
    pf_distances = []

    def pf_wrapper(r):
        return r, candidate_score(TARGET_R, SERIES_SIZE, r, SERIES_SIZE, n_bins=32)

    with Pool(n_workers) as pool:
        results = list(tqdm(pool.imap(pf_wrapper, r_values), total=len(r_values), desc="P-Frob"))

    results.sort(key=lambda x: x[1])
    selected = [r for r, d in results[:PF_TOP_K]]
    print(f"Selected {len(selected)} candidates. Min: {results[0][1]:.4f}, Max: {results[PF_TOP_K - 1][1]:.4f}")
    return selected


def run_stage2_chamfer_selection(r_values, target_values, n_workers):
    print(f"\n=== Stage 2: Chamfer Selection ===")
    args_list = [(r, target_values) for r in r_values]
    with Pool(n_workers) as pool:
        results = list(
            tqdm(pool.imap(compute_chamfer_distance_worker, args_list), total=len(args_list), desc="Chamfer"))

    results.sort(key=lambda x: x[1])
    selected = [r for r, d in results[:CHAMFER_TOP_K]]
    print(f"Selected {len(selected)} donors. Min: {results[0][1]:.4f}, Max: {results[CHAMFER_TOP_K - 1][1]:.4f}")
    return selected


def main():
    print("=" * 80)
    print(f"DONOR SELECTION EXPERIMENT (h={HORIZONS[0]}, Top-{PF_TOP_K}/{CHAMFER_TOP_K})")
    print("=" * 80)

    start_time = time.time()
    n_workers = max(1, cpu_count() - 2)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Target
    target_ts = TimeSeries(series_type="Lorentz", size=SERIES_SIZE, r=TARGET_R)

    # 2. Candidates
    r_candidates = generate_r_values(TOTAL_SERIES, R_MIN, R_MAX)

    # 3. Stage 1 & 2
    pf_selected = run_stage1_pf_selection(r_candidates, target_ts.values, n_workers)
    chamfer_selected = run_stage2_chamfer_selection(pf_selected, target_ts.values, n_workers)

    # 4. Random
    np.random.seed(123)
    remaining_candidates = list(set(r_candidates) - set(chamfer_selected))
    random_donors = np.random.choice(remaining_candidates, RANDOM_K, replace=False)

    # 5. Baselines
    print(f"\n=== Evaluating Baselines ===")
    b10k_result = evaluate_baseline_worker((BASE_SIZE, HORIZONS[0]))
    b30k_result = evaluate_baseline_worker((30000, HORIZONS[0]))

    # 6. Evaluate Donors
    print(f"\n=== Evaluating {len(chamfer_selected)} Selected Donors ===")
    args_sel = [(r, BASE_SIZE, AUX_SIZE, HORIZONS[0]) for r in chamfer_selected]
    with Pool(n_workers) as pool:
        selected_results = list(tqdm(pool.imap(evaluate_donor_worker, args_sel), total=len(args_sel)))

    print(f"\n=== Evaluating {len(random_donors)} Random Donors ===")
    args_rnd = [(r, BASE_SIZE, AUX_SIZE, HORIZONS[0]) for r in random_donors]
    with Pool(n_workers) as pool:
        random_results = list(tqdm(pool.imap(evaluate_donor_worker, args_rnd), total=len(args_rnd)))

    # 7. Запись результатов
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(OUTPUT_DIR, f"final_experiment_h10_{timestamp}.txt")

    with open(out_file, 'w') as f:
        f.write(f"EXPERIMENT RESULTS (h=10)\n")
        f.write(f"Parameters: Base={BASE_SIZE}, Aux={AUX_SIZE}, PF_top={PF_TOP_K}, Chamfer_top={CHAMFER_TOP_K}\n\n")

        f.write("--- BASELINES ---\n")
        f.write(f"B10k (alpha=10/np=0.1): RMSE={b10k_result.get('rmse_01')}, NP={b10k_result.get('np_01')}\n")
        f.write(f"B10k (alpha=3.3/np=0.3): RMSE={b10k_result.get('rmse_03')}, NP={b10k_result.get('np_03')}\n")
        f.write(f"B30k (alpha=10/np=0.1): RMSE={b30k_result.get('rmse_01')}, NP={b30k_result.get('np_01')}\n")
        f.write(f"B30k (alpha=3.3/np=0.3): RMSE={b30k_result.get('rmse_03')}, NP={b30k_result.get('np_03')}\n\n")

        # Сохраним CSV-подобный формат для легкого построения графиков
        f.write("--- SELECTED DONORS ---\n")
        f.write("r,rmse_01,np_01,mape_01,rmse_03,np_03,mape_03\n")
        for res in selected_results:
            if res['success']:
                f.write(
                    f"{res['r']:.6f},{res['rmse_01']},{res['np_01']},{res['mape_01']},{res['rmse_03']},{res['np_03']},{res['mape_03']}\n")

        f.write("\n--- RANDOM DONORS ---\n")
        f.write("r,rmse_01,np_01,mape_01,rmse_03,np_03,mape_03\n")
        for res in random_results:
            if res['success']:
                f.write(
                    f"{res['r']:.6f},{res['rmse_01']},{res['np_01']},{res['mape_01']},{res['rmse_03']},{res['np_03']},{res['mape_03']}\n")

    print(f"\nГотово! Время: {(time.time() - start_time) / 60:.2f} мин. Результаты в {out_file}")


if __name__ == "__main__":
    main()