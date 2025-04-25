# coding: utf-8
import json
import subprocess
import sys
import time

import numpy as np
import optuna
from optuna.samplers import NSGAIISampler
from optuna.visualization import plot_pareto_front

from Research import *
from matplotlib import pyplot as plt
import concurrent.futures


def submit_slurm_job(epsilon, threshold, dbs_neighbours, dbs_eps, result_file):
    sbatch_cmd = f"sbatch optuna_bash {epsilon} {threshold} {dbs_neighbours} {dbs_eps} {result_file}"
    subprocess.run(sbatch_cmd, shell=True, check=True)

def wait_for_result(result_file, timeout=1500 * 60, check_interval=100):
    start_time = time.time()
    while time.time() - start_time < timeout:
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                result = json.load(f)
            os.remove(result_file)
            return result
        time.sleep(check_interval)
    raise optuna.TrialPruned("Timeout waiting for result")

def objective(trial):
    # Гиперпараметры
    epsilon = trial.suggest_float("epsilon", 0.001, 0.01)
    threshold = trial.suggest_float("threshold", 0.2, 0.5)
    dbs_neighbours = trial.suggest_int("dbs_neighboors", 5, 20)
    dbs_eps = trial.suggest_float("dbs_eps", 0.01, 0.03)

    result_file = f"/home/ikvasilev/jsons/result_trial_{trial.number}.json"
    submit_slurm_job(epsilon, threshold, dbs_neighbours, dbs_eps, result_file)
    result = wait_for_result(result_file)
    rmse = result['rmse']
    np_metric = result['np_metric']
    mape = result['mape']
    trial.set_user_attr("RMSE", rmse)
    trial.set_user_attr("NP_metric", np_metric)
    trial.set_user_attr("MAPE", mape)

    penalty = np_metric / 2 if np_metric > 0.83 else 0
    return mape + rmse + penalty
import multiprocessing
import math

# Функция для создания нагрузки на ядро
def cpu_loader():
    while True:
        # Интенсивные вычисления (можно регулировать сложность)
        math.factorial(10000)  # Пример "тяжелой" операции

def main():
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=400,n_jobs = 31)
    best_trial = study.best_trial
    print(f"Лучшая комбинированная ошибка: {best_trial.value:.4f}")
    print("Метрики лучшего trial:")
    print(f"RMSE: {best_trial.user_attrs['RMSE']:.4f}")
    print(f"MAPE: {best_trial.user_attrs['MAPE']:.4f}")
    print(f"NP_metric: {best_trial.user_attrs['NP_metric']:.4f}")


if __name__ == '__main__':
    loader_process = multiprocessing.Process(target=cpu_loader)
    loader_process.daemon = True  # Процесс завершится автоматически с основной программой
    loader_process.start()

    main()

