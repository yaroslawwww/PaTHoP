# coding: utf-8
import sys

import numpy as np
import optuna
from optuna.samplers import NSGAIISampler
from optuna.visualization import plot_pareto_front

from Research import *
from matplotlib import pyplot as plt
import concurrent.futures


def objective(trial):
    # Гиперпараметры
    epsilon = trial.suggest_float("epsilon", 0.001, 0.01)
    threshold = trial.suggest_float("threshold", 0.2, 0.8)
    dbs_eps = trial.suggest_float("dbs_eps", 0.01, 0.02)
    dbs_neighboors = trial.suggest_int("dbs_neighboors", 5, 20)
    mape, rmse, np_metric = validation(10, epsilon, threshold, dbs_neighboors,dbs_eps,window_size=100)
    penalty_factor = 2.5
    if np_metric == 1:
        rmse = 0.5
        mape = 50
    if np_metric > 0.9:
        mape *= penalty_factor
        rmse *= penalty_factor
    return mape, rmse, np_metric


def main():
    study = optuna.create_study(
        directions=["minimize", "minimize", "minimize"],
        sampler=NSGAIISampler()
    )
    study.optimize(objective, n_trials=100)
    # plot_pareto_front(study, targets=lambda t: (t.values[0], t.values[1])).show()
    for i, trial in enumerate(study.best_trials):
        print(f"Решение #{i + 1}:")
        print(f"  Метрики: MAPE={trial.values[0]:.4f}, RMSE={trial.values[1]:.4f}, NP={trial.values[2]:.4f}")
        print(f"  Параметры: epsilon={trial.params['epsilon']:.5f},"
            f" threshold={trial.params['threshold']:.2f},"
            f" dbs_eps={trial.params['dbs_eps']},"
            f"dbs_neighboors={trial.params['dbs_neighboors']}")
        print("-" * 50)


if __name__ == '__main__':
    main()
