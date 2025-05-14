# coding: utf-8
import sys

from Predictions import *
import numpy as np
from sklearn.metrics import mean_squared_error
from concurrent.futures import ProcessPoolExecutor
import os
import matplotlib.pyplot as plt


def rmse(y_true, y_pred):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)

    y_true_masked = y_true[mask]
    y_pred_masked = y_pred[mask]

    if len(y_true_masked) == 0:
        return np.nan

    mse = np.mean((y_true_masked - y_pred_masked) ** 2)
    return np.sqrt(mse)


def mape(y_true, y_pred):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    # Маска для исключения NaN
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true_masked = y_true[mask]
    y_pred_masked = y_pred[mask]
    if len(y_true_masked) == 0:
        return 0
    # Проверка на наличие нулей в y_true
    zero_mask = y_true_masked != 0
    if not np.any(zero_mask):
        return np.nan  # Если все значения в y_true равны нулю после маски

    # Вычисление MAPE только для ненулевых значений
    y_true_non_zero = y_true_masked[zero_mask]
    y_pred_non_zero = y_pred_masked[zero_mask]

    # Расчет абсолютной процентной ошибки
    ape = np.abs((y_true_non_zero - y_pred_non_zero) / y_true_non_zero)
    return np.mean(ape)


def predict_handler(gap_number, window_size,
                    epsilon, ts, ts_processor: TSProcessor):
    ts_size = len(ts.values)
    window_index = ts_size - (gap_number + 1) - window_size
    if window_index > len(ts.values) or window_index < 0:
        raise ValueError("Window index out of range")
    mean_affil, values = ts_processor.predict(ts, window_index, window_size, epsilon)
    real_values = np.array(ts.values[window_index:window_index + window_size])
    pred_values = np.array(values[-window_size:])
    is_np_point = 1 if np.isnan(pred_values[-1]) else 0
    # Проверка результата предсказания на глаз
    mask = ~np.isnan(real_values) & ~np.isnan(pred_values)
    print("res:", abs(real_values[mask] - pred_values[mask]).round(2))
    return pred_values[-1], is_np_point, real_values[-1], mean_affil


def parallel_research(r_values, ts_size, how_many_gaps, test_size_constant, dt=0.01, epsilon=0.01,
                      template_length_constant=4,
                      template_spread_constant=10):
    pred_points_values = []
    is_np_points = []
    real_points_values = []
    list_ts = []
    mean_affils = []
    for i, r in enumerate(r_values):
        if ts_size[i] == 0:
            continue
        ts = TimeSeries("Lorentz", size=ts_size[i], r=r, dt=dt)
        list_ts.append(ts)
    tsproc = TSProcessor()
    tsproc.fit(list_ts[1:], template_length_constant, template_spread_constant)
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(predict_handler, gap, test_size_constant, epsilon, list_ts[0], tsproc)
                   for gap in range(how_many_gaps)]
        for future in futures:
            result = future.result()
            if result is not None and len(result) > 0:
                pred_points_values.append(result[0])
                is_np_points.append(result[1])
                real_points_values.append(result[2])
                mean_affils.append(result[3])
    return rmse(pred_points_values, real_points_values), np.mean(is_np_points), mape(pred_points_values,
                                                                                     real_points_values), np.mean(
        mean_affils)
