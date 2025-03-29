# coding: utf-8

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
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)

    y_true_masked = y_true[mask]
    y_pred_masked = y_pred[mask]

    if len(y_true_masked) == 0:
        return np.nan

    mape = np.mean(np.abs(y_true_masked - y_pred_masked))
    return mape

def pull_handler(gap_number, window_size,
                 epsilon, ts, ts_processor: TSProcessor):
    ts_size = len(ts.values)
    window_index = ts_size - (gap_number + 1) - window_size
    if window_index > len(ts.values) or window_index < 0:
        raise ValueError("Window index out of range")
    fort, values, affiliation_result = ts_processor.pull(ts, window_index, window_size, epsilon)
    real_values = np.array(ts.values[window_index:window_index + window_size])
    pred_values = np.array(values[-window_size:])
    is_np_point = 1 if np.isnan(pred_values[-1]) else 0
    # mask = ~np.isnan(real_values) & ~np.isnan(pred_values)
    # print(real_values,pred_values)
    # print("res:",abs(real_values[mask]-pred_values[mask]).round(2))
    # print(affiliation_result)
    np.savez(
        '/home/ikvasilev/fast_epsilon_counter/data.npz',
        pred=pred_values,
        real=real_values
    )
    if len(affiliation_result) == 1:
        # случай когда 1 ряд
        return pred_values[-1], is_np_point, 0, real_values[-1]
    elif np.isnan(affiliation_result[1]):
        return pred_values[-1], is_np_point, np.NaN, real_values[-1]
    elif np.isnan(affiliation_result[0]):
        return pred_values[-1], is_np_point, np.NaN, real_values[-1]
    else:
        return pred_values[-1], is_np_point, affiliation_result[1] / (affiliation_result[0] + affiliation_result[1]), \
        real_values[-1]


def parallel_research(r_values=None, ts_size=None, gap_number=0, test_size_constant=50, dt=0.01, epsilon=0.01,
                      template_length_constant=4,
                      template_spread_constant=4):
    pred_points_values = []
    is_np_points = []
    affiliations_list = []
    real_points_values = []
    list_ts = []
    for i, r in enumerate(r_values):
        if ts_size[i] == 0:
            continue
        ts = TimeSeries("Lorentz", size=ts_size[i], r=r, dt=dt)
        list_ts.append(ts)
    tsproc = TSProcessor()
    tsproc.fit(list_ts[1:], template_length_constant, template_spread_constant)
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(pull_handler, gap, test_size_constant, epsilon, list_ts[0], tsproc)
                   for gap in range(gap_number)]
        # for future in futures:
        #     result = future.result()
        #     if result is not None and len(result) > 0:
        #         pred_points_values.append(result[0])
        #         is_np_points.append(result[1])
        #         affiliations_list.append(result[2])
        #         real_points_values.append(result[3])
