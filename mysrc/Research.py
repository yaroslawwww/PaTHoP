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


def research(gap_number, r_values, ts_size, window_size, dt,
             epsilon, template_length_constant, template_spread_constant):
    main_ts_size = int(ts_size[0])
    list_ts = []
    for i, r in enumerate(r_values):
        if ts_size[i] == 0:
            continue
        ts = TimeSeries("Lorentz", size=ts_size[i], r=r, dt=dt)
        list_ts.append(ts)
    window_index = main_ts_size - (gap_number + 1) - window_size
    if window_index > main_ts_size or window_index < 0:
        raise ValueError("Window index out of range")
    tsproc = TSProcessor(list_ts, template_length=template_length_constant,
                         max_template_spread=template_spread_constant,
                         window_index=window_index, test_size=window_size)

    fort, values, affiliation_result = tsproc.pull(epsilon)
    real_values = np.array(list_ts[0].values[window_index:window_index + window_size])
    pred_values = np.array(values[-window_size:])
    is_np_point = 1 if np.isnan(pred_values[-1]) else 0
    mask = ~np.isnan(real_values) & ~np.isnan(pred_values)
    # print(real_values,pred_values)
    # print("res:",abs(real_values[mask]-pred_values[mask]).round(2))
    # print(affiliation_result)
    if len(affiliation_result) == 1:
        #случай когда 1 ряд
        return  pred_values[-1], is_np_point, 0, real_values[-1]
    elif np.isnan(affiliation_result[1]):
        return pred_values[-1], is_np_point, np.NaN, real_values[-1]
    elif np.isnan(affiliation_result[0]):
        return pred_values[-1], is_np_point, np.NaN, real_values[-1]
    else:
        return pred_values[-1], is_np_point, affiliation_result[1] / (affiliation_result[0] + affiliation_result[1]), real_values[-1]


def parallel_research(r_values=None, ts_size=None, gap_number=0, test_size_constant=100, dt=0.01, epsilon=0.01,
                      template_length_constant=4,
                      template_spread_constant=4):
    pred_points_values = []
    is_np_points = []
    affiliations_list = []
    real_points_values = []

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(research, gap, r_values, ts_size, test_size_constant,
                                   dt,
                                   epsilon, template_length_constant, template_spread_constant)
                   for gap in range(gap_number)]
        for future in futures:
            result = future.result()
            if result is not None and len(result) > 0:
                pred_points_values.append(result[0])
                is_np_points.append(result[1])
                affiliations_list.append(result[2])
                real_points_values.append(result[3])
    return  rmse(pred_points_values, real_points_values), np.mean(is_np_points), np.nanmean(affiliations_list)
