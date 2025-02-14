# coding: utf-8
from Predictions import *
import numpy as np
from sklearn.metrics import mean_squared_error
from concurrent.futures import ProcessPoolExecutor
import os

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))
def split_range(min_w, max_w, step, num_threads):

    total_values = (max_w - min_w) // step + 1

    values_per_thread = total_values // num_threads
    remainder = total_values % num_threads

    ranges = []
    start = min_w

    for i in range(num_threads):
        if i < remainder:
            end = start + (values_per_thread + 1) * step
        else:
            end = start + values_per_thread * step
        end = min(end, max_w + step)

        ranges.append((start, end - step))

        start = end

    return ranges

def research(min_window_index, max_window_index, r_values=None, ts_size=None, test_size_constant = 50, dt=0.01, epsilon=0.01,template_length_constant=4, template_spread_constant=4):
    divisor = int(0.1 / dt)
    main_ts_size = int(ts_size[0] / divisor)
    if min_window_index > max_window_index or max_window_index + test_size_constant >= main_ts_size:
        return []
    list_ts = []
    rmses = []
    np_points = []
    for i, r in enumerate(r_values):
        ts = TimeSeries("Lorentz", size=ts_size[i], r=r, dt=dt, divisor=divisor)
        list_ts.append(ts)
    for window_index in range(min_window_index, max_window_index, test_size_constant):
        tsproc = TSProcessor(list_ts, template_length=template_length_constant,
                             max_template_spread=template_spread_constant,
                             window_index=window_index, test_size=test_size_constant)

        fort, values = tsproc.pull(epsilon)
        real_values = np.array(list_ts[0].values[window_index:window_index + test_size_constant])
        pred_values = np.array(values[-test_size_constant:])
        if len(real_values) != len(pred_values):
            return rmses
        mask = ~np.isnan(real_values) & ~np.isnan(pred_values)
        if np.all(np.isnan(real_values)) or np.all(np.isnan(pred_values)):
            continue
        rmses.append(rmse(real_values[mask], pred_values[mask]))
        np_points.append(test_size_constant-len(pred_values[mask]))
        # print(mean_squared_error(real_values[mask], pred_values[mask]))
        # print(np_points[-1])
    return rmses,np_points


def threaded_research(r_values=None,ts_size = None,gap_number = 0, test_size_constant = 50, dt=0.01, epsilon=0.01, template_length_constant=4,
                      template_spread_constant=4):
    divisor = int(0.1 / dt)
    main_ts_size = int(ts_size[0] / divisor)
    min_window_index = main_ts_size - test_size_constant - gap_number * test_size_constant - 1
    num_threads = os.cpu_count()
    max_window_index = main_ts_size - test_size_constant - 1
    step = (max_window_index - min_window_index) // num_threads

    ranges = split_range(min_window_index, max_window_index, test_size_constant, num_threads)
    # print("ranges:", ranges)
    # print([[start,end] for start, end in ranges])
    rmses_list = []
    np_points_list = []
    with ProcessPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(research, start, end + 1, r_values, ts_size, test_size_constant, dt,
                                   epsilon, template_length_constant, template_spread_constant)
                   for start, end in ranges]

        for future in futures:
            result = future.result()
            if result is not None and len(result) > 0:
                rmses_list.extend(result[0])
                np_points_list.extend(result[1])
    return np.mean(rmses_list),np.mean(np_points_list)
