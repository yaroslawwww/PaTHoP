# coding: utf-8
from Predictions import *
import numpy as np
from sklearn.metrics import root_mean_squared_error
from concurrent.futures import ThreadPoolExecutor
import os


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

def research(min_window_index, max_window_index, r_values=None, test_size_constant = 50, dt=0.01, epsilon=0.01,template_length_constant=4, template_spread_constant=4, ts_size=100000):
    divisor = int(0.1 / dt)

    list_ts = []
    rmses = []
    for i, r in enumerate(r_values):
        ts = TimeSeries("Lorentz", size=ts_size, r=r, dt=dt, divisor=divisor)
        list_ts.append(ts)
    for window_index in range(min_window_index, max_window_index, test_size_constant):
        tsproc = TSProcessor(list_ts, template_length=template_length_constant,
                             max_template_spread=template_spread_constant,
                             window_index=window_index, test_size=test_size_constant)

        fort, values = tsproc.pull(epsilon)
        real_values = np.array(list_ts[0].values[window_index:window_index + test_size_constant])
        pred_values = np.array(values[-test_size_constant:])

        mask = ~np.isnan(real_values) & ~np.isnan(pred_values)
        rmses.append(root_mean_squared_error(real_values[mask], pred_values[mask]))
        # print(root_mean_squared_error(real_values[mask], pred_values[mask]))
    return rmses


def threaded_research(r_values=None,gap_number = 0, test_size_constant = 50, dt=0.01, epsilon=0.01, template_length_constant=4,
                      template_spread_constant=4, ts_size=100000):
    divisor = int(0.1 / dt)
    min_window_index = int(ts_size / divisor) - test_size_constant - gap_number * test_size_constant - 1
    num_threads = os.cpu_count()
    max_window_index = int(ts_size / divisor) - test_size_constant - 1
    step = (max_window_index - min_window_index) // num_threads

    ranges = split_range(min_window_index, max_window_index, test_size_constant, num_threads)
    # print("ranges:", ranges)
    # print([[start,end] for start, end in ranges])
    rmses_list = []

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(research, start, end + 1, r_values, test_size_constant, dt,
                                   epsilon, template_length_constant, template_spread_constant, ts_size)
                   for start, end in ranges]

        for future in futures:
            rmses_list.extend(future.result())
    return np.mean(rmses_list)
