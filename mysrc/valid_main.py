# coding: utf-8
import json

from Research import *


def main():
    epsilon = float(sys.argv[1])
    threshold = float(sys.argv[2])
    dbs_neighbours = int(sys.argv[3])
    dbs_eps = float(sys.argv[4])
    result_file = str(sys.argv[5])
    rmse, np_metric, mape = validation(10, epsilon, threshold, dbs_neighbours, dbs_eps, window_size=100)
    print("current_metrics ", rmse, np_metric, mape)
    with open(result_file, 'w') as file:
        json.dump({'rmse': rmse, 'np_metric': np_metric, 'mape': mape}, file)

if __name__ == '__main__':
    main()
