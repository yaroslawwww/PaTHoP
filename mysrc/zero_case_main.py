# coding: utf-8
import sys

import numpy as np

from Research import *
from matplotlib import pyplot as plt
import concurrent.futures



def process_deviation(deviation, shares, size):
    # Я брал в качестве тестирующей выборки всегда первый ряд.Остальные создавались для тренировочной.
    result = parallel_research(r_values=[28, 28, 28 + deviation],
                               ts_size=(np.array([np.array(shares)[0] * size + 1250] + list(np.array(shares) * size))).astype(np.uint64),
                               gap_number=1000,
                               test_size_constant=50)
    return result[0], result[1], result[2],result[3]

def main():
    shares = [1,0]
    size = 10000
    deviation = float(sys.argv[1])
    rmses, np_points, mean_affiliation,mape = process_deviation(deviation, shares, size)
    with open(f"/home/ikvasilev/fast_epsilon_counter/rmses_np_points_affiliation_final_dbs.txt", "a") as f:
        f.write(str(deviation) + "," + str(size) + "," + str(rmses) + "," + str(np_points) + "," + str(mape) + "," + str(mean_affiliation) + "\n")


if __name__ == '__main__':
    main()