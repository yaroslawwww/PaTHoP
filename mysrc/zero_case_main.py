
import numpy as np

from Research import *
from matplotlib import pyplot as plt
import concurrent.futures


def process_epsilon(epsilon, shares, size, divisor):
    result = parallel_research(r_values=[28, 28 + epsilon],
                               ts_size=(np.array(shares) * size * divisor).astype(np.uint64),
                               gap_number=10,
                               test_size_constant=100)
    return epsilon, result[0], result[1], result[2]


def main():
    divisor = 10
    base_shares = [1,0]
    size = 250000
    epsilon = 0

    eps, rmse, np_point, affiliation_array = process_epsilon(epsilon, base_shares, size, divisor)

    with open(f"/home/ikvasilev/fast_epsilon_counter/fast_epsilon.txt", "a") as f:
        f.write(str(epsilon) + "," + str(np_point) + "," + str(rmse) + "," + str(affiliation_array) + "\n")


if __name__ == '__main__':
    main()