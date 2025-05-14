# coding: utf-8
import sys

from Research import *


def main():
    deviation = float(sys.argv[1])
    prediction_size = int(sys.argv[2])
    general_size = int(sys.argv[3])
    shares = [float(sys.argv[4]), float(sys.argv[5])]
    experiment = sys.argv[6]
    rmses, np_points, mape = parallel_research(r_values=[28, 28, 28 + deviation],
                                                           ts_size=(np.array(
                                                               [np.array(shares)[0] * general_size + 1250] + list(
                                                                   np.array(shares) * general_size))).astype(
                                                               np.uint64),
                                                           how_many_gaps=1000,
                                                           test_size_constant=prediction_size)
    with open(f"/home/ikvasilev/PaTHoP/results/{experiment}", "a") as f:
        f.write(str(deviation) + "," + str(general_size) + "," + str(prediction_size) + "," + str(rmses) + "," + str(
            np_points) + "," + str(mape) + "," + "\n")


if __name__ == '__main__':
    main()
