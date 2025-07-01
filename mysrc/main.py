# coding: utf-8
import sys

from Research import *


def main():
    deviation = float(sys.argv[1])
    prediction_size = int(sys.argv[2])
    sizes = [10000,int(float(sys.argv[3]))]
    general_size = 10000 + int(float(sys.argv[3]))
    experiment = sys.argv[4]
    rmses, np_points, mape = parallel_research(r_values=[28, 28, 28 + deviation],
                                                           ts_size=np.array([1350] + list(sizes)),
                                                           how_many_gaps=3000,
                                                           test_size_constant=prediction_size)
    with open(f"/home/ikvasilev/PaTHoP/results/{experiment}", "a") as f:
        f.write(str(deviation) + "," + str(int(sys.argv[3])) + "," + str(prediction_size) + "," + str(rmses) + "," + str(
            np_points) + "," + str(mape) + "\n")


if __name__ == '__main__':
    main()