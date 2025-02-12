# coding: utf-8
from Research import *
from matplotlib import pyplot as plt


def main():
    divisor = 10
    shares = [0.5, 0.5]
    size = 20000
    rmses = []
    shares_list = []
    np_points = []
    epsilons = []
    for epsilon in np.arange(0.1, 20.0, 0.5):
        baranov_result = threaded_research(r_values=[28, 28 + epsilon],
                                           ts_size=(np.array(shares) * size * divisor).astype(np.uint64), gap_number=20,
                                           test_size_constant=50)
        rmses.append(baranov_result[0])
        np_points.append(baranov_result[1])
        epsilons.append(epsilon)
    plt.plot(epsilons, rmses)
    plt.show()
    plt.savefig('epsilons_and_rmses.png',dpi=300)
    plt.plot(epsilons, np_points)
    plt.show()
    plt.savefig('epsilons_and_np_points.png')
    for second_share in np.arange(0.1, 0.9, 0.05):
        epsilon = 1
        shares = [1 - second_share, second_share]
        baranov_result = threaded_research(r_values=[28, 28 + epsilon],
                                           ts_size=(np.array(shares) * size * divisor).astype(np.uint64), gap_number=20,
                                           test_size_constant=50)
        rmses.append(baranov_result[0])
        np_points.append(baranov_result[1])
        shares_list.append(second_share)
    plt.plot(epsilons, rmses)
    plt.show()
    plt.savefig('shares_and_rmses.png',dpi=300)
    plt.plot(epsilons, np_points)
    plt.show()
    plt.savefig('shares_and_np_points.png',dpi=300)


if __name__ == '__main__':
    main()
