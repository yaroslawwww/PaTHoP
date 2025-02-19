# coding: utf-8
import numpy as np
from Research import *
from matplotlib import pyplot as plt
import concurrent.futures


def process_epsilon(epsilon, shares, size, divisor):
    # Каждый процесс выполняет вызов threaded_research с заданным epsilon
    result = threaded_research(r_values=[28, 28 + epsilon],
                               ts_size=(np.array(shares) * size * divisor).astype(np.uint64),
                               gap_number=20,
                               test_size_constant=50)
    return epsilon, result[0], result[1], result[2]


def main():
    divisor = 10
    base_shares = [0.5, 0.5]
    size = 20000
    # Параллелизация цикла по epsilon с использованием процессов
    epsilons_range = np.arange(1, 2, 1)
    rmses = []
    np_points = []
    epsilons = []
    affiliation_list = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_epsilon, epsilon, base_shares, size, divisor)
                   for epsilon in epsilons_range]
        for future in concurrent.futures.as_completed(futures):
            eps, rmse, np_point, affiliation_array = future.result()
            epsilons.append(eps)
            rmses.append(rmse)
            np_points.append(np_point)
            affiliation_list.append(affiliation_array)
            with open(f"./fast_epsilon.txt", "a") as f:
                f.write(
                    str(epsilons_range) + "," + str(np_point) + "," + str(rmse) + "," + str(affiliation_list) + "\n")
    # Сортируем результаты по значению epsilon для корректного построения графика
    sorted_results = sorted(zip(epsilons, rmses, np_points), key=lambda x: x[0])
    epsilons, rmses, np_points = map(list, zip(*sorted_results))

    plt.figure()
    plt.plot(epsilons, rmses)
    plt.xlabel('Возмущение epsilon')
    plt.ylabel('Ошибка')
    plt.title('Зависимость ошибки от epsilon')
    plt.savefig('./graphics/epsilons_and_rmses.png', dpi=300)
    plt.show()

    plt.figure()
    plt.plot(epsilons, np_points)
    plt.xlabel('Возмущение epsilon')
    plt.ylabel('Количество NP точек')
    plt.title('Зависимость количества непредсказываемых точек от epsilon')
    plt.savefig('./graphics/epsilons_and_np_points.png')
    plt.show()


if __name__ == '__main__':
    main()
