# coding: utf-8
import numpy as np
from Research import *
from matplotlib import pyplot as plt
import concurrent.futures
import seaborn as sns

def process_epsilon(epsilon, shares, size, divisor):
    # Каждый процесс выполняет вызов threaded_research с заданным epsilon
    result = threaded_research(r_values=[28, 28 + epsilon],
                               ts_size=(np.array(shares) * size * divisor).astype(np.uint64),
                               gap_number=40,
                               test_size_constant=50)
    return epsilon, result[0], result[1]


def process_share(second_share, size, divisor, epsilon):
    current_shares = [1 - second_share, second_share]
    result = threaded_research(r_values=[28, 28 + epsilon],
                               ts_size=(np.array(current_shares) * size * divisor).astype(np.uint64),
                               gap_number=40,
                               test_size_constant=50)
    return second_share, result[0], result[1]


def process_nested(epsilon, size, shares, divisor):
    result = threaded_research(r_values=[28, 28 + epsilon],
                               ts_size=(np.array(shares) * size * divisor).astype(np.uint64),
                               gap_number=40,
                               test_size_constant=50)
    return epsilon, size, result[0], result[1]


def main():
    divisor = 10
    base_shares = [0.5, 0.5]
    size = 40000
    # Параллелизация цикла по epsilon с использованием процессов
    epsilons_range = np.arange(1, 30.0, 0.2)
    rmses = []
    np_points = []
    epsilons = []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_epsilon, epsilon, base_shares, size, divisor)
                   for epsilon in epsilons_range]
        for future in concurrent.futures.as_completed(futures):
            eps, rmse, np_point = future.result()
            epsilons.append(eps)
            rmses.append(rmse)
            np_points.append(np_point)

    # Сортируем результаты по значению epsilon для корректного построения графика
    sorted_results = sorted(zip(epsilons, rmses, np_points), key=lambda x: x[0])
    epsilons, rmses, np_points = map(list, zip(*sorted_results))

    plt.figure()
    sns.regplot(x=epsilons,y= rmses,order = 3)
    plt.xlabel('Возмущение epsilon')
    plt.ylabel('Ошибка')
    plt.title('Зависимость ошибки от epsilon')
    plt.savefig('./graphics/epsilons_and_rmses.png', dpi=300)
    plt.show()

    plt.figure()
    sns.regplot(x=epsilons,y= np_points,order = 3)
    plt.xlabel('Возмущение epsilon')
    plt.ylabel('Количество NP точек')
    plt.title('Зависимость количества непредсказываемых точек от epsilon')
    plt.savefig('./graphics/epsilons_and_np_points.png')
    plt.show()

    # Параллелизация цикла по доле второго ряда (shares)
    second_shares_range = np.arange(0.1, 0.9, 0.01)
    rmses_shares = []
    np_points_shares = []
    shares_list = []
    epsilon_fixed = 1  # фиксированное значение epsilon

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_share, second_share, size, divisor, epsilon_fixed)
                   for second_share in second_shares_range]
        for future in concurrent.futures.as_completed(futures):
            second_share, rmse, np_point = future.result()
            shares_list.append(second_share)
            rmses_shares.append(rmse)
            np_points_shares.append(np_point)

    # Сортировка по доле второго ряда
    sorted_shares = sorted(zip(shares_list, rmses_shares, np_points_shares), key=lambda x: x[0])
    shares_list, rmses_shares, np_points_shares = map(list, zip(*sorted_shares))

    plt.figure()
    sns.regplot(x=shares_list,y= rmses_shares,order = 3)
    plt.xlabel('Доля второго ряда')
    plt.ylabel('Ошибка')
    plt.title('Зависимость ошибки от доли второго ряда')
    plt.savefig('./graphics/shares_and_rmses.png', dpi=300)
    plt.show()

    plt.figure()
    sns.regplot(x=shares_list,y= np_points_shares,order = 3)
    plt.xlabel('Доля второго ряда')
    plt.ylabel('Количество NP точек')
    plt.title('Зависимость количества непредсказываемых точек от доли второго ряда')
    plt.savefig('./graphics/shares_and_np_points.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    main()
