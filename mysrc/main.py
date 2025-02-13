# coding: utf-8
import numpy as np
from Research import *
from matplotlib import pyplot as plt
import concurrent.futures
import seaborn as sns

def process_size(epsilon,sizes, divisor):
    # Каждый процесс выполняет вызов threaded_research с заданным epsilon
    result = threaded_research(r_values=[28, 28 + epsilon],
                               ts_size=sizes,
                               gap_number=40,
                               test_size_constant=50)
    return epsilon,sizes[1], result[0], result[1]



def main():
    divisor = 10

    #
    # Параллелизация цикла по размеру 2 ряда
    second_size_range = [10000,10**5,10**6,10**7]
    epsilon_range = [1]
    rmses_sizes = []
    np_points_sizes = []
    sizes_list = []
    epsilon_list = []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_size, epsilon,[10000,second_size], divisor)
                   for second_size in second_size_range for epsilon in epsilon_range]
        for future in concurrent.futures.as_completed(futures):
            epsilon,second_sizes, rmse, np_point = future.result()
            epsilon_list.append(epsilon)
            sizes_list.append(second_sizes)
            rmses_sizes.append(rmse)
            np_points_sizes.append(np_point)

    # Сортировка по доле второго ряда
    sorted_sizes = sorted(zip(sizes_list, rmses_sizes, np_points_sizes), key=lambda x: x[0])
    sizes_list, rmses_sizes, np_points_sizes = map(list, zip(*sorted_sizes))
    plt.figure()
    plt.plot(x=sizes_list, y=np_points_sizes)
    plt.xscale('log')
    plt.xlabel('Длина добавленного ряда')
    plt.ylabel('Количество NP точек')
    plt.title('Зависимость количества непредсказываемых точек от size')
    plt.savefig('./graphics/sizes_and_np_points.png')
    plt.show()

    plt.figure()
    plt.plot(x=sizes_list, y=rmses_sizes)
    plt.xscale('log')
    plt.xlabel('Длина добавленного ряда')
    plt.ylabel('RMSE')
    plt.title('Зависимость ошибки от size')
    plt.savefig('./graphics/sizes_and_rmses.png')
    plt.show()

if __name__ == '__main__':
    main()
