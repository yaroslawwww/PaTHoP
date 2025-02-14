# coding: utf-8
import numpy as np
from Research import *
from matplotlib import pyplot as plt
import concurrent.futures
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata


def process_size(epsilon, sizes, divisor):
    # Каждый процесс выполняет вызов threaded_research с заданным epsilon
    result = threaded_research(r_values=[28, 28 + epsilon],
                               ts_size=sizes,
                               gap_number=20,
                               test_size_constant=50)
    return epsilon, int(sizes[1] / 10), result[0], result[1]


def main():
    divisor = 10

    # Параллелизация цикла по размеру 2 ряда
    second_size_range = np.logspace(5, 6, 30, base=10)
    epsilon_range = [1, 5, 8, 10, 15, 30]
    rmses_sizes = []
    np_points_sizes = []
    epsilon_and_sizes_list = []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_size, epsilon, [100000, int(second_size / 10) * 10], divisor)
                   for second_size in second_size_range for epsilon in epsilon_range]
        for future in concurrent.futures.as_completed(futures):
            epsilon, second_size, rmse, np_point = future.result()
            epsilon_and_sizes_list.append([epsilon, second_size])
            rmses_sizes.append(rmse)
            np_points_sizes.append(np_point)
            with open(f"./outputs/fast_output{epsilon}.txt", "a") as f:
                f.write(str(second_size) + "," + str(np_point) + "," + str(rmse) + "\n")

    # Сортировка по доле второго ряда
    sorted_sizes = sorted(zip(epsilon_and_sizes_list, rmses_sizes, np_points_sizes), key=lambda x: x[0])
    epssizes, rmses_sizes, np_points_sizes = map(list, zip(*sorted_sizes))

    # Разделяем epsilon и sizes из epssizes
    epsilons, sizes = zip(*epssizes)

    # Создаем сетку для построения поверхности
    grid_epsilon, grid_size = np.meshgrid(np.linspace(min(epsilons), max(epsilons), 100),
                                          np.linspace(min(sizes), max(sizes), 100))

    # Интерполяция данных для np_points
    grid_np_points = griddata((epsilons, sizes), np_points_sizes, (grid_epsilon, grid_size), method='cubic')

    # Интерполяция данных для rmse
    grid_rmse = griddata((epsilons, sizes), rmses_sizes, (grid_epsilon, grid_size), method='cubic')

    # Построение поверхности для np_points
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(grid_epsilon, grid_size, grid_np_points, cmap='viridis')
    ax.set_xlabel('Epsilon')
    ax.set_ylabel('Size')
    ax.set_zlabel('Количество NP точек')
    ax.set_title('Зависимость количества непредсказываемых точек от epsilon и size')
    plt.savefig(f'./graphics/fast_epsilon_size_np_points.png', dpi=300)
    plt.show()

    # Построение поверхности для rmse
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(grid_epsilon, grid_size, grid_rmse, cmap='plasma')
    ax.set_xlabel('Epsilon')
    ax.set_ylabel('Size')
    ax.set_zlabel('RMSE')
    ax.set_title('Зависимость ошибки от epsilon и size')
    plt.savefig(f'./graphics/fast_epsilon_size_rmse.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    main()
