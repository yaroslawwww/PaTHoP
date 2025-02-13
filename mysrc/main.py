# coding: utf-8
import concurrent.futures

from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

from Research import *


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
    second_size_range = [10**4*10,10**5*10,10**6*10,10**7*10]
    epsilon_range = [1,8,10,15,30]
    rmses_sizes = []
    np_points_sizes = []
    sizes_list = []
    epsilon_and_sizes_list = []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_size, epsilon,[100000,second_size], divisor)
                   for second_size in second_size_range for epsilon in epsilon_range]
        for future in concurrent.futures.as_completed(futures):
            epsilon,second_sizes, rmse, np_point = future.result()
            epsilon_and_sizes_list.append([epsilon,second_sizes])
            rmses_sizes.append(rmse)
            np_points_sizes.append(np_point)

    # Сортировка по доле второго ряда
    sorted_sizes = sorted(zip(sizes_list, rmses_sizes, np_points_sizes), key=lambda x: x[0])
    sizes_list, rmses_sizes, np_points_sizes = map(list, zip(*sorted_sizes))

    # Преобразование данных в массив
    epsilon = np.array([item[0] for item in epsilon_and_sizes_list])
    second_sizes = np.array([item[1] for item in epsilon_and_sizes_list])
    rmses = np.array(rmses_sizes)
    np_points = np.array(np_points_sizes)

    # Логарифмическое преобразование для second_sizes
    log_second_sizes = np.log10(second_sizes)

    # Создание сетки для аппроксимации
    xi = np.linspace(epsilon.min(), epsilon.max(), 100)
    yi = np.linspace(log_second_sizes.min(), log_second_sizes.max(), 100)
    xi, yi = np.meshgrid(xi, yi)
    # Функция для аппроксимации поверхности
    def func(X, a, b, c, d, e, f):
        x, y = X
        return a + b * x + c * y + d * x ** 2 + e * y ** 2 + f * x * y

    # Аппроксимация для RMSE
    popt_rmse, _ = curve_fit(func, (epsilon, log_second_sizes), rmses)
    rmse_grid = func((xi, yi), *popt_rmse)

    # Аппроксимация для np_points
    popt_np, _ = curve_fit(func, (epsilon, log_second_sizes), np_points)
    np_grid = func((xi, yi), *popt_np)

    # Создание графиков
    fig = plt.figure(figsize=(18, 8))

    # График для RMSE
    ax1 = fig.add_subplot(121, projection='3d')
    surf1 = ax1.plot_surface(xi, 10 ** yi, rmse_grid, cmap='viridis', alpha=0.8)
    ax1.scatter(epsilon, second_sizes, rmses, c='r', s=50)
    ax1.set_xlabel('Epsilon', fontsize=12)
    ax1.set_ylabel('Second Sizes (log scale)', fontsize=12)
    ax1.set_zlabel('RMSE', fontsize=12)
    ax1.set_yscale('log')
    ax1.set_title('RMSE vs Epsilon and Second Sizes', fontsize=14)
    fig.colorbar(surf1, ax=ax1, shrink=0.5)

    # График для np_points
    ax2 = fig.add_subplot(122, projection='3d')
    surf2 = ax2.plot_surface(xi, 10 ** yi, np_grid, cmap='plasma', alpha=0.8)
    ax2.scatter(epsilon, second_sizes, np_points, c='g', s=50)
    ax2.set_xlabel('Epsilon', fontsize=12)
    ax2.set_ylabel('Second Sizes (log scale)', fontsize=12)
    ax2.set_zlabel('np_points', fontsize=12)
    ax2.set_yscale('log')
    ax2.set_title('np_points vs Epsilon and Second Sizes', fontsize=14)
    fig.colorbar(surf2, ax=ax2, shrink=0.5)
    ax2.view_init(elev=25, azim=-45)
    plt.tight_layout()
    plt.savefig('fig.png', dpi=300)
    plt.show()
if __name__ == '__main__':
    main()
