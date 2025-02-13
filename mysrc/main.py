# coding: utf-8
import numpy as np
from Research import *
from matplotlib import pyplot as plt
import concurrent.futures


def process_epsilon(epsilon, shares, size, divisor):
    # Каждому вызову передаём epsilon и используемые параметры
    result = threaded_research(r_values=[28, 28 + epsilon],
                               ts_size=(np.array(shares) * size * divisor).astype(np.uint64),
                               gap_number=20,
                               test_size_constant=50)
    return epsilon, result[0], result[1]


def main():
    divisor = 10
    shares = [0.5, 0.5]
    size = 20000

    epsilons_range = np.arange(1, 30.0, 0.5)
    rmses = []
    np_points = []
    epsilons = []

    # Параллельный запуск цикла по epsilon
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Подготавливаем список заданий для каждого epsilon
        futures = [executor.submit(process_epsilon, epsilon, shares, size, divisor) for epsilon in epsilons_range]
        for future in concurrent.futures.as_completed(futures):
            eps, rmse, np_point = future.result()
            epsilons.append(eps)
            rmses.append(rmse)
            np_points.append(np_point)

    # Сортируем по epsilons, если порядок важен
    sorted_results = sorted(zip(epsilons, rmses, np_points), key=lambda x: x[0])
    epsilons, rmses, np_points = map(list, zip(*sorted_results))

    # Построение графиков
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

    # То же самое можно сделать и для второго цикла (по shares)
    second_shares_range = np.arange(0.1, 0.9, 0.02)
    rmses_shares = []
    np_points_shares = []
    shares_list = []
    epsilon_fixed = 1  # фиксированное значение epsilon для этого примера

    def process_share(second_share, size, divisor, epsilon):
        current_shares = [1 - second_share, second_share]
        result = threaded_research(r_values=[28, 28 + epsilon],
                                   ts_size=(np.array(current_shares) * size * divisor).astype(np.uint64),
                                   gap_number=20,
                                   test_size_constant=50)
        return second_share, result[0], result[1]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_share, second_share, size, divisor, epsilon_fixed) for second_share in
                   second_shares_range]
        for future in concurrent.futures.as_completed(futures):
            second_share, rmse, np_point = future.result()
            shares_list.append(second_share)
            rmses_shares.append(rmse)
            np_points_shares.append(np_point)

    # Чтобы сохранить порядок по shares:
    sorted_shares = sorted(zip(shares_list, rmses_shares, np_points_shares), key=lambda x: x[0])
    shares_list, rmses_shares, np_points_shares = map(list, zip(*sorted_shares))

    plt.figure()
    plt.plot(shares_list, rmses_shares)
    plt.xlabel('Доля второго ряда')
    plt.ylabel('Ошибка')
    plt.title('Зависимость ошибки от доли второго ряда')
    plt.savefig('./graphics/shares_and_rmses.png', dpi=300)
    plt.show()

    plt.figure()
    plt.plot(shares_list, np_points_shares)
    plt.xlabel('Доля второго ряда')
    plt.ylabel('Количество NP точек')
    plt.title('Зависимость количества непредсказываемых точек от доли второго ряда')
    plt.savefig('./graphics/shares_and_np_points.png', dpi=300)
    plt.show()

    # Пример параллелизации вложенного цикла: по epsilon и size
    # Для хранения результатов можно использовать словари или другой удобный формат
    rmses_nested = {}
    np_points_nested = {}
    shares = [0.5, 0.5]

    def process_nested(epsilon, size, shares, divisor):
        result = threaded_research(r_values=[28, 28 + epsilon],
                                   ts_size=(np.array(shares) * size * divisor).astype(np.uint64),
                                   gap_number=20,
                                   test_size_constant=50)
        return epsilon, size, result[0], result[1]

    epsilons_nested = np.arange(1, 30.0, 2)
    sizes_range = range(20000, 35000, 1000)

    tasks = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for epsilon in epsilons_nested:
            for size_val in sizes_range:
                tasks.append(executor.submit(process_nested, epsilon, size_val, shares, divisor))
        for future in concurrent.futures.as_completed(tasks):
            epsilon, size_val, rmse, np_point = future.result()
            rmses_nested[(epsilon, size_val)] = rmse
            np_points_nested[(epsilon, size_val)] = np_point

    # Создадим сетку и матрицу значений для 3D графика (например, для MSE)
    X, Y = np.meshgrid(list(sizes_range), epsilons_nested)
    Z = np.array([[rmses_nested[(eps, sz)] for sz in sizes_range] for eps in epsilons_nested])

    # Построение 3D графика
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', antialiased=True)
    ax.set_xlabel('Size', fontsize=12)
    ax.set_ylabel('Epsilon', fontsize=12)
    ax.set_zlabel('MSE', fontsize=12)
    ax.set_title('3D Surface Plot of MSE', pad=20)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.view_init(elev=25, azim=-45)
    plt.savefig('./graphics/surface_plot.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    main()


if __name__ == '__main__':
    main()
