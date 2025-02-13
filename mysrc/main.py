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
    for epsilon in np.arange(1, 30.0, 0.5):
        baranov_result = threaded_research(r_values=[28, 28 + epsilon],
                                           ts_size=(np.array(shares) * size * divisor).astype(np.uint64), gap_number=20,
                                           test_size_constant=50)
        rmses.append(baranov_result[0])
        np_points.append(baranov_result[1])
        epsilons.append(epsilon)
    plt.plot(epsilons, rmses)
    plt.show()
    plt.savefig('./graphics/epsilons_and_rmses.png', dpi=300)
    plt.plot(epsilons, np_points)
    plt.show()
    plt.savefig('./graphics/epsilons_and_np_points.png')
    rmses = []
    shares_list = []
    np_points = []
    epsilons = []
    for second_share in np.arange(0.1, 0.9, 0.02):
        epsilon = 1
        shares = [1 - second_share, second_share]
        baranov_result = threaded_research(r_values=[28, 28 + epsilon],
                                           ts_size=(np.array(shares) * size * divisor).astype(np.uint64), gap_number=20,
                                           test_size_constant=50)
        rmses.append(baranov_result[0])
        np_points.append(baranov_result[1])
        shares_list.append(second_share)
    plt.plot(epsilons, rmses)
    plt.xlabel('Возмущение epsilon')  # Подпись для оси х
    plt.ylabel('Ошибка')  # Подпись для оси y
    plt.title('Зависимость ошибки от eps')  # Название
    plt.show()
    plt.savefig('./graphics/shares_and_rmses.png', dpi=300)

    plt.plot(epsilons, np_points)
    plt.xlabel('Возмущение epsilon')  # Подпись для оси х
    plt.ylabel('Количество NP точек')  # Подпись для оси y
    plt.title('Зависимость количества непредсказываемых точек от eps')  # Название
    plt.show()
    plt.savefig('./graphics/shares_and_np_points.png', dpi=300)

    rmses = {}
    shares = [0.5, 0.5]
    np_points = {}
    epsilons = []
    for epsilon in np.arange(1, 30.0, 2):
        for size in range(20000, 35000, 1000):
            baranov_result = threaded_research(r_values=[28, 28 + epsilon],
                                               ts_size=(np.array(shares) * size * divisor).astype(np.uint64),
                                               gap_number=20,
                                               test_size_constant=50)
            rmses[[epsilon, size]] = baranov_result[0]
            np_points[[epsilon, size]] = baranov_result[1]
    # Создание сетки для 3D графика
    X, Y = np.meshgrid(range(20000, 35000, 1000), np.arange(1, 30.0, 2))

    # Построение 3D графика
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(X, Y, rmses,
                           cmap='viridis',
                           edgecolor='none',
                           antialiased=True)

    # Настройка оформления
    ax.set_xlabel('Size', fontsize=12)
    ax.set_ylabel('Epsilon', fontsize=12)
    ax.set_zlabel('Mse', fontsize=12)
    ax.set_title('3D Surface Plot of MSE', pad=20)

    fig.colorbar(surf, shrink=0.5, aspect=5)

    # Оптимальный угол обзора
    ax.view_init(elev=25, azim=-45)

    plt.show()
    plt.savefig('./graphics/surface_plot.png', dpi=300)


if __name__ == '__main__':
    main()
