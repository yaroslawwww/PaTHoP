# coding: utf-8
from TimeSeries import TimeSeries
import numpy as np


class Templates:
    def __init__(self, template_length, max_template_spread):
        self.train_set = None
        self.template_length = template_length
        self.max_template_spread = max_template_spread

        # Основные оптимизации:
        # 1. Векторизованное создание шаблонов
        # 2. Предварительное выделение памяти
        # 3. Убраны промежуточные кортежи

        templates_quantity = max_template_spread ** (template_length - 1)
        templates = np.zeros((templates_quantity, template_length), dtype=int)

        for i in range(1, template_length):
            step_size = max_template_spread ** (template_length - i - 1)
            repeat_count = max_template_spread ** i

            # Генерация шаблонов с использованием векторных операций
            block = np.repeat(np.arange(1, max_template_spread + 1), step_size)
            templates[:, i] = np.tile(block, repeat_count // max_template_spread) + templates[:, i - 1]

        self.templates = templates

        # Оптимизированный расчет observation_indexes
        shapes = np.diff(templates, axis=1)
        self.observation_indexes = shapes[:, ::-1].cumsum(axis=1)[:, ::-1] * -1
        
    # def create_train_set(self, time_series_list: list[TimeSeries]):
    #     # необходим ряд не менее чем template_length * max_template_spread
    #     train_part = []
    #     for time_series in time_series_list:
    #         if time_series.train is not None:
    #             train_part.extend(time_series.train)
    #         else:
    #             train_part.extend(time_series.values)
    #     train_part = np.array(train_part)
    #     x_dim = self.templates.shape[0]
    #     y_dim = 0
    #     for i in range(x_dim):
    #         template_window: int = self.templates[i][-1]
    #         y_dim = max(train_part.shape[0] - template_window, y_dim)
    #     z_dim = self.templates.shape[1]
    #     self.train_set: np.ndarray = np.full(shape=(x_dim, y_dim, z_dim), fill_value=np.inf, dtype=float)
    #     for i in range(len(self.templates)):
    #         current_template = self.templates[i]
    #         template_window: int = current_template[-1]
    #         time_series_indexes = current_template + np.arange(len(train_part) - template_window)[:, None]
    #         time_series_vectors = train_part[time_series_indexes]
    #         self.train_set[i, :len(time_series_vectors)] = time_series_vectors

    def create_train_set(self, time_series_list):
        all_train_sets = []  # Список для хранения train_set каждого временного ряда

        for time_series in time_series_list:
            # Выбираем данные из временного ряда
            if time_series.train is not None:
                data = np.array(time_series.train)
            else:
                data = np.array(time_series.values)

            if len(data) == 0:
                continue  # Пропускаем пустые ряды

            # Вычисляем размерности для текущего временного ряда
            x_dim = self.templates.shape[0]
            y_dim = 0
            for i in range(x_dim):
                template_window = self.templates[i][-1]
                y_dim = max(len(data) - template_window, y_dim)

            z_dim = self.templates.shape[1]

            if y_dim <= 0:
                continue  # Ряд слишком короткий для любых шаблонов

            # Инициализируем train_set для текущего ряда
            individual_train_set = np.full((x_dim, y_dim, z_dim), np.inf, dtype=float)

            # Заполняем данными
            for i in range(len(self.templates)):
                current_template = self.templates[i]
                template_window = current_template[-1]
                n_windows = len(data) - template_window

                if n_windows > 0:
                    time_series_indexes = current_template + np.arange(n_windows)[:, None]
                    time_series_vectors = data[time_series_indexes]
                    individual_train_set[i, :n_windows] = time_series_vectors

            all_train_sets.append(individual_train_set)

        # Объединяем все train_set'ы вдоль оси временных окон (axis=1)
        if all_train_sets:
            self.train_set = np.concatenate(all_train_sets, axis=1)
        else:
            # На случай, если все ряды были слишком короткими
            self.train_set = np.empty((self.templates.shape[0], 0, self.templates.shape[1]))