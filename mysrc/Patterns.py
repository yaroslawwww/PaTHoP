from TimeSeries import TimeSeries
import numpy as np


class Templates:
    def __init__(self, template_length: int , max_template_spread: int ,ozu_limit = 4000):

        templates_quantity = max_template_spread ** (template_length - 1)
        # сами шаблоны
        templates = (np.repeat(0, templates_quantity).reshape(-1, 1),)

        # Уже понятный код, который заполняет шаблоны нужными значениями.
        for i in range(1, template_length):
            templates_quantity_in_i_step = max_template_spread ** (template_length - (i + 1))
            templates_quantity_in_next_step = templates_quantity_in_i_step * max_template_spread
            # генерация всех смещений от 1 до max_template_spread + 1
            shifts = np.repeat(np.arange(1, max_template_spread + 1, dtype=int), templates_quantity_in_i_step)
            # генерация строки из следующих значений
            row = (shifts + templates[i - 1][::templates_quantity_in_next_step])
            col = row.reshape(-1, 1)
            templates += (col,)  # добавляем новый столбец

        all_templates: np.ndarray = np.hstack(templates)
        np.random.shuffle(all_templates)
        self.templates = all_templates[:ozu_limit]
        self.train_set = None
        # # формы шаблонов, т.е. [1, 1, 1], [1, 1, 2] и т.д.
        shapes: np.ndarray = self.templates[:, 1:] - self.templates[:, :-1]
        self.observation_indexes = np.cumsum(-shapes[:, ::-1], axis=1)[:, ::-1]


    def create_train_set(self, time_series_list: list[TimeSeries]):
        # необходим ряд не менее чем template_length * max_template_spread
        train_part = []
        for time_series in time_series_list:
            if time_series.train is not None:
                train_part.extend(time_series.train)
            else:
                train_part.extend(time_series.values)
        train_part = np.array(train_part)
        x_dim = self.templates.shape[0]
        y_dim = 0
        for i in range(x_dim):
            template_window:int = self.templates[i][-1]
            y_dim = max(train_part.shape[0] - template_window,y_dim)
        z_dim = self.templates.shape[1]
        self.train_set: np.ndarray = np.full(shape=(x_dim, y_dim, z_dim), fill_value=np.inf, dtype=float)
        for i in range(len(self.templates)):
            current_template = self.templates[i]
            template_window:int = current_template[-1]
            time_series_indexes = current_template + np.arange(len(train_part) - template_window)[:, None]
            time_series_vectors = train_part[time_series_indexes]
            self.train_set[i, :len(time_series_vectors)] = time_series_vectors
