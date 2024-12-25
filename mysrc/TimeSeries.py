from Lorentz import Lorentz
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

class TimeSeries:
    def __init__(self, series_type: str, size: int = 0,array = None):
        if series_type == "Lorentz":
            x, y, z = Lorentz().generate(0.1, size)
            x = (x - x.min()) / (x.max() - x.min())  # нормализация чисел
            self.values = list(x)
        else:
            x = np.array(array)
            x = (x - x.min()) / (x.max() - x.min())
            self.values = list(x)
        self.train = None
        self.test = None
        self.val = None
        self.time = [i for i in range(len(self.values))]
    def split_train_val_test(self, train_size: int = 5000, val_size: int = 0, test_size: int = 200):
        self.train = self.values[:train_size]
        self.val = self.values[train_size:train_size + val_size]
        self.test =  self.values[train_size + val_size:train_size + val_size + test_size]
        # #возвращает начало соответсвующих выборок
        # return train_size, train_size + val_size, train_size + val_size + test_size
    def print(self, size: int = 500):
        plt.plot(self.time[:size], self.values[:size])
        plt.show()



