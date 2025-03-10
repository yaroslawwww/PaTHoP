# coding: utf-8
from Lorentz import Lorentz
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

class TimeSeries:
    def __init__(self, series_type = "Lorentz", size = 0,r = 28,dt = 0.01,divisor = 10,array = None):
        if series_type == "Lorentz":
            x, y, z = Lorentz().generate(dt = dt, steps= size,r = r)
            if (x.max() == x.min()):
                print("SOS",size,r,dt,divisor)
            x = (x - x.min()) / (x.max() - x.min())  # нормализация чисел
            self.values = list(x)[::divisor]
        else:
            x = np.array(array)
            x = (x - x.min()) / (x.max() - x.min())
            self.values = list(x)
        self.train = None
        self.after_test_train = None
        self.test = None
        self.val = []
        self.time = [i for i in range(len(self.values))]
    def split_train_val_test(self,window_index, test_size = 50):
        if window_index + test_size> len(self.values):
            raise ValueError("test index out of range")

        self.after_test_train = self.values[window_index + test_size + 1:]
        self.train = self.values[:window_index]
        self.test = self.values[window_index:window_index+test_size]