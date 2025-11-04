# coding: utf-8
import sys
from evaluation import *
import os

def main():
    print(sys.argv[1:])
    r_values = [float(sys.argv[1]),float(sys.argv[2])]
    ts_sizes = [int(sys.argv[3]),int(sys.argv[4])]
    file_name = sys.argv[5]
    rmse, np_points, mape = evaluation(r_values,ts_sizes)
    OUTPUT_DIR = "/home/ikvasilev/PaTHoP/assets/results/mannayuitni"
    BASELINES_FILE = os.path.join(OUTPUT_DIR, f"{file_name}.txt")
    with open(BASELINES_FILE, 'a') as f:
        f.write(f"{rmse},{np_points},{mape},{r_values[1]}\n")
if __name__ == "__main__":
    main()