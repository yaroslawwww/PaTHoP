import os
import numpy as np
import gc
from evaluation import research

# Настройки
HORIZONS = [1]  # Можно добавить 1, если нужно
HOW_MANY_GAPS = 1000
SIZES = np.linspace(10000, 40000, 24).astype(int)
OUTPUT_FILE = "exp1_fast_recalc.csv"


def run():
    if not os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "w") as f:
            f.write("Size,Horizon,rmse_01,np_01,mape_01\n")

    for h in HORIZONS:
        for N in SIZES:
            print(f"\n>>> Расчет N={N}, h={h}")
            test_size = N + HOW_MANY_GAPS + 100

            # Вызываем расчет
            # Внутри research вызовется tsproc.fit, который теперь на 8 ядрах
            res = research(
                r_values=[28.0, 28.0],
                ts_size=np.array([test_size, N]),
                how_many_gaps=HOW_MANY_GAPS,
                test_size_constant=h
            )

            with open(OUTPUT_FILE, "a") as f:
                f.write(f"{N},{h},{res[3]:.6f},{res[4]:.6f},{res[5]:.6f}\n")

            gc.collect()  # Очистка после каждой точки


if __name__ == "__main__":
    run()
