# coding: utf-8
import sys
import os
import numpy as np
from evaluation import evaluation
from multiprocessing import Pool

OUTPUT_DIR = "/home/ikvasilev/PaTHoP/assets/results/manna2"


def run_single_candidate(r_cand, main_r, main_size, cand_size, pred_len, output_file1,output_file2):
    """Запуск evaluation для одного кандидата и запись результата"""
    try:
        rmse1, np1, mape1, rmse2, np2, mape2= evaluation(
            r_values=[main_r, r_cand],
            ts_sizes=[main_size, cand_size],
            prediction_size=pred_len
        )
        with open(output_file1, 'a') as f:
            f.write(f"{rmse1},{np1},{mape1},{r_cand:.12f}\n")
        with open(output_file2, 'a') as f:
            f.write(f"{rmse2},{np2},{mape2},{r_cand:.12f}\n")
    except Exception as e:
        print(f"Ошибка при r={r_cand}: {e}", file=sys.stderr)


def main():
    # Ожидаем ровно 6 аргументов:
    # 1. R_TARGET (main_r)
    # 2. SIZE_TARGET (main_size)
    # 3. SIZE_CANDIDATE (cand_size)
    # 4. group_name (например: best_top48, pf_chamfer, random)
    # 5. prediction_len
    # 6. path_to_r_file — файл с 48 значениями r (по одному на строку)
    if len(sys.argv) != 7:
        print("Использование: python this_script.py "
              "<R_TARGET> <SIZE_TARGET> <SIZE_CANDIDATE> "
              "<group_name> <prediction_len> <r_file_path>")
        sys.exit(1)

    main_r = float(sys.argv[1])
    main_size = int(sys.argv[2])
    cand_size = int(sys.argv[3])
    group_name = sys.argv[4]
    pred_len = int(sys.argv[5])
    r_file_path = sys.argv[6]

    # Формируем имя выходного файла
    output_file1 = os.path.join(OUTPUT_DIR, f"{group_name}_10000_{pred_len}_1.txt")
    output_file2 = os.path.join(OUTPUT_DIR, f"{group_name}_10000_{pred_len}_2.txt")

    # Читаем все кандидатные r из файла
    try:
        candidate_r_values = np.loadtxt(r_file_path)
        if candidate_r_values.ndim == 0:
            candidate_r_values = candidate_r_values.reshape(1)
        print(f"Загружено {len(candidate_r_values)} кандидатов из {r_file_path}")
    except Exception as e:
        print(f"Не удалось прочитать файл с r-значениями {r_file_path}: {e}")
        sys.exit(1)

    # Параллельный запуск на всех доступных ядрах (обычно 48 при --cpus-per-task=48)
    with Pool(12) as pool:
        args = [
            (r_cand, main_r, main_size, cand_size, pred_len, output_file1,output_file2)
            for r_cand in candidate_r_values
        ]
        list(pool.starmap(run_single_candidate, args))

    print(f"Готово: {group_name}, предсказание на {pred_len} шагов, результаты в {output_file1},{output_file2}")


if __name__ == "__main__":
    main()