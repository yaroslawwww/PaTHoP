# coding: utf-8
from evaluation import *
from candidate_scorer import *
from tqdm import tqdm
R_TARGET = 28.0
SIZE_TARGET = 10000
SIZE_CANDIDATE = 20000
R_DISTANCE = 5.0
N_CANDIDATES = 1000
N_SELECT = 47
OUTPUT_DIR = "/home/ikvasilev/PaTHoP/assets/results/mannayuitni"
BASELINES_FILE = os.path.join(OUTPUT_DIR, "baselines.txt")


DATA_FILE = "candidates_scores.npz"
if os.path.exists(DATA_FILE):
    print("Загружаем сохранённые кандидатов и скоры...")
    data = np.load(DATA_FILE)
    candidate_r_values = data['r_values']
    scores = data['scores'].tolist()
else:
    print("Файл не найден. Генерируем новых кандидатов...")
    candidate_r_values = np.random.uniform(
        low=R_TARGET - R_DISTANCE,
        high=R_TARGET + R_DISTANCE,
        size=N_CANDIDATES
    )
    main_series = TimeSeries(r=R_TARGET, size=SIZE_TARGET)
    true_cov = candidate_score(R_TARGET, SIZE_TARGET, R_TARGET, SIZE_TARGET)
    scores = []

    for r in tqdm(candidate_r_values):
        score = abs(candidate_score_ts(main_series, r, SIZE_CANDIDATE) - true_cov)
        scores.append(score)
    np.savez(DATA_FILE, r_values=candidate_r_values, scores=np.array(scores))
    print(f"Сохранено {len(candidate_r_values)} кандидатов в {DATA_FILE}")

def main():


    sorted_indices = np.argsort(scores)
    sorted_indices = list(sorted_indices)
    best_indices = sorted_indices[:N_SELECT]
    remaining_indices = sorted_indices[N_SELECT:]
    random_indices = np.random.choice(remaining_indices, N_SELECT, replace=False)

    best_r = candidate_r_values[best_indices]
    print(best_r,"best")
    random_r = candidate_r_values[random_indices]

    for r in best_r:
        os.system(f"sbatch -A proj_1716 ./subbash {R_TARGET} {r} {SIZE_TARGET} {SIZE_CANDIDATE} best")

    for r in random_r:
        os.system(f"sbatch -A proj_1716 ./subbash {R_TARGET} {r} {SIZE_TARGET} {SIZE_CANDIDATE} random")

    baseline_small = evaluation([R_TARGET], np.array([SIZE_TARGET]))
    baseline_large = evaluation([R_TARGET], np.array([SIZE_TARGET + SIZE_CANDIDATE]))

    with open(BASELINES_FILE, 'w') as f:
        f.write(f"{baseline_small[0]},{baseline_small[1]},{baseline_small[2]}\n")
        f.write(f"{baseline_large[0]},{baseline_large[1]},{baseline_large[2]}")


if __name__ == "__main__":
    main()