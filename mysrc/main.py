# coding: utf-8
from scipy.spatial import KDTree

from evaluation import *
from candidate_scorer import *
from tqdm import tqdm
R_TARGET = 28.0
SIZE_TARGET = 10000
SIZE_CANDIDATE = 20000
R_DISTANCE = 5.0
N_CANDIDATES = 10000
N_SELECT = 480
OUTPUT_DIR = "/home/ikvasilev/PaTHoP/assets/results/mannayuitni"
BASELINES_FILE = os.path.join(OUTPUT_DIR, "baselines_validation2.txt")


DATA_FILE = "candidates_scores.npz"



def reconstruct_attractor(x: np.ndarray, dim: int, delay: int) -> np.ndarray:
    n = len(x)
    max_idx = n - (dim - 1) * delay
    if max_idx <= 0:
        raise ValueError("Ряд слишком короткий для заданных параметров dim и delay.")
    attractor = np.column_stack([x[i:i + max_idx] for i in range(0, dim * delay, delay)])
    return attractor




def chamfer_distance_metric(x1: np.ndarray, x2: np.ndarray, dim: int = 3, delay: int = 10) -> float:
    y1 = reconstruct_attractor(x1, dim, delay)
    y2 = reconstruct_attractor(x2, dim, delay)

    tree1 = KDTree(y1)
    tree2 = KDTree(y2)

    dist_y1_to_y2, _ = tree2.query(y1, k=1, workers=-1)
    dist_y2_to_y1, _ = tree1.query(y2, k=1, workers=-1)

    chamfer_dist = np.mean(dist_y1_to_y2 ** 2) + np.mean(dist_y2_to_y1 ** 2)

    return chamfer_dist


def chamfer_distance_score(r1, size1, r2, size2):

    ts1 = TimeSeries(series_type="Lorentz", size=size1, r=r1)
    ts2 = TimeSeries(series_type="Lorentz", size=size2, r=r2)
    x1, x2 = ts1.values, ts2.values
    return chamfer_distance_metric(x1, x2)
def main():
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
        scores = []
        for r in tqdm(candidate_r_values):
            score = candidate_score_ts(main_series, r, SIZE_CANDIDATE)
            scores.append(score)
        np.savez(DATA_FILE, r_values=candidate_r_values, scores=np.array(scores))
    print(f"Сохранено {len(candidate_r_values)} кандидатов в {DATA_FILE}")
    sorted_indices = np.argsort(scores)
    sorted_indices = list(sorted_indices)
    best_indices = sorted_indices[:N_SELECT]
    print(np.array(scores)[best_indices])
    random_indices = np.random.choice(sorted_indices, 48, replace=False)

    best_of_the_best = candidate_r_values[best_indices]
    scores = []
    for r in tqdm(candidate_r_values):
        score = chamfer_distance_score(28,SIZE_TARGET, r, SIZE_CANDIDATE)
        scores.append(score)
    np.savez(DATA_FILE, r_values=candidate_r_values, scores=np.array(scores))
    print(best_of_the_best,"best")

    random_r = candidate_r_values[random_indices]
    for r in best_of_the_best:
        os.system(f"sbatch -A proj_1716 ./subbash {R_TARGET} {r} {SIZE_TARGET} {SIZE_CANDIDATE} best")

    for r in random_r:
        os.system(f"sbatch -A proj_1716 ./subbash {R_TARGET} {r} {SIZE_TARGET} {SIZE_CANDIDATE} random")

    # baseline_small = evaluation([R_TARGET], np.array([SIZE_TARGET]))
    # baseline_large = evaluation([R_TARGET], np.array([SIZE_TARGET + SIZE_CANDIDATE]))
    #
    # with open(BASELINES_FILE, 'w') as f:
    #     f.write(f"{baseline_small[0]},{baseline_small[1]},{baseline_small[2]}\n")
    #     f.write(f"{baseline_large[0]},{baseline_large[1]},{baseline_large[2]}")


if __name__ == "__main__":
    main()