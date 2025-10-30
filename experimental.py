import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from statsmodels.tsa.stattools import pacf


def lorenz_pacf(r, sigma=10.0, beta=8.0 / 3.0, t_transient=50.0, n_points=100000, dt=0.01, nlags=50):
    t_end = t_transient + n_points * dt
    t_eval = np.linspace(0, t_end, int(t_end / dt) + 1)

    def ode(t, s):
        x, y, z = s
        return [
            sigma * (y - x),
            x * (r - z) - y,
            x * y - beta * z
        ]

    sol = solve_ivp(ode, (0, t_end), [1.0, 1.0, 1.0], t_eval=t_eval, rtol=1e-9, atol=1e-12)
    x = sol.y[0]
    x_stationary = x[int(t_transient / dt):]

    if not np.isfinite(x_stationary).all():
        raise ValueError(f"r={r}: ряд содержит NaN или Inf")
    if np.std(x_stationary) < 1e-8:
        raise ValueError(f"r={r}: ряд почти константа")

    # Используем устойчивый метод Levinson-Durbin
    p = pacf(x_stationary, nlags=nlags, method='ldb')

    # Теперь эта проверка почти никогда не сработает
    if np.any(np.abs(p) > 1.01):
        print(f"⚠️  Неожиданно: r={r} — PACF вне [-1,1] даже с 'ldb'")
        print(f"    min={p.min():.3f}, max={p.max():.3f}")

    return p

if __name__ == '__main__':
    r_list = [27.99, 27.9, 28.0, 28.0001, 28.01, 28.1]
    nlags = 30

    plt.figure(figsize=(10, 6))
    for i, r in enumerate(r_list):
        p = lorenz_pacf(r, n_points=100000, nlags=nlags)
        offset = i * 0.025
        plt.plot(np.arange(len(p)), p + offset, label=f'r = {r}', linewidth=1.6)

    plt.axhline(0, color='black', linewidth=0.5)
    plt.title('PACF (100k точек после transient)')
    plt.xlabel('Lag')
    plt.ylabel('PACF (+ vertical offset)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()
