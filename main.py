import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2


# функція розподілу F(t)=1-e^(-rate*t), t>=0
def F_t(t: float, rate: float) -> float:
    if t <= 0:
        return 0.0
    return 1.0 - np.exp(-rate * t)


def х_square_verification(x: np.ndarray, rate_used: float, m: int = 12, alpha: float = 0.05, q: int = 0):
    if rate_used <= 0:
        raise ValueError("rate_used має бути > 0")
    if m < 2:
        raise ValueError("Кількість інтервалів m має бути >= 2")
    if q < 0:
        raise ValueError("q має бути >= 0")

    x = np.asarray(x)
    n = x.size

    # Межі інтервалів за квантилями
    p = np.arange(0, m + 1) / m
    edges = -np.log(1 - p[:-1]) / rate_used  # F^(-1)(t)= -(ln(1-p)/rate
    edges = np.append(edges, np.inf)

    # Спостережені частоти n_i
    obs, _ = np.histogram(x, bins=edges)

    # Теоретичні частоти np_i^T та xi^2
    exp_counts = np.zeros(m, dtype=float)
    contributions = np.zeros(m, dtype=float)

    for i in range(m):
        a, b = edges[i], edges[i + 1]

        Fa = F_t(a, rate_used)
        Fb = 1.0 if np.isinf(b) else F_t(b, rate_used)

        prob = Fb - Fa
        exp_counts[i] = n * prob

        # xi^2 = (n_i - np_i^T)^2 / (np_i^T)
        contributions[i] = (obs[i] - exp_counts[i]) ** 2 / exp_counts[i]

    chi2_stat = float(np.sum(contributions))

    # df = k - 1 - q
    df = m - 1 - q
    if df <= 0:
        raise ValueError(f"Некоректні df={df}, (треба df>0).")

    # Критичне значення χ²
    chi2_crit = float(chi2.ppf(1 - alpha, df))

    # Таблиця
    bounds_str = []
    for i in range(m):
        left = f"{edges[i]:.4f}"
        right = "∞" if np.isinf(edges[i + 1]) else f"{edges[i + 1]:.4f}"
        bounds_str.append(f"[{left}, {right})")

    table = pd.DataFrame({
        "№ інтервалу": np.arange(1, m + 1),
        "Межі [a_i, b_i)": bounds_str,
        "n_i (спост.)": obs,
        "np_i^T (теор.)": np.round(exp_counts, 4),
        "розрахунок χ²": np.round(contributions, 4),
    })

    min_expected = float(np.min(exp_counts))
    return chi2_stat, chi2_crit, df, table, min_expected


def main():
    # 1) Генерація 3000 чисел
    n = 3000
    np.random.seed(42)

    while True:
        try:
            lam = float(input("Введіть параметр λ (>0): ").replace(",", "."))
            if lam <= 0:
                print("λ має бути > 0. Спробуйте ще раз.")
                continue
            break
        except ValueError:
            print("Некоректний ввід. Спробуйте ще раз.")

    xi = np.random.uniform(0.0, 1.0, size=n)  # ξ ~ U(0,1)

    x = -(1.0 / lam) * np.log(xi ** 2)  # x = -(1/λ) ln(ξ^2)

    # 2) Гістограма частот
    plt.figure()
    plt.hist(x, bins=30, edgecolor="black")
    plt.title(f"Гістограма частот (n={n}, λ={lam})")
    plt.xlabel("x")
    plt.ylabel("частота")
    plt.show()

    # 3) Середнє та дисперсія
    x_m = float(np.mean(x))
    x_d = float(np.var(x, ddof=1))

    print(f"n = {n}")
    print(f"λ = {lam}")
    print(f"Середнє m = {x_m:.6f}")
    print(f"Дисперсія D = {x_d:.6f}")

    rate_fixed = lam / 2.0
    print(f"\nrate = λ/2 = {rate_fixed:.6f}")
    print(f"E[X] = 2/λ   = {2/lam:.6f}")
    print(f"D[X] = 4/λ^2 = {4/(lam**2):.6f}")

    # 4-5) критерій χ²
    alpha = 0.05
    m = 12

    q = 0

    chi2_stat, chi2_crit, df, table, min_exp = х_square_verification(
        x, rate_used=rate_fixed, m=m, alpha=alpha, q=q
    )

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)

    print("\nχ² критерій")
    print(f"α = {alpha}, k = {m}, q = {q}, df = {df}")

    print(table.to_string(index=False))

    print(f"\nχ²_спост = {chi2_stat:.4f}")
    print(f"χ²_крит  = {chi2_crit:.4f}")

    if chi2_stat < chi2_crit:
        print("Висновок: χ²_спост < χ²_крит ⇒ НЕ відхиляємо H0 (вибірка узгоджується з експоненційним розподілом).")
    else:
        print("Висновок: χ²_спост ≥ χ²_крит ⇒ ВІДХИЛЯЄМО H0 (вибірка не узгоджується з експоненційним розподілом).")


if __name__ == "__main__":
    main()
