"""Train the SOG model and backtest it.

Split (by season start year):  train 2022-23 + 2023-24 | dispersion fit 2024-25 | test 2025-26 and 2026-27 to date
Run: python src/train.py
"""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy import stats, optimize
from sklearn.ensemble import HistGradientBoostingRegressor
from features import build, FEATURES

ROOT = Path(__file__).resolve().parents[1]
LINES = [0.5, 1.5, 2.5, 3.5, 4.5]


def nb_cdf(k, mu, alpha):
    """P(X <= k) for NB2 with mean mu and variance mu + alpha*mu^2."""
    n = 1.0 / alpha
    return stats.nbinom.cdf(k, n, n / (n + mu))


def nb_logpmf(y, mu, alpha):
    n = 1.0 / alpha
    return stats.nbinom.logpmf(y, n, n / (n + mu))


def fit_alpha(y, mu):
    res = optimize.minimize_scalar(lambda a: -nb_logpmf(y, mu, a).sum(), bounds=(1e-4, 2), method="bounded")
    return float(res.x)


def p_over(line, mu, alpha):
    return 1 - nb_cdf(np.floor(line), mu, alpha)


def evaluate(df, mu, alpha, label):
    y = df.sog.values
    out = {"model": label, "n": len(y), "nb_loglik_per_game": float(nb_logpmf(y, mu, alpha).mean()),
           "mae": float(np.abs(y - mu).mean()), "mean_pred": float(mu.mean()), "mean_actual": float(y.mean())}
    for L in LINES:
        p = np.clip(p_over(L, mu, alpha), 1e-6, 1 - 1e-6)
        o = (y > L).astype(float)
        out[f"logloss_o{L}"] = float(-(o * np.log(p) + (1 - o) * np.log(1 - p)).mean())
        out[f"brier_o{L}"] = float(((p - o) ** 2).mean())
    return out


def calib_table(df, mu, alpha, line):
    p = p_over(line, mu, alpha)
    o = (df.sog.values > line).astype(float)
    bins = pd.cut(p, np.arange(0, 1.05, 0.1))
    return (pd.DataFrame({"bin": bins, "p": p, "o": o}).groupby("bin", observed=True)
              .agg(n=("o", "size"), predicted=("p", "mean"), actual=("o", "mean")).round(3))


if __name__ == "__main__":
    g = pd.read_parquet(ROOT / "data/games.parquet")
    d = build(g)
    d = d[(d.n_prior >= 10) & d.opp_sog_allowed.notna()].copy()
    tr, cal = d[d.season.isin([2022, 2023])], d[d.season == 2024]
    tests = {"2025-26": d[d.season == 2025], "2026-27 (to date)": d[d.season == 2026]}

    gbm = HistGradientBoostingRegressor(loss="poisson", max_iter=600, learning_rate=0.03, max_leaf_nodes=31,
                                        min_samples_leaf=200, l2_regularization=1.0, early_stopping=True,
                                        validation_fraction=0.15, random_state=0)
    gbm.fit(tr[FEATURES], tr.sog)

    preds = {
        "Naive (last-15 avg SOG)": lambda x: x.sog_ewm15.clip(lower=0.05).values,
        "Structural (TOI x rate x opp)": lambda x: x.mu_struct.clip(lower=0.05).values,
        "GBM (final)": lambda x: np.clip(gbm.predict(x[FEATURES]), 0.05, None),
    }
    # League-level drift: rescale each model so its mean matches the calibration season, then fit dispersion
    scales = {k: float(cal.sog.mean() / f(cal).mean()) for k, f in preds.items()}
    preds = {k: (lambda x, f=f, s=scales[k]: f(x) * s) for k, f in preds.items()}
    print("scale:", {k: round(v, 3) for k, v in scales.items()})
    alphas = {k: fit_alpha(cal.sog.values, f(cal)) for k, f in preds.items()}
    print("NB dispersion alpha:", {k: round(v, 3) for k, v in alphas.items()})

    rows = []
    for tname, t in tests.items():
        for k, f in preds.items():
            r = evaluate(t, f(t), alphas[k], k); r["test"] = tname; rows.append(r)
        # Props-relevant subset: regular top-9 forwards / top-4 D
        sub = t[t.toi_proj >= 15]
        r = evaluate(sub, preds["GBM (final)"](sub), alphas["GBM (final)"], "GBM (final), TOI>=15 only"); r["test"] = tname; rows.append(r)
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(res[["test", "model", "n", "nb_loglik_per_game", "mae", "mean_pred", "mean_actual",
               "logloss_o1.5", "logloss_o2.5", "logloss_o3.5", "brier_o2.5"]].round(4).to_string(index=False))

    t = tests["2025-26"]
    mu = preds["GBM (final)"](t)
    calib = {L: calib_table(t, mu, alphas["GBM (final)"], L) for L in [1.5, 2.5, 3.5]}
    for L, c in calib.items():
        print(f"\nCalibration 2025-26, over {L}:\n{c}")

    # Refit on everything through 2025-26 for live use; keep the dispersion from the holdout fit
    final = HistGradientBoostingRegressor(**gbm.get_params()).fit(
        d[d.season <= 2025][FEATURES], d[d.season <= 2025].sog)
    (ROOT / "models").mkdir(exist_ok=True)
    recent = d[d.season == 2026]
    live_scale = float(recent.sog.mean() / np.clip(final.predict(recent[FEATURES]), 0.05, None).mean())
    print("live scale (refit model vs 2026-27 to date):", round(live_scale, 3))
    joblib.dump({"model": final, "alpha": alphas["GBM (final)"], "scale": live_scale, "features": FEATURES}, ROOT / "models/sog_gbm.joblib")
    (ROOT / "results").mkdir(exist_ok=True)
    res.to_csv(ROOT / "results/backtest_metrics.csv", index=False)
    pd.concat({f"over_{L}": c for L, c in calib.items()}).to_csv(ROOT / "results/calibration_2025_26.csv")
    pd.DataFrame({"name": t.name, "gameDate": t.gameDate, "team": t.playerTeam, "opp": t.opposingTeam,
                  "mu": mu, "sog": t.sog}).to_csv(ROOT / "results/preds_2025_26.csv", index=False)
    json.dump(alphas, open(ROOT / "results/alphas.json", "w"), indent=1)
