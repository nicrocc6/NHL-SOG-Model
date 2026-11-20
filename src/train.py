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
from data_io import load_games

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


PARAMS = dict(loss="poisson", max_iter=600, learning_rate=0.03, max_leaf_nodes=31, min_samples_leaf=200,
              l2_regularization=1.0, early_stopping=True, validation_fraction=0.15, random_state=0)


def retrain(backtest=True):
    g = load_games()
    d = build(g)
    d = d[(d.n_prior >= 10) & d.opp_sog_allowed.notna()].copy()
    tr, cal = d[d.season.isin([2022, 2023])], d[d.season == 2024]
    gbm = HistGradientBoostingRegressor(**PARAMS).fit(tr[FEATURES], tr.sog)
    preds = {
        "Naive (last-15 avg SOG)": lambda x: x.sog_ewm15.clip(lower=0.05).values,
        "Structural (TOI x rate x opp)": lambda x: x.mu_struct.clip(lower=0.05).values,
        "GBM (final)": lambda x: np.clip(gbm.predict(x[FEATURES]), 0.05, None),
    }
    # League-level drift: rescale each model so its mean matches the calibration season, then fit dispersion
    scales = {k: float(cal.sog.mean() / f(cal).mean()) for k, f in preds.items()}
    preds = {k: (lambda x, f=f, s=scales[k]: f(x) * s) for k, f in preds.items()}
    alphas = {k: fit_alpha(cal.sog.values, f(cal)) for k, f in preds.items()}
    (ROOT / "results").mkdir(exist_ok=True)

    if backtest:
        tests = {"2025-26": d[d.season == 2025], "2026-27 (to date)": d[d.season == 2026]}
        rows = []
        for tname, t in tests.items():
            for k, f in preds.items():
                r = evaluate(t, f(t), alphas[k], k); r["test"] = tname; rows.append(r)
            sub = t[t.toi_proj >= 15]
            r = evaluate(sub, preds["GBM (final)"](sub), alphas["GBM (final)"], "GBM (final), TOI>=15 only")
            r["test"] = tname; rows.append(r)
        res = pd.DataFrame(rows)
        pd.set_option("display.width", 250)
        print(res[["test", "model", "n", "nb_loglik_per_game", "mae", "logloss_o1.5", "logloss_o2.5",
                   "logloss_o3.5"]].round(4).to_string(index=False))
        res.to_csv(ROOT / "results/backtest_metrics.csv", index=False)
        t = tests["2025-26"]
        calib = {L: calib_table(t, preds["GBM (final)"](t), alphas["GBM (final)"], L) for L in [1.5, 2.5, 3.5]}
        pd.concat({f"over_{L}": c for L, c in calib.items()}).to_csv(ROOT / "results/calibration_2025_26.csv")
        json.dump(alphas, open(ROOT / "results/alphas.json", "w"), indent=1)

    # Live model: learn from EVERY game up to yesterday. The drift scale comes from a twin model that has
    # not seen the last 30 days, scored on those 30 days (out of sample), so it reflects current shot rates.
    cutoff = d.gameDate.max() - pd.Timedelta(days=30)
    early, recent = d[d.gameDate <= cutoff], d[d.gameDate > cutoff]
    twin = HistGradientBoostingRegressor(**PARAMS).fit(early[FEATURES], early.sog)
    mu_recent = np.clip(twin.predict(recent[FEATURES]), 0.05, None)
    scale = float(recent.sog.mean() / mu_recent.mean())
    alpha = alphas["GBM (final)"]
    holdout = evaluate(recent, mu_recent * scale, alpha, "last-30-days holdout")
    final = HistGradientBoostingRegressor(**PARAMS).fit(d[FEATURES], d.sog)
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump({"model": final, "alpha": alpha, "scale": scale, "features": FEATURES,
                 "trained_through": str(d.gameDate.max().date()), "n_rows": int(len(d))}, ROOT / "models/sog_gbm.joblib")
    hist = ROOT / "results/model_history.csv"
    row = pd.DataFrame([{"trained_on": pd.Timestamp.now(tz="America/Toronto").strftime("%Y-%m-%d"),
                         "data_through": str(d.gameDate.max().date()), "training_rows": len(d), "scale": round(scale, 4),
                         "alpha": round(alpha, 4), "holdout_games": holdout["n"],
                         "holdout_logloss_o1.5": round(holdout["logloss_o1.5"], 4),
                         "holdout_logloss_o2.5": round(holdout["logloss_o2.5"], 4),
                         "holdout_logloss_o3.5": round(holdout["logloss_o3.5"], 4), "holdout_mae": round(holdout["mae"], 4)}])
    row.to_csv(hist, mode="a", header=not hist.exists(), index=False)
    print("retrained:", row.to_dict("records")[0])


if __name__ == "__main__":
    retrain(backtest=True)
