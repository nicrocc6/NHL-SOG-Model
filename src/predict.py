"""Project shots on goal for every skater playing on a given date.

Usage: python src/predict.py [YYYY-MM-DD]      (default: today, Toronto time)
Run src/fetch_data.py first on game days so the logs include last night's games.
Writes results/projections_<date>.csv with mean SOG, P(over) and fair American odds for common lines.
"""
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import joblib
import numpy as np
import pandas as pd
import requests
from features import build
from train import p_over

ROOT = Path(__file__).resolve().parents[1]
API = "https://api-web.nhle.com/v1"
LINES = [0.5, 1.5, 2.5, 3.5, 4.5]


def american(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.where(p >= 0.5, -100 * p / (1 - p), 100 * (1 - p) / p).round().astype(int)


def schedule(date):
    day = requests.get(f"{API}/schedule/{date}", timeout=30).json()["gameWeek"][0]
    assert day["date"] == date, f"no games found for {date}"
    return [(x["id"], x["awayTeam"]["abbrev"], x["homeTeam"]["abbrev"]) for x in day["games"] if x["gameType"] == 2]


def roster_ids(team):
    r = requests.get(f"{API}/roster/{team}/current", timeout=30).json()
    return {p["id"] for k in ("forwards", "defensemen") for p in r.get(k, [])}


if __name__ == "__main__":
    date = sys.argv[1] if len(sys.argv) > 1 else datetime.now(ZoneInfo("America/Toronto")).strftime("%Y-%m-%d")
    bundle = joblib.load(ROOT / "models/sog_gbm.joblib")
    g = pd.read_parquet(ROOT / "data/games.parquet")
    g = g[g.gameDate < pd.Timestamp(date)]
    season = int(g.season.max())
    last = g.sort_values("gameDate").groupby("playerId").tail(1).set_index("playerId")

    new = []
    for gid, away, home in schedule(date):
        for team, opp, ha in ((away, home, "AWAY"), (home, away, "HOME")):
            # Skaters on the current roster who dressed in at least one of the team's last 3 games
            recent = g[g.playerTeam == team].gameId.drop_duplicates().nlargest(3)
            dressed = set(g[g.gameId.isin(recent) & (g.playerTeam == team)].playerId)
            for pid in roster_ids(team) & dressed:
                r = last.loc[pid]
                new.append({"playerId": pid, "season": season, "name": r["name"], "gameId": gid, "playerTeam": team,
                            "opposingTeam": opp, "home_or_away": ha, "gameDate": pd.Timestamp(date),
                            "position": r["position"], "icetime": np.nan, "I_F_shotsOnGoal": np.nan,
                            "I_F_shotAttempts": np.nan, "pp_toi": np.nan})
    new = pd.DataFrame(new)
    d = build(pd.concat([g, new], ignore_index=True))
    d = d[(d.gameDate == pd.Timestamp(date)) & (d.n_prior >= 10)].copy()

    d["proj_sog"] = np.clip(bundle["model"].predict(d[bundle["features"]]), 0.05, None) * bundle["scale"]
    out = d[["name", "playerTeam", "opposingTeam", "home_or_away", "position", "toi_proj", "pp_toi_ewm15", "proj_sog"]].rename(
        columns={"playerTeam": "team", "opposingTeam": "opp", "home_or_away": "venue", "toi_proj": "proj_toi",
                 "pp_toi_ewm15": "pp_toi_recent"})
    for L in LINES:
        p = p_over(L, out.proj_sog.values, bundle["alpha"])
        out[f"p_over_{L}"] = p.round(3)
        out[f"fair_over_{L}"] = american(p)
        out[f"fair_under_{L}"] = american(1 - p)
    out = out.sort_values("proj_sog", ascending=False).round(2)
    (ROOT / "results").mkdir(exist_ok=True)
    path = ROOT / f"results/projections_{date}.csv"
    out.to_csv(path, index=False)
    print(f"{len(out)} skaters -> {path}")
    print(out[["name", "team", "opp", "proj_toi", "proj_sog", "p_over_2.5", "fair_over_2.5", "p_over_3.5", "fair_over_3.5"]]
          .head(25).to_string(index=False))
