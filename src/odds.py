"""Pull FanDuel NHL shots-on-goal props from The Odds API and compare them to the model.

Setup:  export ODDS_API_KEY=your_key      (free key: https://the-odds-api.com)
Usage:  python src/odds.py [YYYY-MM-DD]    (run predict.py for the same date first)

Every run saves a timestamped snapshot of the lines to data/odds_snapshots/. Run it again just before
puck drop; the last snapshot before each game starts is used as the closing line for CLV in clv.py.
Writes results/edges_<date>.csv: model probability vs FanDuel's no-vig probability, edge and EV per side.

Cost: 1 credit per game per run (one market, one bookmaker). A 7-game night is 7 credits; the free plan
has 500 per month.
"""
import os, re, sys, unicodedata
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import requests
import joblib
from train import p_over

ROOT = Path(__file__).resolve().parents[1]
API = "https://api.the-odds-api.com/v4/sports/icehockey_nhl"
BOOK = os.environ.get("ODDS_BOOK", "fanduel")
MARKETS = "player_shots_on_goal"
TZ = ZoneInfo("America/Toronto")
ALIASES = {}  # book name -> model name, for any player whose names don't match after normalizing


def norm(name):
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\b(jr|sr|ii|iii)\b", "", s)
    return re.sub(r"[^a-z]", "", s)


def implied(american):
    a = np.asarray(american, dtype=float)
    return np.where(a < 0, -a / (-a + 100), 100 / (a + 100))


def payout(american):
    a = np.asarray(american, dtype=float)
    return np.where(a < 0, 100 / -a, a / 100)  # profit per $1 staked


def fetch(date, key):
    events = requests.get(f"{API}/events", params={"apiKey": key}, timeout=30)
    events.raise_for_status()
    rows = []
    for ev in events.json():
        start = datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00"))
        if start.astimezone(TZ).strftime("%Y-%m-%d") != date or start < datetime.now(timezone.utc):
            continue
        r = requests.get(f"{API}/events/{ev['id']}/odds", timeout=30, params={
            "apiKey": key, "bookmakers": BOOK, "markets": MARKETS, "oddsFormat": "american"})
        r.raise_for_status()
        print(f"{ev['away_team']} @ {ev['home_team']}  (credits left: {r.headers.get('x-requests-remaining')})")
        rows += parse_event(r.json())
    return pd.DataFrame(rows)


def parse_event(ev):
    rows = []
    for b in ev.get("bookmakers", []):
        for m in b["markets"]:
            for o in m["outcomes"]:
                rows.append({"event_id": ev["id"], "commence_time": ev["commence_time"],
                             "game": f"{ev['away_team']} @ {ev['home_team']}", "book": b["key"],
                             "market": m["key"], "player": o["description"], "side": o["name"].lower(),
                             "line": float(o["point"]), "price": int(o["price"]), "book_update": m["last_update"]})
    return rows


def edges(lines, proj, alpha):
    w = lines.pivot_table(index=["game", "commence_time", "player", "line"], columns="side",
                          values="price", aggfunc="first").reset_index()
    for side in ("over", "under"):
        if side not in w:
            w[side] = np.nan
    w["key"] = w.player.map(lambda n: norm(ALIASES.get(n, n)))
    proj = proj.assign(key=proj.name.map(norm))
    m = w.merge(proj[["key", "name", "team", "opp", "proj_toi", "proj_sog"]], on="key", how="left")
    unmatched = sorted(m[m.proj_sog.isna()].player.unique())
    if unmatched:
        print("No projection for (check lineup/name, add to ALIASES):", ", ".join(unmatched))
    m = m.dropna(subset=["proj_sog"]).copy()

    m["model_over"] = p_over(m.line.values, m.proj_sog.values, alpha)
    both = m.over.notna() & m.under.notna()
    io, iu = implied(m.over.fillna(100)), implied(m.under.fillna(100))
    m["book_hold"] = np.where(both, io + iu - 1, np.nan)
    m["novig_over"] = np.where(both, io / (io + iu), np.nan)  # one-sided lines: no fair line to compare
    m["edge_over"] = m.model_over - m.novig_over
    m["ev_over"] = np.where(m.over.notna(), m.model_over * payout(m.over.fillna(100)) - (1 - m.model_over), np.nan)
    m["ev_under"] = np.where(m.under.notna(), (1 - m.model_over) * payout(m.under.fillna(100)) - m.model_over, np.nan)
    m["best_side"] = np.where(m.ev_over.fillna(-9) >= m.ev_under.fillna(-9), "over", "under")
    m["best_ev"] = m[["ev_over", "ev_under"]].max(axis=1)
    cols = ["game", "name", "team", "line", "over", "under", "proj_toi", "proj_sog", "model_over", "novig_over",
            "edge_over", "book_hold", "ev_over", "ev_under", "best_side", "best_ev", "commence_time"]
    return m[cols].sort_values("best_ev", ascending=False).round(3)


if __name__ == "__main__":
    date = sys.argv[1] if len(sys.argv) > 1 else datetime.now(TZ).strftime("%Y-%m-%d")
    key = os.environ.get("ODDS_API_KEY") or sys.exit("Set ODDS_API_KEY first (free key at the-odds-api.com)")
    proj_path = ROOT / f"results/projections_{date}.csv"
    if not proj_path.exists():
        sys.exit(f"Run python src/predict.py {date} first")
    lines = fetch(date, key)
    if lines.empty:
        sys.exit("FanDuel has no shots-on-goal props posted yet for upcoming games on that date")
    stamp = datetime.now(TZ).strftime("%Y-%m-%dT%H%M")
    snap = ROOT / "data/odds_snapshots"; snap.mkdir(parents=True, exist_ok=True)
    lines.assign(pulled_at=stamp).to_csv(snap / f"{date}_{stamp}.csv", index=False)

    alpha = joblib.load(ROOT / "models/sog_gbm.joblib")["alpha"]
    e = edges(lines, pd.read_csv(proj_path), alpha)
    e.to_csv(ROOT / f"results/edges_{date}.csv", index=False)
    print(f"\n{len(e)} props priced. Best EV per $1 (bet only after confirming the player is in the lineup):")
    print(e[e.best_ev > 0.03][["game", "name", "line", "over", "under", "proj_sog", "model_over", "novig_over",
                                "best_side", "best_ev"]].head(30).to_string(index=False))
