"""Pull FanDuel NHL shots-on-goal props from The Odds API and compare them to the model.

Setup:  export ODDS_API_KEY=your_key      (free key: https://the-odds-api.com)
Usage:  python src/odds.py [YYYY-MM-DD]            opening pull: price all props, log them (run predict.py first)
        python src/odds.py [YYYY-MM-DD] --close    closing pull: only games starting in the next 45 min

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
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(a < 0, -a / (-a + 100), 100 / (a + 100))
    return np.where(a < 0, -a / (-a + 100), 100 / (a + 100))


def payout(american):
    a = np.asarray(american, dtype=float)
    return np.where(a < 0, 100 / -a, a / 100)  # profit per $1 staked


def fetch(date, key, window_min=None, skip=()):
    """Pull FanDuel SOG props for unstarted games on `date`. With window_min, only games starting within
    that many minutes (closing-line mode). The events list is free; each game's odds cost 1 credit."""
    events = requests.get(f"{API}/events", params={"apiKey": key}, timeout=30)
    events.raise_for_status()
    rows, now = [], datetime.now(timezone.utc)
    for ev in events.json():
        start = datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00"))
        if start.astimezone(TZ).strftime("%Y-%m-%d") != date or start < now or ev["id"] in skip:
            continue
        if window_min is not None and (start - now).total_seconds() > window_min * 60:
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
    m = w.merge(proj[["key"] + (["playerId"] if "playerId" in proj else []) + ["name", "team", "opp", "proj_toi", "proj_sog"]], on="key", how="left")
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
    cols = ["game"] + (["playerId"] if "playerId" in m else []) + ["name", "team", "line", "over", "under", "proj_toi", "proj_sog", "model_over", "novig_over",
            "edge_over", "book_hold", "ev_over", "ev_under", "best_side", "best_ev", "commence_time"]
    return m[cols].sort_values("best_ev", ascending=False).round(3)


SNAP = ROOT / "data/odds_snapshots"
PROPS_LOG = ROOT / "results/props_log.csv"
PAPER_EV = 0.05  # every prop at or above this EV is tracked as a paper pick, whether or not you bet it


def save_snapshot(lines, date):
    stamp = datetime.now(TZ).strftime("%Y-%m-%dT%H%M")
    SNAP.mkdir(parents=True, exist_ok=True)
    lines.assign(pulled_at=stamp).to_csv(SNAP / f"{date}_{stamp}.csv", index=False)
    return stamp


def entry(date, key):
    """Opening pull: price every prop vs the model and log them all (paper picks flagged)."""
    proj_path = ROOT / f"results/projections_{date}.csv"
    if not proj_path.exists():
        print(f"odds: no projections for {date} (no games?)"); return None
    lines = fetch(date, key)
    if lines.empty:
        print("odds: FanDuel has no SOG props posted yet"); return None
    stamp = save_snapshot(lines, date)
    alpha = joblib.load(ROOT / "models/sog_gbm.joblib")["alpha"]
    e = edges(lines, pd.read_csv(proj_path), alpha)
    e.to_csv(ROOT / f"results/edges_{date}.csv", index=False)
    log = e.assign(date=date, pulled_at=stamp, paper_pick=e.best_ev >= PAPER_EV)
    if PROPS_LOG.exists():  # keep the FIRST price seen for each prop: that is the price a paper bet would get
        old = pd.read_csv(PROPS_LOG)
        log = pd.concat([old, log], ignore_index=True).drop_duplicates(["date", "name", "line"], keep="first")
    log.to_csv(PROPS_LOG, index=False)
    print(f"odds: {len(e)} props priced, {int((e.best_ev >= PAPER_EV).sum())} paper picks at {PAPER_EV:.0%}+ EV")
    return e


def close(date, key, window_min=45):
    """Closing pull: snapshot only games about to start, once each."""
    done_path = SNAP / f"closed_{date}.txt"
    done = set(done_path.read_text().split()) if done_path.exists() else set()
    lines = fetch(date, key, window_min=window_min, skip=done)
    if lines.empty:
        print("odds close: no games starting soon"); return
    save_snapshot(lines, date)
    SNAP.mkdir(parents=True, exist_ok=True)
    done_path.write_text("\n".join(sorted(done | set(lines.event_id))))
    print(f"odds close: saved closing lines for {lines.game.nunique()} game(s)")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    date = args[0] if args else datetime.now(TZ).strftime("%Y-%m-%d")
    key = os.environ.get("ODDS_API_KEY") or sys.exit("Set ODDS_API_KEY first (free key at the-odds-api.com)")
    if "--close" in sys.argv:
        close(date, key)
    else:
        e = entry(date, key)
        if e is not None:
            print(e[e.best_ev > 0.03][["game", "name", "line", "over", "under", "proj_sog", "model_over",
                                        "novig_over", "best_side", "best_ev"]].head(30).to_string(index=False))
