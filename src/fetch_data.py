"""ONE-TIME HISTORICAL BACKFILL ONLY. MoneyPuck asks anyone scraping regularly to get a data license
(moneypuck.com@gmail.com). Daily updates use src/update_data.py (official NHL API) instead.

Download MoneyPuck game-by-game skater logs and keep the columns the model needs.

Usage: python src/fetch_data.py [first_season]   (season = start year, e.g. 2023 for 2023-24)
Writes data/games.parquet: one row per player-game, with all-situations and 5on4 (power play) splits.
"""
import io, sys, time, concurrent.futures as cf
from pathlib import Path
import pandas as pd, requests

BASE = "https://moneypuck.com/moneypuck/playerData"
FIRST = int(sys.argv[1]) if len(sys.argv) > 1 else 2023
CURRENT = 2026
OUT = Path(__file__).resolve().parents[1] / "data"
KEEP = ["playerId", "season", "name", "gameId", "playerTeam", "opposingTeam", "home_or_away",
        "gameDate", "position", "icetime", "shifts", "I_F_shotsOnGoal", "I_F_shotAttempts",
        "I_F_missedShots", "I_F_blockedShotAttempts", "I_F_xGoals", "I_F_oZoneShiftStarts", "I_F_dZoneShiftStarts"]
S = requests.Session(); S.headers["User-Agent"] = "nhl-sog-model/1.0"

def get(url, tries=4):
    for i in range(tries):
        try:
            r = S.get(url, timeout=60)
            if r.status_code == 200: return r.text
            if r.status_code == 404: return None
        except requests.RequestException: pass
        time.sleep(2 ** i)
    return None

def player_ids():
    ids = set()
    for s in range(FIRST, CURRENT + 1):
        t = get(f"{BASE}/seasonSummary/{s}/regular/skaters.csv")
        ids |= set(pd.read_csv(io.StringIO(t))["playerId"].astype(int))
    return sorted(ids)

FAILED = []

def one(pid):
    t = get(f"{BASE}/careers/gameByGame/regular/skaters/{pid}.csv")
    if t is None:
        FAILED.append(pid); return None
    d = pd.read_csv(io.StringIO(t), usecols=lambda c: c in KEEP + ["situation"])
    d = d[(d.season >= FIRST) & d.situation.isin(["all", "5on4"])]
    if d.empty: return None
    a = d[d.situation == "all"].drop(columns="situation")
    pp = d[d.situation == "5on4"][["gameId", "icetime", "I_F_shotsOnGoal", "I_F_shotAttempts"]]
    pp.columns = ["gameId", "pp_toi", "pp_sog", "pp_att"]
    return a.merge(pp, on="gameId", how="left")

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    ids = player_ids(); print(len(ids), "players")
    frames = []
    with cf.ThreadPoolExecutor(8) as ex:
        for i, f in enumerate(ex.map(one, ids)):
            if f is not None: frames.append(f)
            if i % 200 == 0: print(i, flush=True)
    # Retry failures slowly, one at a time, so a rate-limit blip doesn't silently drop players
    for attempt in range(3):
        if not FAILED: break
        retry, FAILED[:] = list(FAILED), []
        print(f"retrying {len(retry)} players")
        for pid in retry:
            f = one(pid); time.sleep(0.5)
            if f is not None: frames.append(f)
    if FAILED:
        print(f"WARNING: {len(FAILED)} players still missing: {FAILED[:20]}")
    g = pd.concat(frames, ignore_index=True)
    g["gameDate"] = pd.to_datetime(g.gameDate.astype(str))
    g.to_parquet(OUT / "games.parquet", index=False)
    print(g.shape, g.gameDate.min(), g.gameDate.max())
