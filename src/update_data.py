"""Add newly completed games to data/games.parquet from the NHL's official API (api-web.nhle.com).

Usage: python src/update_data.py            (adds every finished regular-season game not yet in the file)
       python src/update_data.py 2026-11-01  (start scanning from this date; default = day after last game)

About 2 requests per game (box score + play-by-play), so a normal night is ~15-30 requests.
Per skater-game it records: TOI, shifts, shots on goal, missed shots, shots blocked by the opponent,
and total shot attempts (= SOG + missed + blocked), matching the columns of the historical file.
"""
import sys, time
from datetime import date, timedelta
from pathlib import Path
import pandas as pd
import requests

API = "https://api-web.nhle.com/v1"
ROOT = Path(__file__).resolve().parents[1]
S = requests.Session(); S.headers["User-Agent"] = "nhl-sog-model/1.0"


def get(path):
    for i in range(4):
        try:
            r = S.get(f"{API}/{path}", timeout=30)
            if r.status_code == 200:
                return r.json()
        except requests.RequestException:
            pass
        time.sleep(2 ** i)
    raise RuntimeError(f"NHL API failed: {path}")


def mmss(s):
    m, sec = s.split(":")
    return int(m) * 60 + int(sec)


def game_rows(gid):
    box, pbp = get(f"gamecenter/{gid}/boxscore"), get(f"gamecenter/{gid}/play-by-play")
    names = {p["playerId"]: f'{p["firstName"]["default"]} {p["lastName"]["default"]}' for p in pbp["rosterSpots"]}
    missed, blocked = {}, {}
    for e in pbp["plays"]:
        pid = e.get("details", {}).get("shootingPlayerId")
        if e["typeDescKey"] == "missed-shot" and pid:
            missed[pid] = missed.get(pid, 0) + 1
        elif e["typeDescKey"] == "blocked-shot" and pid:
            blocked[pid] = blocked.get(pid, 0) + 1
    start = int(str(box["season"])[:4])
    rows = []
    for side, opp_side, ha in (("homeTeam", "awayTeam", "HOME"), ("awayTeam", "homeTeam", "AWAY")):
        team, opp = box[side]["abbrev"], box[opp_side]["abbrev"]
        stats = box["playerByGameStats"][side]
        for p in stats["forwards"] + stats["defense"]:
            toi = mmss(p["toi"])
            if toi == 0:
                continue
            pid = p["playerId"]
            m, b = missed.get(pid, 0), blocked.get(pid, 0)
            rows.append({"playerId": pid, "season": start, "name": names.get(pid, p["name"]["default"]),
                         "gameId": gid, "playerTeam": team, "opposingTeam": opp, "home_or_away": ha,
                         "gameDate": pd.Timestamp(box["gameDate"]), "position": p["position"],
                         "icetime": float(toi), "shifts": float(p["shifts"]), "I_F_shotsOnGoal": float(p["sog"]),
                         "I_F_missedShots": float(m), "I_F_blockedShotAttempts": float(b),
                         "I_F_shotAttempts": float(p["sog"] + m + b), "source": "nhl_api"})
    return rows, names


def finished_games(start, end):
    out, d = [], start
    while d <= end:
        wk = get(f"schedule/{d.isoformat()}")
        for day in wk["gameWeek"]:
            dd = date.fromisoformat(day["date"])
            if start <= dd <= end:
                out += [g["id"] for g in day["games"] if g["gameType"] == 2 and g["gameState"] in ("OFF", "FINAL")]
        d += timedelta(days=7)
    return sorted(set(out))


if __name__ == "__main__":
    path = ROOT / "data/games.parquet"
    g = pd.read_parquet(path)
    start = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else (g.gameDate.max() + pd.Timedelta(days=1)).date()
    have = set(g.gameId.unique())
    todo = [x for x in finished_games(start, date.today()) if x not in have]
    print(f"{len(todo)} new games since {start}")
    rows, names = [], {}
    for gid in todo:
        r, n = game_rows(gid); rows += r; names.update(n); time.sleep(0.3)
    if rows:
        g = pd.concat([g, pd.DataFrame(rows)], ignore_index=True)
    # Current rosters (32 requests) give official names for every active player
    for t in g[g.season == g.season.max()].playerTeam.unique():
        r = get(f"roster/{t}/current")
        for k in ("forwards", "defensemen"):
            for p in r.get(k, []):
                names.setdefault(p["id"], f'{p["firstName"]["default"]} {p["lastName"]["default"]}')
    # Official names fix accented players that the historical file garbled (e.g. "Sttzle" -> "Stützle")
    g["name"] = g.playerId.map(names).fillna(g.name)
    g.to_parquet(path, index=False)
    print(g.shape, "through", g.gameDate.max().date())
