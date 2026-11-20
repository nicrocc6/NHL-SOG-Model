"""Add newly completed games to data/games/<season>.csv.gz from the NHL's official API (api-web.nhle.com).

Usage: python src/update_data.py            (adds every finished regular-season game not yet stored)
       python src/update_data.py 2026-11-01  (start scanning from this date)

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


def update(start=None):
    """Append every finished regular-season game not yet stored. Returns the number of games added."""
    from data_io import load_games, save_games
    g = load_games()
    start = start or (g.gameDate.max() - pd.Timedelta(days=3)).date()  # small overlap catches late-final games
    have = set(g.gameId.unique())
    todo = [x for x in finished_games(start, date.today()) if x not in have]
    rows = []
    for gid in todo:
        rows += game_rows(gid)[0]; time.sleep(0.3)
    if rows:
        new = pd.DataFrame(rows).drop(columns="source")
        save_games(pd.concat([g, new], ignore_index=True))
    print(f"update_data: {len(todo)} new games, data through "
          f"{max(g.gameDate.max(), pd.Timestamp(rows[-1]['gameDate']) if rows else g.gameDate.max()).date()}")
    return len(todo)


if __name__ == "__main__":
    update(date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else None)
