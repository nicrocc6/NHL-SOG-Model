"""Rebuild the full skater game log from the official NHL API (one-time; ~2 requests per game).

Usage: python src/backfill_history.py 2022 [2026]   (season start years)
Writes data/games/<season>.csv.gz.
"""
import sys, time, concurrent.futures as cf
from datetime import date, timedelta
from pathlib import Path
import pandas as pd
from update_data import game_rows, finished_games

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/games"; OUT.mkdir(parents=True, exist_ok=True)
first = int(sys.argv[1]) if len(sys.argv) > 1 else 2022
last = int(sys.argv[2]) if len(sys.argv) > 2 else 2026

def safe(gid):
    for i in range(3):
        try:
            return game_rows(gid)[0]
        except Exception as e:
            time.sleep(5 * (i + 1))
    print("FAILED", gid, flush=True); return []

for season in range(first, last + 1):
    path = OUT / f"{season}.csv.gz"
    start, end = date(season, 9, 25), min(date(season + 1, 6, 30), date.today() - timedelta(days=1))
    ids = [g for g in finished_games(start, end) if str(g)[4:6] == "02"]
    print(season, len(ids), "games", flush=True)
    rows = []
    with cf.ThreadPoolExecutor(4) as ex:
        for i, r in enumerate(ex.map(safe, ids)):
            rows += r
            if i % 250 == 0: print(f"  {season}: {i}/{len(ids)}", flush=True)
    df = pd.DataFrame(rows).drop(columns="source")
    df.to_csv(path, index=False, compression="gzip")
    print(season, df.shape, flush=True)
