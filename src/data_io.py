"""Game-log storage: one gzipped CSV per season in data/games/ (small enough to live in the repo)."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GAMES = ROOT / "data/games"


def load_games() -> pd.DataFrame:
    files = sorted(GAMES.glob("*.csv.gz"))
    if not files:
        raise SystemExit("No game data in data/games/. Run python src/backfill_history.py first.")
    g = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    g["gameDate"] = pd.to_datetime(g.gameDate)
    return g.drop_duplicates(["gameId", "playerId"], keep="last")


def save_games(g: pd.DataFrame) -> None:
    GAMES.mkdir(parents=True, exist_ok=True)
    for season, part in g.groupby("season"):
        part = part.sort_values(["gameDate", "gameId", "playerId"])
        part.assign(gameDate=part.gameDate.dt.strftime("%Y-%m-%d")).to_csv(
            GAMES / f"{season}.csv.gz", index=False, compression="gzip")
