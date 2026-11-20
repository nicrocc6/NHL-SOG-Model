"""Nightly automation entry point (run by GitHub Actions, or by hand).

  python src/daily.py morning   add last night's games, grade bets + all props, retrain on Mondays,
                                project today's games, opening odds pull (logs every prop + paper picks)
  python src/daily.py close     closing-line pull for games starting in the next 45 minutes

Each step is independent: one failing (e.g. no props posted yet) doesn't stop the rest.
"""
import os, subprocess, sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TZ = ZoneInfo("America/Toronto")


def run(*args):
    print(f"\n=== {' '.join(args)}", flush=True)
    r = subprocess.run([sys.executable, *args], cwd=ROOT)
    if r.returncode != 0:
        print(f"!!! step failed (exit {r.returncode}): {' '.join(args)}", flush=True)
    return r.returncode == 0


def needs_retrain():
    path = ROOT / "models/sog_gbm.joblib"
    if not path.exists() or datetime.now(TZ).weekday() == 0:  # every Monday
        return True
    b = joblib.load(path)
    through = b.get("trained_through")
    return through is None or (datetime.now(TZ).date() - datetime.fromisoformat(through).date()).days > 8


def morning():
    today = datetime.now(TZ).strftime("%Y-%m-%d")
    ok = run(str(SRC / "update_data.py"))
    if ok and needs_retrain():
        run("-c", "import sys; sys.path.insert(0, 'src'); from train import retrain; retrain(backtest=True)")
    if (ROOT / "results/bets.csv").exists():
        run(str(SRC / "clv.py"))
    run(str(SRC / "grade_props.py"))
    if run(str(SRC / "predict.py"), today) and os.environ.get("ODDS_API_KEY"):
        run(str(SRC / "odds.py"), today)


def close():
    if os.environ.get("ODDS_API_KEY"):
        run(str(SRC / "odds.py"), datetime.now(TZ).strftime("%Y-%m-%d"), "--close")


if __name__ == "__main__":
    {"morning": morning, "close": close}[sys.argv[1] if len(sys.argv) > 1 else "morning"]()
