"""Grade logged bets: closing-line value (CLV) and results.

Log each bet you place in results/bets.csv (create it with this header):
    date,name,line,side,price
    2026-11-10,Nathan MacKinnon,2.5,over,-150

Usage: python src/update_data.py && python src/clv.py
Closing line = the last saved odds snapshot taken before puck drop (so run odds.py close to game time).
CLV = closing no-vig probability of your side minus the break-even probability of the price you took.
Positive average CLV over 100+ bets is the best evidence the model has a real edge.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from odds import implied, norm, payout

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    bets = pd.read_csv(ROOT / "results/bets.csv")
    bets["key"] = bets.name.map(norm)
    snaps = pd.concat([pd.read_csv(f) for f in (ROOT / "data/odds_snapshots").glob("*.csv")], ignore_index=True)
    snaps["key"] = snaps.player.map(norm)
    snaps["pulled"] = pd.to_datetime(snaps.pulled_at, format="%Y-%m-%dT%H%M").dt.tz_localize("America/Toronto")
    snaps["start"] = pd.to_datetime(snaps.commence_time).dt.tz_convert("America/Toronto")
    snaps["date"] = snaps.start.dt.strftime("%Y-%m-%d")
    pre = snaps[snaps.pulled <= snaps.start].sort_values("pulled")
    close = (pre.groupby(["date", "key", "line", "side"]).price.last().unstack("side").reset_index())
    close.columns.name = None
    close = close.rename(columns={"over": "close_over", "under": "close_under"})

    b = bets.merge(close, on=["date", "key", "line"], how="left")
    io, iu = implied(b.close_over), implied(b.close_under)
    nv_over = io / (io + iu)
    b["close_price"] = np.where(b.side == "over", b.close_over, b.close_under)
    b["close_novig"] = np.where(b.side == "over", nv_over, 1 - nv_over)
    b["clv"] = b.close_novig - implied(b.price)
    b["moved_my_way"] = implied(b.close_price) > implied(b.price)  # close got more expensive = you got the better number

    g = pd.read_parquet(ROOT / "data/games.parquet")
    g = g.assign(key=g.name.map(norm), date=g.gameDate.dt.strftime("%Y-%m-%d"))
    b = b.merge(g[["key", "date", "I_F_shotsOnGoal"]].rename(columns={"I_F_shotsOnGoal": "sog"}),
                on=["key", "date"], how="left")
    won = np.where(b.side == "over", b.sog > b.line, b.sog < b.line)
    b["profit"] = np.where(b.sog.isna(), np.nan, np.where(won, payout(b.price), -1.0))

    out = b[["date", "name", "line", "side", "price", "close_price", "clv", "moved_my_way", "sog", "profit"]].round(3)
    out.to_csv(ROOT / "results/bets_graded.csv", index=False)
    print(out.to_string(index=False))
    done = out.dropna(subset=["profit"])
    print(f"\nBets: {len(out)} | avg CLV: {out.clv.mean():+.1%} | beat close: {(out.clv > 0).mean():.0%}"
          f" | price moved my way: {out.moved_my_way.mean():.0%}"
          f" | graded: {len(done)} | units: {done.profit.sum():+.2f} | ROI: {done.profit.mean():+.1%}")
