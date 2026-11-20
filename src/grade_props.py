"""Grade every FanDuel prop the model priced (not just bets you placed) and write the running scorecard.

Usage: python src/grade_props.py
Reads  results/props_log.csv (written by odds.py), data/odds_snapshots/, data/games/, results/bets.csv
Writes results/props_graded.csv and results/SUMMARY.md

Three questions it answers as the season goes on:
  1. Paper picks (every prop at 5%+ EV at the opening pull): record, units, CLV.
  2. Your real bets: record, units, CLV.
  3. Is the model better calibrated than FanDuel? Log loss of the model vs FanDuel's no-vig line on every prop.
"""
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
from odds import implied, payout, norm
from data_io import load_games

ROOT = Path(__file__).resolve().parents[1]
TZ = ZoneInfo("America/Toronto")


def closing_lines():
    files = [f for f in (ROOT / "data/odds_snapshots").glob("*.csv")]
    if not files:
        return pd.DataFrame(columns=["date", "key", "line", "close_over", "close_under"])
    s = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    s["pulled"] = pd.to_datetime(s.pulled_at, format="%Y-%m-%dT%H%M").dt.tz_localize(TZ)
    s["start"] = pd.to_datetime(s.commence_time).dt.tz_convert(TZ)
    s["date"] = s.start.dt.strftime("%Y-%m-%d")
    s["key"] = s.player.map(norm)
    s = s[s.pulled <= s.start].sort_values("pulled")
    c = s.groupby(["date", "key", "line", "side"]).price.last().unstack("side").reset_index()
    c.columns.name = None
    return c.rename(columns={"over": "close_over", "under": "close_under"})


def logloss(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def record(df, profit="profit", clv="clv", moved="moved_my_way"):
    d = df.dropna(subset=[profit])
    if d.empty:
        return "no graded bets yet"
    w, l = int((d[profit] > 0).sum()), int((d[profit] < 0).sum())
    return (f"{len(d)} bets, {w}-{l}, {d[profit].sum():+.2f} units, ROI {d[profit].mean():+.1%}, "
            f"avg CLV {d[clv].mean():+.1%}, line moved your way {d[moved].mean():.0%}")


if __name__ == "__main__":
    log_path = ROOT / "results/props_log.csv"
    if not log_path.exists():
        raise SystemExit("grade_props: nothing logged yet")
    today = datetime.now(TZ).strftime("%Y-%m-%d")
    p = pd.read_csv(log_path)
    p = p[p.date < today].copy()
    p["key"] = p.name.map(norm)
    p = p.merge(closing_lines(), on=["date", "key", "line"], how="left")

    g = load_games()
    g = g.assign(date=g.gameDate.dt.strftime("%Y-%m-%d"))
    sog = g[["playerId", "date", "I_F_shotsOnGoal"]].rename(columns={"I_F_shotsOnGoal": "sog"})
    p = p.merge(sog, on=["playerId", "date"], how="left")
    played_dates = set(g.date)
    p["status"] = np.where(p.sog.notna(), "graded", np.where(p.date.isin(played_dates), "void", "pending"))

    p["over_hit"] = np.where(p.sog.notna(), (p.sog > p.line).astype(float), np.nan)
    io, iu = implied(p.close_over), implied(p.close_under)
    p["novig_close_over"] = io / (io + iu)
    side_over = p.best_side == "over"
    p["pick_price"] = np.where(side_over, p.over, p.under)
    p["pick_close"] = np.where(side_over, p.close_over, p.close_under)
    p["clv"] = np.where(side_over, p.novig_close_over, 1 - p.novig_close_over) - implied(p.pick_price)
    p["moved_my_way"] = np.where(p.pick_close.notna(), implied(p.pick_close) > implied(p.pick_price), np.nan)
    won = np.where(side_over, p.over_hit == 1, p.over_hit == 0)
    p["profit"] = np.where(p.status == "graded", np.where(won, payout(p.pick_price), -1.0), np.nan)
    cols = ["date", "game", "playerId", "name", "line", "over", "under", "proj_sog", "model_over", "novig_over",
            "close_over", "close_under", "novig_close_over", "best_side", "best_ev", "paper_pick", "pick_price",
            "pick_close", "clv", "moved_my_way", "sog", "over_hit", "status", "profit"]
    p[cols].round(4).to_csv(ROOT / "results/props_graded.csv", index=False)

    gr = p[p.status == "graded"]
    paper = p[p.paper_pick == True]
    lines = [f"# Model scorecard", "", f"_Updated {datetime.now(TZ):%Y-%m-%d %H:%M} ET by the nightly job._", "",
             "## Paper picks (every FanDuel prop at 5%+ EV at the morning pull)", "",
             f"- All time: {record(paper)}"]
    for days in (7, 30):
        since = (pd.Timestamp(today) - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
        lines.append(f"- Last {days} days: {record(paper[paper.date >= since])}")
    bg = ROOT / "results/bets_graded.csv"
    if bg.exists():
        lines += ["", "## Your real bets", "", f"- {record(pd.read_csv(bg))}"]
    lines += ["", "## Model vs FanDuel on every graded prop", ""]
    have_close = gr.dropna(subset=["novig_close_over", "novig_over"])
    if len(have_close):
        y = have_close.over_hit.values
        lm, lo, lc = (logloss(have_close.model_over.values, y), logloss(have_close.novig_over.values, y),
                      logloss(have_close.novig_close_over.values, y))
        lb = logloss(0.5 * have_close.model_over.values + 0.5 * have_close.novig_over.values, y)
        lines += [f"Props graded: {len(have_close)} (lower log loss = better probabilities)", "",
                  "| Forecast | Log loss |", "|---|---|",
                  f"| Model | {lm:.4f} |", f"| FanDuel no-vig, morning | {lo:.4f} |",
                  f"| FanDuel no-vig, closing | {lc:.4f} |", f"| 50/50 blend of model + morning line | {lb:.4f} |", "",
                  "If the model's number drops below FanDuel's closing line over a few hundred props, it is beating the market."]
    else:
        lines.append("Not enough graded props with closing lines yet.")
    (ROOT / "results/SUMMARY.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
