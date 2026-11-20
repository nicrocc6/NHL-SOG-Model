# NHL Shots on Goal Model

Projects shots on goal (SOG) for every NHL skater in tonight's games. It gives the over/under probability and fair American odds for SOG prop lines (0.5 through 4.5), so you can compare against sportsbook prices.

**It runs itself.** A GitHub Actions job runs every day. It adds last night's games, grades every FanDuel prop it priced, retrains on Mondays, projects today's slate and records opening and closing lines. Everything is committed back to this repo. **Live scorecard: [`results/SUMMARY.md`](results/SUMMARY.md)**

## Automation

`.github/workflows/daily.yml`, all times Eastern:

| When | What |
|---|---|
| 9:00 AM daily | `daily.py morning`: add last night's games from the NHL API, grade logged bets and every priced prop, retrain on Mondays, project today's skaters, pull FanDuel's opening lines and log every prop (5%+ EV props become **paper picks**) |
| Every 30 min, 11 AM–10:30 PM | `daily.py close`: save FanDuel's closing lines for games starting within 45 min, once per game |

- **Learning:** The live model is refit every Monday on every game through the night before. The drift scale is set out of sample on the last 30 days. Each retrain appends a row to `results/model_history.csv`, so you can see whether the holdout error is improving.
- **Tracking with no bets placed:** Every prop is logged in `results/props_log.csv` and graded in `results/props_graded.csv`, with the model's probability, FanDuel's opening and closing no-vig prices, the result and CLV. `SUMMARY.md` compares the model's log loss to FanDuel's line on every graded prop. That comparison shows whether the model is beating the market.
- **Game lines:** each morning, FanDuel's moneyline and total for every game are saved to `data/game_lines/` (2 credits). Game pace isn't in the model yet. This builds the history to train on it, which can't be pulled later on the free plan.
- **Odds API cost:** 1 credit per game for the opening pull and 1 for the close, plus 2 a day for game lines. That's about 480 a month across a full NHL schedule (the free plan has 500). The key is stored as the repo secret `ODDS_API_KEY`.
- **Manual run:** Actions → Nightly model run → Run workflow.

## How it works

Shots on goal ≈ **ice time × shot rate × matchup**.

1. **Ice time:** Recent time on ice weighted toward the most recent games (5-, 15- and 40-game exponential averages), plus the last game. (Power-play time was tested and dropped. It changed backtest log loss by just 0.0003, because total TOI already captures it.)
2. **Shot rate:** SOG per 60 and shot attempts per 60 over the same windows. A shrunk rate pulls players with little ice time toward the average for their position (forward or defence). Attempts are a steadier signal than SOG, so they help separate real skill from hot streaks.
3. **Matchup and context:** The opponent's recent shots allowed, the player's own team's shot volume, home/away, and back-to-backs for both teams.
4. **Model:** A gradient-boosted model with a Poisson loss predicts mean SOG. A **negative binomial** distribution (dispersion α ≈ 0.06) turns that mean into probabilities. Shot counts vary more than a Poisson distribution allows, so this prices overs and unders more accurately.
5. **Drift correction:** League shot rates move from season to season, so predictions are rescaled to match the current season's average.

Every feature uses only games played **before** the game being predicted. There is no lookahead.

## Data

All game data comes from the official [NHL API](https://api-web.nhle.com): box scores (TOI, SOG, shifts) and play-by-play (missed and blocked attempts). It's stored in `data/games/<season>.csv.gz`, 2022-23 to now, and `backfill_history.py` rebuilds it. As a check, it was compared to [MoneyPuck](https://moneypuck.com) on 199,580 player-games: SOG 99.95% and attempts 99.72% identical, TOI 100% within 5 seconds. (The model was first built on MoneyPuck data. MoneyPuck asks anyone pulling its data regularly to get a license, so it is no longer used.)

## Backtest

Trained on 2022-23 and 2023-24. The scale and dispersion were fit on 2024-25. Tested on games the model never saw.

| Test set | Model | Games | Log loss O1.5 | Log loss O2.5 | Log loss O3.5 | MAE |
|---|---|---|---|---|---|---|
| 2025-26 | Naive (last-15 avg SOG) | 46,061 | 0.6224 | 0.4692 | 0.2885 | 1.040 |
| 2025-26 | Structural (TOI × rate × opp) | 46,061 | 0.6091 | 0.4573 | 0.2811 | 1.013 |
| 2025-26 | **GBM (this model)** | 46,061 | **0.6072** | **0.4553** | **0.2793** | **1.010** |
| 2026-27 to Nov 19 | Naive | 12,463 | 0.6218 | 0.4671 | 0.2823 | 1.026 |
| 2026-27 to Nov 19 | **GBM (this model)** | 12,463 | **0.6066** | **0.4522** | **0.2718** | **0.994** |

Lower log loss is better. On the main lines the model beats a "recent average" baseline by about 3%. Most of that gain comes from the TOI × rate structure. The boosted model adds only a little on top of the structural one.

**Calibration (2025-26, over 2.5):** Predicted probabilities track actual hit rates closely. For example, picks rated 40–50% hit 46.5%, and picks rated 50–60% hit 58%. The model ran about 2% low across 2025-26 because shot rates rose that season. Live predictions use the current season's scale to correct for that. Full tables are in `results/`.

**What a backtest can't tell you:** Whether the model beats the books. That's what the nightly prop tracking measures. See `results/SUMMARY.md`.

## Usage

```bash
pip install -r requirements.txt
python src/daily.py morning       # the whole morning job (what GitHub runs at 9 AM)
python src/update_data.py        # add last night's games from the NHL API
python src/train.py               # backtest + refit the live model on all data
python src/predict.py             # today's projections -> results/projections_YYYY-MM-DD.csv
python src/predict.py 2026-11-12  # or a specific date

export ODDS_API_KEY=your_key      # free key at the-odds-api.com
python src/odds.py                # FanDuel SOG props vs model -> results/edges_YYYY-MM-DD.csv + props log
python src/odds.py --close        # closing lines for games starting in the next 45 min
python src/clv.py                 # grade bets logged in results/bets.csv: CLV + profit
python src/grade_props.py         # grade every logged prop, write results/SUMMARY.md
```

## Odds and CLV

`odds.py` pulls FanDuel's `player_shots_on_goal` lines from [The Odds API](https://the-odds-api.com). For each player and line it reports:

- `model_over`: the model's probability of the over at FanDuel's line, including alternate lines
- `novig_over`: FanDuel's probability with the vig removed, and `book_hold`, the margin FanDuel charges
- `edge_over`: model minus no-vig
- `ev_over` / `ev_under`: expected profit per $1 at FanDuel's actual price
- `best_side` / `best_ev`: the better of the two sides

Each pull costs 1 credit per game. Every pull is saved to `data/odds_snapshots/`. The nightly job handles both the morning pull and the closing pulls.

Log bets in `results/bets.csv` (`date,name,line,side,price`). `clv.py` matches each bet to the last snapshot before puck drop. It reports CLV (closing no-vig probability minus the break-even probability of your price) and grades each bet against the actual result.

**Reading the edges:** An EV above 15% is more likely a missing scratch, a line change or a stale line than a real edge. Confirm lineups before betting. FanDuel's prices come from The Odds API's US feed, so FanDuel Ontario's prices can differ by a few cents. Always use the price you actually got in `bets.csv`.

## Known limitations / next steps

- **Lineups aren't confirmed.** The model includes rostered skaters who dressed in one of their team's last 3 games. Check morning skate and line rushes for scratches, injuries and line changes, especially promotions to the top power-play unit.
- **No goalie or game-total input yet.** Adding the betting total or a pace estimate is the most likely next improvement.
- **Linemate and deployment changes** (like a new top-line or PP1 role) only show up after a few games.
- Odds source is FanDuel US via The Odds API; set `ODDS_BOOK=draftkings` (etc.) to price another book.
