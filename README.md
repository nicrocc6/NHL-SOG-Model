# NHL Shots on Goal Model

Projects shots on goal (SOG) for every NHL skater in tonight's games. It gives the over/under probability and fair American odds for SOG prop lines (0.5 through 4.5), so you can compare against sportsbook prices.

## How it works

Shots on goal ≈ **ice time × shot rate × matchup**.

1. **Ice time:** Recent time on ice weighted toward the most recent games (5-, 15- and 40-game exponential averages), plus the last game and power-play time.
2. **Shot rate:** SOG per 60 and shot attempts per 60 over the same windows. A shrunk rate pulls players with little ice time toward the average for their position (forward or defence). Attempts are a steadier signal than SOG, so they help separate real skill from hot streaks.
3. **Matchup and context:** The opponent's recent shots allowed, the player's own team's shot volume, home/away, and back-to-backs for both teams.
4. **Model:** A gradient-boosted model with a Poisson loss predicts mean SOG. A **negative binomial** distribution (dispersion α ≈ 0.06) turns that mean into probabilities. Shot counts vary more than a Poisson distribution allows, so this prices overs and unders more accurately.
5. **Drift correction:** League shot rates move from season to season, so predictions are rescaled to match the current season's average.

Every feature uses only games played **before** the game being predicted. There is no lookahead.

## Data

- [MoneyPuck](https://moneypuck.com) game-by-game skater logs, 2022-23 to now: TOI, SOG, attempts, and 5-on-4 splits
- [NHL API](https://api-web.nhle.com) for tonight's schedule and current rosters

## Backtest

Trained on 2022-23 and 2023-24. The scale and dispersion were fit on 2024-25. Tested on games the model never saw.

| Test set | Model | Games | Log loss O1.5 | Log loss O2.5 | Log loss O3.5 | MAE |
|---|---|---|---|---|---|---|
| 2025-26 | Naive (last-15 avg SOG) | 44,202 | 0.6217 | 0.4696 | 0.2889 | 1.040 |
| 2025-26 | Structural (TOI × rate × opp) | 44,202 | 0.6086 | 0.4577 | 0.2816 | 1.013 |
| 2025-26 | **GBM (this model)** | 44,202 | **0.6064** | **0.4557** | **0.2795** | **1.010** |
| 2026-27 to date | Naive | 9,439 | 0.6191 | 0.4679 | 0.2825 | 1.027 |
| 2026-27 to date | **GBM (this model)** | 9,439 | **0.6038** | **0.4546** | **0.2720** | **0.996** |

Lower log loss is better. On the main lines the model beats a "recent average" baseline by about 3%. Most of that gain comes from the TOI × rate structure. The boosted model adds only a little on top of the structural one.

**Calibration (2025-26, over 2.5):** Predicted probabilities track actual hit rates closely. For example, picks rated 40–50% hit 46.5%, and picks rated 50–60% hit 58%. The model ran about 2% low across 2025-26 because shot rates rose that season. Live predictions use the current season's scale to correct for that. Full tables are in `results/`.

**What a backtest can't tell you:** Whether the model beats the books. That needs closing-line odds, which this data doesn't include. Log every bet with the line you took and the closing line, and judge the model by closing-line value (CLV) over 100+ picks.

## Usage

```bash
pip install -r requirements.txt
python src/fetch_data.py 2022     # ~2.5 min; rerun each game day for last night's games
python src/train.py               # backtest + saves models/sog_gbm.joblib (retrain every few weeks)
python src/predict.py             # today's projections -> results/projections_YYYY-MM-DD.csv
python src/predict.py 2026-11-12  # or a specific date

export ODDS_API_KEY=your_key      # free key at the-odds-api.com
python src/odds.py                # FanDuel SOG props vs model -> results/edges_YYYY-MM-DD.csv
python src/clv.py                 # grade bets logged in results/bets.csv: CLV + profit
```

## Odds and CLV

`odds.py` pulls FanDuel's `player_shots_on_goal` lines from [The Odds API](https://the-odds-api.com). For each player and line it reports:

- `model_over`: the model's probability of the over at FanDuel's line, including alternate lines
- `novig_over`: FanDuel's probability with the vig removed, and `book_hold`, the margin FanDuel charges
- `edge_over`: model minus no-vig
- `ev_over` / `ev_under`: expected profit per $1 at FanDuel's actual price
- `best_side` / `best_ev`: the better of the two sides

Each run costs 1 credit per game. The free plan's 500 credits a month covers about two pulls per game day. Every pull is saved to `data/odds_snapshots/`. **Run it once in the morning to find bets, and again just before puck drop to capture the closing line.**

Log bets in `results/bets.csv` (`date,name,line,side,price`). `clv.py` matches each bet to the last snapshot before puck drop. It reports CLV (closing no-vig probability minus the break-even probability of your price) and grades each bet against the actual result.

**Reading the edges:** An EV above 15% is more likely a missing scratch, a line change or a stale line than a real edge. Confirm lineups before betting. FanDuel's prices come from The Odds API's US feed, so FanDuel Ontario's prices can differ by a few cents. Always use the price you actually got in `bets.csv`.

## Known limitations / next steps

- **Lineups aren't confirmed.** The model includes rostered skaters who dressed in one of their team's last 3 games. Check morning skate and line rushes for scratches, injuries and line changes, especially promotions to the top power-play unit.
- **No goalie or game-total input yet.** Adding the betting total or a pace estimate is the most likely next improvement.
- **Linemate and deployment changes** (like a new top-line or PP1 role) only show up after a few games.
- Odds source is FanDuel US via The Odds API; set `ODDS_BOOK=draftkings` (etc.) to price another book.
