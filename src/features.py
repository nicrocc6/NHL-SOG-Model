"""Build pre-game features. Every feature uses only games played BEFORE the game being predicted."""
import numpy as np
import pandas as pd

SPANS = (5, 15, 40)


def team_games(g: pd.DataFrame) -> pd.DataFrame:
    """Team-level shots for/against per game, built by summing skater rows."""
    t = (g.groupby(["gameId", "gameDate", "season", "playerTeam", "opposingTeam", "home_or_away"], as_index=False)
           .agg(sog_for=("I_F_shotsOnGoal", "sum"), att_for=("I_F_shotAttempts", "sum")))
    opp = t[["gameId", "playerTeam", "sog_for", "att_for"]].rename(
        columns={"playerTeam": "opposingTeam", "sog_for": "sog_against", "att_for": "att_against"})
    t = t.merge(opp, on=["gameId", "opposingTeam"], how="left").sort_values(["playerTeam", "gameDate"])
    gb = t.groupby("playerTeam")
    for c in ["sog_for", "sog_against", "att_against"]:
        t[f"{c}_ewm"] = gb[c].transform(lambda s: s.shift().ewm(span=20, min_periods=3).mean())
    t["team_rest"] = gb["gameDate"].diff().dt.days
    return t


def build(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values(["playerId", "gameDate"]).copy()
    g["toi"] = g.icetime / 60
    g["pp_toi"] = g.pp_toi.fillna(0) / 60
    g["sog"] = g.I_F_shotsOnGoal
    g["att"] = g.I_F_shotAttempts
    g["is_d"] = (g.position == "D").astype(int)
    g["home"] = (g.home_or_away == "HOME").astype(int)

    p = g.groupby("playerId")
    for c in ["toi", "pp_toi", "sog", "att"]:
        lag = p[c].shift()
        for s in SPANS:
            g[f"{c}_ewm{s}"] = lag.groupby(g.playerId).transform(lambda x, s=s: x.ewm(span=s, min_periods=1).mean())
    g["toi_last"] = p["toi"].shift()
    g["n_prior"] = p.cumcount()
    g["n_prior_season"] = g.groupby(["playerId", "season"]).cumcount()
    g["rest"] = p["gameDate"].diff().dt.days

    # Per-60 rates, shrunk toward position averages (weight = prior TOI in minutes)
    for c in ["sog", "att"]:
        for s in SPANS:
            g[f"{c}60_{s}"] = 60 * g[f"{c}_ewm{s}"] / g[f"toi_ewm{s}"].clip(lower=1)
    lg = g.groupby("is_d")[["sog", "att", "toi"]].sum()
    prior60 = (60 * lg["sog"] / lg["toi"]).to_dict()
    cum_sog = p["sog"].transform(lambda x: x.shift().rolling(82, min_periods=1).sum()).fillna(0)
    cum_toi = p["toi"].transform(lambda x: x.shift().rolling(82, min_periods=1).sum()).fillna(0)
    k = 300.0  # minutes of prior weight (about 15-20 games)
    g["sog60_shrunk"] = (cum_sog * 60 + g.is_d.map(prior60) * k) / (cum_toi + k)
    g["sog_per_att"] = (p["sog"].transform(lambda x: x.shift().rolling(82, min_periods=1).sum()) /
                        p["att"].transform(lambda x: x.shift().rolling(82, min_periods=1).sum()).clip(lower=1))

    # Team context: opponent shots allowed, own team shot volume
    t = team_games(g)
    lg_sog = t.sog_for.mean()
    opp = t[["gameId", "playerTeam", "sog_against_ewm", "att_against_ewm", "team_rest"]].rename(
        columns={"playerTeam": "opposingTeam", "sog_against_ewm": "opp_sog_allowed",
                 "att_against_ewm": "opp_att_allowed", "team_rest": "opp_rest"})
    own = t[["gameId", "playerTeam", "sog_for_ewm"]].rename(columns={"sog_for_ewm": "team_sog_for"})
    g = g.merge(opp, on=["gameId", "opposingTeam"], how="left").merge(own, on=["gameId", "playerTeam"], how="left")
    g["opp_factor"] = (g.opp_sog_allowed / lg_sog).fillna(1.0)
    g["team_factor"] = (g.team_sog_for / lg_sog).fillna(1.0)
    g["b2b"] = (g.rest == 1).astype(int)
    g["opp_b2b"] = (g.opp_rest == 1).astype(int)

    # Structural baseline: projected TOI x shrunk shot rate x opponent factor
    g["toi_proj"] = 0.5 * g.toi_ewm5 + 0.5 * g.toi_ewm15
    g["mu_struct"] = g.toi_proj / 60 * g.sog60_shrunk * g.opp_factor
    return g


FEATURES = (["is_d", "home", "b2b", "opp_b2b", "rest", "n_prior", "n_prior_season", "toi_last",
             "sog60_shrunk", "sog_per_att", "opp_factor", "team_factor", "opp_att_allowed", "mu_struct", "toi_proj"]
            + [f"{c}_ewm{s}" for c in ["toi", "pp_toi", "sog", "att"] for s in SPANS]
            + [f"{c}60_{s}" for c in ["sog", "att"] for s in SPANS])
