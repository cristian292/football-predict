"""Load team-perspective CSVs as one row per completed fixture."""
from pathlib import Path

import pandas as pd

CANONICAL_COLUMNS = ["date", "season", "home_team", "away_team", "home_goals", "away_goals",
                     "home_sh", "away_sh", "home_sot", "away_sot", "home_poss", "away_poss"]


def load_matches(path: str | Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    required = {"date", "season", "venue", "team", "opponent", "gf", "ga"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    raw["date"] = pd.to_datetime(raw["date"], errors="raise").dt.normalize()
    if raw[["date", "season", "team", "opponent", "venue"]].isna().any().any():
        raise ValueError("Missing fixture identity")
    if not raw.venue.isin(["Home", "Away"]).all():
        raise ValueError("venue must be Home or Away")
    for column in ("gf", "ga"):
        raw[column] = pd.to_numeric(raw[column], errors="raise")
        present = raw[column].dropna()
        if ((present < 0) | (present % 1 != 0)).any():
            raise ValueError("Goals must be nonnegative integers")
    for column in ("sh", "sot", "poss"):
        if column not in raw:
            raw[column] = float("nan")
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    aliases = {"Wolverhampton Wanderers": "Wolves", "Manchester United": "Manchester Utd",
               "Newcastle United": "Newcastle", "Nottingham Forest": "Nottingham",
               "Brighton and Hove Albion": "Brighton", "Brighton & Hove Albion": "Brighton",
               "Tottenham Hotspur": "Tottenham", "West Ham United": "West Ham",
               "West Bromwich Albion": "West Brom"}
    for column in ("team", "opponent"):
        raw[column] = raw[column].str.strip().replace(aliases)
        if raw[column].eq("").any():
            raise ValueError("Team names must not be blank")
    if raw.team.eq(raw.opponent).any():
        raise ValueError("A team cannot play itself")
    unplayed = raw[["gf", "ga"]].isna().any(axis=1)
    unplayed_count = int(unplayed.sum())
    raw = raw.loc[~unplayed].copy()
    identity = ["date", "season", "venue", "team", "opponent"]
    if (raw.groupby(identity)[["gf", "ga", "sh", "sot", "poss"]].nunique(dropna=True) > 1).any().any():
        raise ValueError("Conflicting duplicate fixture rows")
    duplicate_count = int(raw.duplicated(identity).sum())
    raw = raw.groupby(identity, as_index=False, sort=False).first()
    home = raw.loc[raw.venue.eq("Home")].copy()
    away = raw.loc[raw.venue.eq("Away")].copy()
    home = home.rename(columns={"team": "home_team", "opponent": "away_team", "gf": "home_goals",
                                "ga": "away_goals", "sh": "home_sh", "sot": "home_sot", "poss": "home_poss"})
    away = away.rename(columns={"opponent": "home_team", "team": "away_team", "sh": "away_sh",
                                "sot": "away_sot", "poss": "away_poss"})
    keys = ["date", "season", "home_team", "away_team"]
    reverse = away[keys + ["gf", "ga"]]
    paired = home.merge(reverse, on=keys, how="inner")
    if (paired.home_goals.ne(paired.ga) | paired.away_goals.ne(paired.gf)).any():
        raise ValueError("Inconsistent reverse fixture scores")
    result = home.merge(away[keys + ["away_sh", "away_sot", "away_poss"]], on=keys, how="left")[CANONICAL_COLUMNS].sort_values(keys).reset_index(drop=True)
    result.attrs.update(duplicate_rows=duplicate_count, unplayed_rows=unplayed_count)
    return result


def standings(matches: pd.DataFrame, season=None, before=None) -> pd.DataFrame:
    """Rank the selected season (latest observed by default), strictly before cutoff.

    Ties use points, goal difference, goals for, then alphabetical team name.
    Teams with no observed fixtures are deliberately absent: no future roster leaks.
    """
    columns = ["team", "played", "points", "gf", "ga", "gd", "position"]
    selected = matches
    if before is not None:
        selected = selected.loc[selected.date < pd.Timestamp(before)]
    if selected.empty:
        return pd.DataFrame(columns=columns)
    if season is None:
        season = selected.sort_values("date").iloc[-1].season
    selected = selected.loc[selected.season.eq(season)]
    totals = {}
    for match in selected.itertuples(index=False):
        for team, gf, ga in ((match.home_team, match.home_goals, match.away_goals),
                             (match.away_team, match.away_goals, match.home_goals)):
            value = totals.setdefault(team, [0, 0, 0, 0])
            value[0] += 1
            value[1] += 3 if gf > ga else int(gf == ga)
            value[2] += gf
            value[3] += ga
    table = pd.DataFrame([[team, *values] for team, values in totals.items()], columns=columns[:5])
    table["gd"] = table.gf - table.ga
    table = table.sort_values(["points", "gd", "gf", "team"], ascending=[False, False, False, True]).reset_index(drop=True)
    table["position"] = range(1, len(table) + 1)
    return table[columns]
