"""Pre-match rolling predictors; current fixture values are never predictors."""
from collections import defaultdict, deque
from itertools import groupby
from operator import attrgetter

import pandas as pd

ROLLING_WINDOW = 3
BASIC_FEATURES = [
    f"{side}_recent_{stat}"
    for side in ("home", "away") for stat in ("gf", "ga", "ppg")
]
COMPLEX_FEATURES = BASIC_FEATURES + [
    f"{side}_recent_{stat}"
    for side in ("home", "away") for stat in ("sh", "sot", "poss")
] + [
    "home_position", "away_position", "points_delta", "position_delta",
    "h2h_home_ppg", "h2h_goal_diff", "home_rest_days", "away_rest_days",
]
_PRIORS = {"gf": 1.3, "ga": 1.3, "ppg": 1.0, "sh": 10.0, "sot": 3.0, "poss": 50.0}


def build_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Return canonical fixtures sorted by date/home/away, with pre-match features.

    Rolling form and H2H use the last three fixtures, across season boundaries.
    Tables reset by season. Every calendar-date batch is read before any updates.
    Missing observations use fixed priors, never whole-dataset averages.
    Deltas are home minus away; a negative position delta favors the home team.
    """
    ordered = matches.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True).copy()
    history = defaultdict(lambda: deque(maxlen=ROLLING_WINDOW))
    rows = []
    tables = defaultdict(dict)
    last_seen = {}
    h2h = defaultdict(lambda: deque(maxlen=ROLLING_WINDOW))
    for _, batch_iter in groupby(ordered.itertuples(index=False), key=attrgetter("date")):
        batch = list(batch_iter)
        for match in batch:
            table = tables[match.season]
            ranked = sorted(table, key=lambda team: (-table[team][0], -(table[team][1] - table[team][2]), -table[team][1], team))
            positions = {team: i + 1 for i, team in enumerate(ranked)}
            hp = positions.get(match.home_team, len(table) + 1)
            ap = positions.get(match.away_team, len(table) + 1)
            features: dict[str, float] = dict(home_position=hp, away_position=ap, position_delta=hp-ap,
                            points_delta=table.get(match.home_team, [0])[0] - table.get(match.away_team, [0])[0])
            for side, team in (("home", match.home_team), ("away", match.away_team)):
                features[f"{side}_rest_days"] = (match.date - last_seen[team]).days if team in last_seen else 7
                for stat in _PRIORS:
                    observed = [item[stat] for item in history[team] if pd.notna(item[stat])]
                    features[f"{side}_recent_{stat}"] = sum(observed) / len(observed) if observed else _PRIORS[stat]
            meetings = h2h[(match.home_team, match.away_team)]
            features["h2h_home_ppg"] = sum(item[0] for item in meetings) / len(meetings) if meetings else 1.0
            features["h2h_goal_diff"] = sum(item[1] for item in meetings) / len(meetings) if meetings else 0.0
            rows.append(features)
        for match in batch:
            for side, team, gf, ga in (("home", match.home_team, match.home_goals, match.away_goals),
                                       ("away", match.away_team, match.away_goals, match.home_goals)):
                values: dict[str, float] = dict(gf=gf, ga=ga, ppg=3 if gf > ga else int(gf == ga))
                values.update({stat: getattr(match, f"{side}_{stat}", float("nan")) for stat in ("sh", "sot", "poss")})
                history[team].append(values)
                last_seen[team] = match.date
                total = tables[match.season].setdefault(team, [0, 0, 0])
                total[0] += values["ppg"]
                total[1] += gf
                total[2] += ga
                opponent = match.away_team if side == "home" else match.home_team
                h2h[(team, opponent)].append((values["ppg"], gf-ga))
    for column in COMPLEX_FEATURES:
        ordered[column] = [row[column] for row in rows]
    ordered["result"] = pd.Series([0 if h > a else 1 if h == a else 2 for h, a in zip(ordered.home_goals, ordered.away_goals)], dtype="int64")
    return ordered


def fixture_features(matches: pd.DataFrame, home: str, away: str, date) -> pd.DataFrame:
    """One pre-match predictor row, using only observations strictly before date.

    Numeric seasons follow the supplied CSV's ending-year convention with an
    August boundary (including its extended July 2020 season). No current/future
    match statistics are consulted; this API targets the supplied English league.
    """
    if home == away:
        raise ValueError("A team cannot play itself")
    known = set(matches.home_team) | set(matches.away_team)
    unknown = {home, away} - known
    if unknown:
        raise ValueError(f"Unknown team(s): {', '.join(sorted(unknown))}")
    cutoff = pd.Timestamp(date)
    if pd.isna(cutoff):
        raise ValueError("A valid fixture date is required")
    history = matches.loc[matches.date < cutoff].copy()
    season = cutoff.year + int(cutoff.month >= 8)
    query = pd.DataFrame([dict(date=cutoff, season=season, home_team=home, away_team=away,
                               home_goals=0, away_goals=0)])
    combined = pd.concat([history, query], ignore_index=True)
    return build_features(combined).iloc[[-1]][COMPLEX_FEATURES].reset_index(drop=True)
