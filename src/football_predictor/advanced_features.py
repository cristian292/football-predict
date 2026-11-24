"""Causal richer predictors; legacy columns are delegated, never reimplemented.

Form uses completed matches across seasons, with fixed cold-start priors. All
rows on a calendar date are read before any observations from that date update
state. Missing goal pairs are not observations. No raw match statistic is a
predictor in FEATURE_SETS.

``form`` contains 5/10/20-match averages, last-ten venue averages, exponentially
weighted form (alpha .2, initialized from the first observation), their history
indicators and rest. ``strength`` contains Elo and last-ten mean result-minus-
Elo-expectation residuals. Elo starts at 1500, uses home advantage 60 and K=20;
new seasons retain 75% of each rating's deviation and reset residual histories.
``extended`` additionally retains every legacy predictor and last-three observed
counts/missing indicators for shots, shots on target and possession. No xG is
inferred. Legacy priors and season tables are deliberately unchanged.

Inputs follow the canonical fixture contract of ``features.build_features``:
pandas date values represent fixture calendar dates, with numeric season labels.
"""
from collections import defaultdict, deque

import pandas as pd

from .features import BASIC_FEATURES, COMPLEX_FEATURES, build_features

WINDOWS = (5, 10, 20)
STATS = ("gf", "ga", "ppg", "gd")
PRIORS = dict(gf=1.3, ga=1.3, ppg=1.0, gd=0.0)
EWMA_ALPHA = 0.2
ELO_INITIAL = 1500.0
ELO_HOME_ADVANTAGE = 60.0
ELO_K = 20.0
ELO_SEASON_RETAIN = 0.75
FORM_FEATURES = [
    f"{side}_form_{window}_{stat}"
    for side in ("home", "away") for window in WINDOWS
    for stat in (*STATS, "count", "missing")
] + [
    f"form_{window}_{stat}_delta" for window in WINDOWS for stat in STATS
] + [
    f"{side}_{kind}_{stat}" for side in ("home", "away")
    for kind in ("venue", "ewma") for stat in STATS
] + [
    f"{side}_{stat}" for side in ("home", "away")
    for stat in ("venue_count", "venue_missing", "history_count", "history_missing", "rest_days")
] + ["rest_days_delta"]
STRENGTH_FEATURES = ["home_elo", "away_elo", "elo_delta", "home_elo_expected",
                     "home_adjusted_form", "away_adjusted_form", "adjusted_form_delta"]
OBSERVATION_FEATURES = [f"{side}_recent_{stat}_{indicator}"
                        for side in ("home", "away") for stat in ("sh", "sot", "poss")
                        for indicator in ("count", "missing")]
FEATURE_SETS = {
    "legacy_basic": list(BASIC_FEATURES),
    "legacy_complex": list(COMPLEX_FEATURES),
    "form": FORM_FEATURES,
    "strength": STRENGTH_FEATURES,
    "extended": list(dict.fromkeys(COMPLEX_FEATURES + FORM_FEATURES + STRENGTH_FEATURES + OBSERVATION_FEATURES)),
}


def _summary(history):
    return {stat: sum(item[stat] for item in history) / len(history)
            if history else PRIORS[stat] for stat in STATS}


def build_advanced_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Return the legacy canonical rows plus numeric, strictly pre-date features."""
    ordered = build_features(matches)
    history = defaultdict(lambda: deque(maxlen=20))
    venue = defaultdict(lambda: deque(maxlen=10))
    ewma, counts = {}, defaultdict(int)
    elo = defaultdict(lambda: ELO_INITIAL)
    adjusted = defaultdict(lambda: deque(maxlen=10))
    observations = defaultdict(lambda: deque(maxlen=3))
    rows = []
    seen_seasons = set()
    for _, batch in ordered.groupby(ordered.date.dt.normalize(), sort=False):
        # Season transitions are determined without inspecting labels. Shrink all
        # previously seen teams once, before any fixture in the new season date.
        for season in batch.season.unique():
            if season not in seen_seasons:
                if seen_seasons:
                    for team in elo:
                        elo[team] = ELO_INITIAL + ELO_SEASON_RETAIN * (elo[team] - ELO_INITIAL)
                    adjusted.clear()
                seen_seasons.add(season)
        for match in batch.itertuples(index=False):
            row = {}
            he, ae = elo[match.home_team], elo[match.away_team]
            expected = 1 / (1 + 10 ** ((ae - he - ELO_HOME_ADVANTAGE) / 400))
            row.update(home_elo=he, away_elo=ae, elo_delta=he-ae, home_elo_expected=expected)
            for side, team in (("home", match.home_team), ("away", match.away_team)):
                for window in WINDOWS:
                    past = list(history[team])[-window:]
                    row.update({f"{side}_form_{window}_{stat}": value
                                for stat, value in _summary(past).items()})
                    row[f"{side}_form_{window}_count"] = len(past)
                    row[f"{side}_form_{window}_missing"] = int(not past)
                past = venue[(team, side)]
                row.update({f"{side}_venue_{stat}": value for stat, value in _summary(past).items()})
                row.update({f"{side}_ewma_{stat}": value for stat, value in ewma.get(team, PRIORS).items()})
                row[f"{side}_venue_count"] = len(past)
                row[f"{side}_venue_missing"] = int(not past)
                row[f"{side}_history_count"] = counts[team]
                row[f"{side}_history_missing"] = int(not counts[team])
                for stat in ("sh", "sot", "poss"):
                    count = sum(pd.notna(item[stat]) for item in observations[team])
                    row[f"{side}_recent_{stat}_count"] = count
                    row[f"{side}_recent_{stat}_missing"] = int(count == 0)
                residuals = adjusted[team]
                row[f"{side}_adjusted_form"] = sum(residuals) / len(residuals) if residuals else 0.0
            row["adjusted_form_delta"] = row["home_adjusted_form"] - row["away_adjusted_form"]
            for window in WINDOWS:
                for stat in STATS:
                    row[f"form_{window}_{stat}_delta"] = row[f"home_form_{window}_{stat}"] - row[f"away_form_{window}_{stat}"]
            row["rest_days_delta"] = match.home_rest_days - match.away_rest_days
            rows.append(row)
        for match, snapshot in zip(batch.itertuples(index=False), rows[-len(batch):]):
            if pd.isna(match.home_goals) or pd.isna(match.away_goals):
                continue
            expected = snapshot["home_elo_expected"]
            score = 1.0 if match.home_goals > match.away_goals else 0.5 if match.home_goals == match.away_goals else 0.0
            residual = score - expected
            elo[match.home_team] += ELO_K * residual
            elo[match.away_team] -= ELO_K * residual
            adjusted[match.home_team].append(residual)
            adjusted[match.away_team].append(-residual)
            for side, team, gf, ga in (("home", match.home_team, match.home_goals, match.away_goals),
                                       ("away", match.away_team, match.away_goals, match.home_goals)):
                values = dict(gf=gf, ga=ga, ppg=3.0 if gf > ga else float(gf == ga), gd=gf-ga)
                history[team].append(values)
                observations[team].append({stat: getattr(match, f"{side}_{stat}", float("nan"))
                                           for stat in ("sh", "sot", "poss")})
                venue[(team, side)].append(values)
                previous = ewma.get(team)
                ewma[team] = {stat: EWMA_ALPHA * values[stat] + (1-EWMA_ALPHA) * previous[stat]
                              if previous else values[stat] for stat in STATS}
                counts[team] += 1
    additions = pd.DataFrame(rows, index=ordered.index, columns=[c for c in FORM_FEATURES + STRENGTH_FEATURES + OBSERVATION_FEATURES if c not in ordered])
    return pd.concat([ordered, additions], axis=1)


def fixture_advanced_features(matches: pd.DataFrame, home: str, away: str, date) -> pd.DataFrame:
    """One extended predictor row for a future English-league fixture.

    Numeric season labels use the legacy ending-year/August convention. Query
    goals remain missing, not a fabricated draw. Input history is never changed.
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
    if matches.empty or cutoff.normalize() <= matches.date.max().normalize():
        raise ValueError("Fixture date must be strictly after the history's final calendar date")
    season = cutoff.year + int(cutoff.month >= 8)
    query = pd.DataFrame([dict(date=cutoff, season=season, home_team=home, away_team=away,
                               home_goals=float("nan"), away_goals=float("nan"))])
    combined = pd.concat([matches, query], ignore_index=True)
    return build_advanced_features(combined).iloc[[-1]][FEATURE_SETS["extended"]].reset_index(drop=True)
