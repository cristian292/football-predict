"""Behavioral contracts for leakage-safe richer pre-match predictors."""
import importlib

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from football_predictor.features import BASIC_FEATURES, COMPLEX_FEATURES, build_features


def api():
    return importlib.import_module("football_predictor.advanced_features")


def matches():
    return pd.DataFrame([
        ("2023-08-01", 2024, "A", "B", 2., 0.),
        ("2023-08-08", 2024, "B", "C", 1., 1.),
        ("2023-08-15", 2024, "A", "C", 0., 1.),
        ("2023-08-22", 2024, "B", "A", 3., 1.),
    ], columns=["date", "season", "home_team", "away_team", "home_goals", "away_goals"]).assign(date=lambda x: pd.to_datetime(x.date))


def test_multiwindow_venue_ewma_and_observation_counts():
    actual = api().build_advanced_features(matches())
    row = actual.iloc[3]
    for window in (5, 10, 20):
        assert row[f"home_form_{window}_gf"] == 0.5  # B: 0, 1
        assert row[f"away_form_{window}_ppg"] == 1.5  # A: 3, 0
        assert row[f"away_form_{window}_count"] == 2
        assert row[f"form_{window}_ppg_delta"] == -1
        assert row[f"form_{window}_gd_delta"] == -1.5
    assert row.home_venue_gf == 1
    assert row.home_venue_count == 1
    assert row.away_venue_count == 0
    assert row.away_venue_missing == 1
    assert row.away_venue_gf == 1.3
    assert row.away_ewma_gf == pytest.approx(1.6)
    assert row.away_history_count == 2
    assert actual.iloc[0].home_history_missing == 1
    assert actual.iloc[0].home_form_20_gf == 1.3
    assert row.rest_days_delta == 7


def test_elo_fixed_priors_home_advantage_and_opponent_adjustment():
    actual = api().build_advanced_features(matches())
    expected = 1 / (1 + 10 ** (-60 / 400))
    gain = 20 * (1 - expected)
    first = actual.iloc[0]
    assert first.home_elo == first.away_elo == 1500
    assert first.home_elo_expected == pytest.approx(expected)
    assert first.elo_delta == 0
    assert actual.iloc[1].home_elo == pytest.approx(1500 - gain)
    assert actual.iloc[1].away_elo == 1500
    assert actual.iloc[2].home_elo == pytest.approx(1500 + gain)
    assert actual.iloc[2].home_adjusted_form == pytest.approx(1 - expected)
    assert actual.iloc[0].home_adjusted_form == 0


def test_season_shrink_happens_before_first_batch_and_form_carries():
    source = matches().iloc[:1].copy()
    following = source.iloc[0].copy()
    following.date = pd.Timestamp("2024-08-01")
    following.season = 2025
    following.away_team = "C"
    source = pd.concat([source, following.to_frame().T], ignore_index=True)
    source["date"] = pd.to_datetime(source.date)
    actual = api().build_advanced_features(source)
    gain = 20 * (1 - actual.iloc[0].home_elo_expected)
    assert actual.iloc[1].home_elo == pytest.approx(1500 + 0.75 * gain)
    assert actual.iloc[1].away_elo == 1500
    assert actual.iloc[1].home_adjusted_form == 0
    assert actual.iloc[1].home_form_20_count == 1
    assert actual.iloc[1].home_form_20_gf == 2
    assert actual.iloc[1].points_delta == 0


def test_future_fixture_matches_masked_append_without_mutating_history():
    source = matches().assign(home_sh=np.nan, away_sh=np.nan)
    before = source.copy(deep=True)
    date = pd.Timestamp("2024-08-01")
    query = pd.DataFrame([dict(date=date, season=2025, home_team="A", away_team="B",
                               home_goals=np.nan, away_goals=np.nan)])
    expected = api().build_advanced_features(pd.concat([source, query], ignore_index=True))
    columns = api().FEATURE_SETS["extended"]
    actual = api().fixture_advanced_features(source, "A", "B", date)
    assert list(actual.columns) == columns
    assert_frame_equal(actual, expected.iloc[[-1]][columns].reset_index(drop=True))
    assert_frame_equal(source, before)


@pytest.mark.parametrize("home,away,date", [
    ("A", "A", "2023-09-01"), ("A", "Unknown", "2023-09-01"),
    ("A", "B", "2023-08-22"), ("A", "B", "2023-08-01"),
    ("A", "B", pd.NaT), ("A", "B", "not-a-date"),
    ("A", "B", "2023-08-22 12:00"),
])
def test_fixture_rejects_invalid_teams_or_nonfuture_calendar_dates(home, away, date):
    with pytest.raises(ValueError):
        api().fixture_advanced_features(matches(), home, away, date)


def test_current_future_goals_and_raw_statistics_cannot_change_prior_features():
    source = matches().assign(home_sh=12., away_sh=8., home_sot=4., away_sot=2.,
                              home_poss=55., away_poss=45.)
    before = source.copy(deep=True)
    columns = api().FEATURE_SETS["extended"]
    baseline = api().build_advanced_features(source)
    assert_frame_equal(source, before)
    for current in range(len(source)):
        changed = source.copy(deep=True)
        changed.loc[current:, ["home_goals", "away_goals", "home_sh", "away_sh",
                               "home_sot", "away_sot", "home_poss", "away_poss"]] = [9, 8, 90, 80, 40, 30, 90, 10]
        actual = api().build_advanced_features(changed)
        assert_frame_equal(actual.loc[:current, columns], baseline.loc[:current, columns])


def test_same_day_batch_is_invariant_to_labels_and_input_order():
    source = matches()
    source.loc[1, "date"] = source.loc[0, "date"]
    columns = api().FEATURE_SETS["extended"]
    baseline = api().build_advanced_features(source)
    changed = source.copy(deep=True)
    changed.loc[0, ["home_goals", "away_goals"]] = [0, 9]
    actual = api().build_advanced_features(changed.iloc[::-1])
    assert_frame_equal(actual.iloc[:2][columns], baseline.iloc[:2][columns])
    assert baseline.iloc[1].home_history_count == 0
    assert_frame_equal(api().build_advanced_features(source.iloc[::-1]), baseline)
    # The changed result should affect later forecasts, not silently be ignored.
    assert actual.iloc[2].home_elo != baseline.iloc[2].home_elo


def test_optional_statistics_expose_observations_without_fabricating_xg():
    source = matches().assign(home_sh=10., away_sh=8., home_sot=np.nan, away_sot=2.,
                              home_poss=np.nan, away_poss=np.nan, home_xg=99., away_xg=99.)
    actual = api().build_advanced_features(source)
    row = actual.iloc[2]  # A's single previous home game has no SOT/possession.
    assert row.home_recent_sh_count == 1
    assert row.home_recent_sot_count == 0
    assert row.home_recent_poss_count == 0
    assert row.home_recent_poss_missing == 1
    assert row.away_recent_sot_count == 1
    assert row.away_recent_sot_missing == 0
    assert not any("xg" in name for name in api().FEATURE_SETS["extended"])
    assert not any("xg" in name for name in api().FEATURE_SETS["form"])


def test_windows_have_distinct_horizons_and_counts_are_capped():
    source = pd.DataFrame([
        dict(date=date, season=2024, home_team="A", away_team="B", home_goals=float(i), away_goals=0.)
        for i, date in enumerate(pd.date_range("2023-08-01", periods=26, freq="7D"))
    ])
    row = api().build_advanced_features(source).iloc[-1]
    for window in (5, 10, 20):
        assert row[f"home_form_{window}_gf"] == pytest.approx(sum(range(25-window, 25))/window)
        assert row[f"home_form_{window}_count"] == window
    assert row.home_venue_gf == pytest.approx(sum(range(15, 25))/10)
    assert row.home_venue_count == 10
    assert row.home_history_count == 25


def test_missing_goal_pair_is_not_a_form_or_strength_observation():
    source = matches()
    source.loc[0, "home_goals"] = np.nan
    actual = api().build_advanced_features(source)
    assert actual.iloc[1].home_elo == 1500
    assert actual.iloc[1].home_history_count == 0
    assert actual.iloc[2].home_form_20_count == 0
    assert actual.iloc[2].home_ewma_gf == 1.3


def test_empty_input_preserves_schema_and_legacy_columns():
    source = matches().iloc[:0]
    actual = api().build_advanced_features(source)
    assert actual.empty
    legacy = build_features(source)
    assert_frame_equal(actual[legacy.columns], legacy)
    assert set(api().FEATURE_SETS["extended"]) <= set(actual.columns)


def test_schema_preserves_canonical_legacy_rows():
    source = matches().iloc[::-1]
    actual = api().build_advanced_features(source)
    legacy = build_features(source)
    assert_frame_equal(actual[legacy.columns], legacy)
    sets = api().FEATURE_SETS
    assert set(sets) == {"legacy_basic", "legacy_complex", "form", "strength", "extended"}
    assert sets["legacy_basic"] == BASIC_FEATURES
    assert sets["legacy_complex"] == COMPLEX_FEATURES
    for columns in sets.values():
        assert isinstance(columns, list)
        assert len(columns) == len(set(columns))
        assert set(columns) <= set(actual.columns)
        assert not set(columns) & {"date", "season", "home_team", "away_team", "home_goals", "away_goals", "result"}
        assert np.isfinite(actual[columns].to_numpy()).all()
