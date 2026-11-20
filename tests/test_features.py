"""End-to-end tests for strictly historical fixture predictors."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def fixture(date="2025-01-01", home="A", away="B", hg=2, ag=0, season=2025, **stats):
    return dict(date=pd.Timestamp(date), season=season, home_team=home, away_team=away,
                home_goals=hg, away_goals=ag, home_sh=10, away_sh=6,
                home_sot=4, away_sot=2, home_poss=60, away_poss=40) | stats


def test_basic_features_are_previous_three_games_only_with_three_way_labels():
    from football_predictor.features import build_features, BASIC_FEATURES
    matches = pd.DataFrame([fixture(), fixture("2025-01-02", hg=1, ag=1),
        fixture("2025-01-03", hg=0, ag=3), fixture("2025-01-04", hg=4, ag=0),
        fixture("2025-01-05", hg=99, ag=99)])
    original = matches.copy(deep=True)
    built = build_features(matches)
    assert built.result.tolist() == [0, 1, 2, 0, 1]
    assert np.issubdtype(built.result.dtype, np.integer)
    assert built.iloc[0].home_recent_gf == 1.3
    assert built.iloc[-1].home_recent_gf == pytest.approx(5 / 3)
    assert built.iloc[-1].home_recent_ga == pytest.approx(4 / 3)
    assert built.iloc[-1].home_recent_ppg == pytest.approx(4 / 3)
    assert built.iloc[-1].away_recent_gf == pytest.approx(4 / 3)
    assert built.iloc[-1].away_recent_ga == pytest.approx(5 / 3)
    assert len(BASIC_FEATURES) == 6
    assert set(BASIC_FEATURES).issubset(built.columns)
    pd.testing.assert_frame_equal(matches, original)


def test_same_date_is_a_batch_and_future_changes_do_not_affect_earlier_predictors():
    from football_predictor.features import build_features, BASIC_FEATURES
    matches = pd.DataFrame([fixture(), fixture(home="A", away="C", hg=88),
        fixture("2025-01-02", home="A", away="C")])
    built = build_features(matches)
    assert built.iloc[1].home_recent_gf == 1.3
    altered = matches.copy()
    altered.loc[altered.date >= pd.Timestamp("2025-01-02"), ["home_goals", "away_goals", "home_sh"]] = 999
    pd.testing.assert_frame_equal(built[BASIC_FEATURES], build_features(altered)[BASIC_FEATURES])


def test_complex_rolling_shots_use_only_observed_history_and_finite_priors():
    from football_predictor.features import build_features, BASIC_FEATURES, COMPLEX_FEATURES
    built = build_features(pd.DataFrame([fixture(home_sh=np.nan, home_sot=np.nan, home_poss=np.nan),
        fixture("2025-01-02", home_sh=20, home_sot=8, home_poss=70), fixture("2025-01-03")]))
    assert set(BASIC_FEATURES) < set(COMPLEX_FEATURES)
    assert built.iloc[0].home_recent_sh == 10
    assert built.iloc[1].home_recent_poss == 50
    assert built.iloc[2].home_recent_sh == 20
    assert built.iloc[2].home_recent_sot == 8
    assert built.iloc[2].home_recent_poss == 70
    assert built.iloc[2].away_recent_sh == 6
    assert np.isfinite(built[COMPLEX_FEATURES].to_numpy()).all()
    assert not set(COMPLEX_FEATURES) & {"home_goals", "away_goals", "home_sh", "away_sh", "result"}


def test_season_table_predictors_reset_while_rolling_history_is_retained():
    from football_predictor.features import build_features
    built = build_features(pd.DataFrame([fixture("2024-05-01", season=2024),
        fixture("2025-01-01"), fixture("2025-01-02", home="B", away="A"),
        fixture("2025-08-01", season=2026)]))
    assert built.iloc[0].points_delta == 0
    assert built.iloc[1].points_delta == 0
    assert built.iloc[1].home_recent_gf == 2
    assert built.iloc[2].points_delta == -3
    assert built.iloc[2].position_delta == 1
    assert built.iloc[2].home_position == 2
    assert built.iloc[2].away_position == 1
    assert built.iloc[3].points_delta == 0
    assert built.iloc[3].position_delta == 0


def test_h2h_is_from_current_home_perspective_across_reversed_venues():
    from football_predictor.features import build_features
    built = build_features(pd.DataFrame([fixture(), fixture("2025-01-02", home="B", away="A", hg=1, ag=1),
        fixture("2025-01-03", home="B", away="A")]))
    assert built.iloc[0].h2h_home_ppg == 1
    assert built.iloc[1].h2h_home_ppg == 0
    assert built.iloc[1].h2h_goal_diff == -2
    assert built.iloc[2].h2h_home_ppg == 0.5
    assert built.iloc[2].h2h_goal_diff == -1


def test_rest_days_are_historical_with_neutral_debut_prior():
    from football_predictor.features import build_features
    built = build_features(pd.DataFrame([fixture(), fixture("2025-01-04", home="A", away="C"),
        fixture("2025-01-10", home="B", away="C")]))
    assert built.iloc[0].home_rest_days == 7
    assert built.iloc[1].home_rest_days == 3
    assert built.iloc[1].away_rest_days == 7
    assert built.iloc[2].home_rest_days == 9
    assert built.iloc[2].away_rest_days == 6


def test_fixture_features_matches_training_and_ignores_current_and_future_rows():
    from football_predictor.features import build_features, fixture_features, COMPLEX_FEATURES
    matches = pd.DataFrame([fixture(), fixture("2025-01-04", home="A", away="C"),
        fixture("2025-01-10", home="B", away="C"), fixture("2025-02-01", hg=100)])
    actual = fixture_features(matches, "B", "C", "2025-01-10")
    expected = build_features(matches).iloc[[2]][COMPLEX_FEATURES].reset_index(drop=True)
    pd.testing.assert_frame_equal(actual[COMPLEX_FEATURES], expected, check_dtype=False)
    changed = matches.copy()
    changed.loc[changed.date >= pd.Timestamp("2025-01-10"), ["home_goals", "away_goals", "home_sh"]] = 555
    pd.testing.assert_frame_equal(actual, fixture_features(changed, "B", "C", "2025-01-10"))
    assert actual.columns.tolist() == COMPLEX_FEATURES


@pytest.mark.parametrize("home,away,date", [("A", "A", "2025-01-02"), ("Z", "B", "2025-01-02"),
    ("A", "Z", "2025-01-02"), ("A", "B", None), ("A", "B", "NaT")])
def test_fixture_rejects_same_or_unknown_teams_and_missing_dates(home, away, date):
    from football_predictor.features import fixture_features
    with pytest.raises(ValueError):
        fixture_features(pd.DataFrame([fixture()]), home, away, date)


def test_july_2020_extended_season_fixture_uses_historical_season_table():
    from football_predictor.features import build_features, fixture_features, COMPLEX_FEATURES
    matches = pd.DataFrame([fixture("2020-06-30", season=2020), fixture("2020-07-10", season=2020)])
    expected = build_features(matches).iloc[[1]][COMPLEX_FEATURES].reset_index(drop=True)
    actual = fixture_features(matches, "A", "B", "2020-07-10")
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False)


def test_empty_history_has_stable_schema_and_integer_target():
    from football_predictor.features import build_features, COMPLEX_FEATURES
    empty = pd.DataFrame([fixture()]).iloc[:0]
    result = build_features(empty)
    assert result.empty
    assert result.columns.tolist() == [*empty.columns, *COMPLEX_FEATURES, "result"]
    assert np.issubdtype(result.result.dtype, np.integer)
