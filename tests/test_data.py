"""Canonical fixture loading and historical league tables."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def write_rows(tmp_path, rows):
    path = tmp_path / "matches.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def row(**changes):
    values = dict(date="2025-01-01", season=2025, venue="Home", team="Arsenal",
                  opponent="Wolves", gf=2, ga=1, sh=12, sot=5, poss=60)
    return values | changes


def test_load_pairs_home_and_away_perspectives(tmp_path):
    from football_predictor.data import load_matches
    matches = load_matches(write_rows(tmp_path, [row(), row(venue="Away", team="Wolves",
        opponent="Arsenal", gf=1, ga=2, sh=7, sot=2, poss=40)]))
    assert len(matches) == 1
    fixture = matches.iloc[0]
    assert fixture.date == pd.Timestamp("2025-01-01")
    assert (fixture.home_team, fixture.away_team) == ("Arsenal", "Wolves")
    assert (fixture.home_goals, fixture.away_goals) == (2, 1)
    assert (fixture.home_sh, fixture.away_sh, fixture.away_sot, fixture.away_poss) == (12, 7, 2, 40)


def test_load_normalizes_aliases_skips_unplayed_and_reports_duplicates(tmp_path):
    from football_predictor.data import load_matches
    matches = load_matches(write_rows(tmp_path, [row(opponent="Wolverhampton Wanderers"), row(),
        row(date="2025-02-01", gf=None, ga=None), row(venue="Away", team="Wolverhampton Wanderers",
        opponent="Arsenal", gf=1, ga=2, sh=7)]))
    assert len(matches) == 1
    assert matches.iloc[0].away_team == "Wolves"
    assert matches.iloc[0].away_sh == 7
    assert matches.attrs["duplicate_rows"] == 1
    assert matches.attrs["unplayed_rows"] == 1


@pytest.mark.parametrize("changes", [dict(gf=-1), dict(gf=1.5), dict(gf="oops"),
    dict(date="not-a-date"), dict(team=" "), dict(team="Wolves"), dict(venue="Neutral")])
def test_load_rejects_invalid_rows(tmp_path, changes):
    from football_predictor.data import load_matches
    with pytest.raises(ValueError):
        load_matches(write_rows(tmp_path, [row(**changes)]))


def test_load_requires_identity_columns_but_optional_stats_can_be_missing(tmp_path):
    from football_predictor.data import load_matches
    minimal = row()
    for col in ("sh", "sot", "poss"):
        minimal.pop(col)
    matches = load_matches(write_rows(tmp_path, [minimal]))
    assert matches[["home_sh", "away_sh", "home_sot", "away_poss"]].isna().all().all()
    minimal.pop("season")
    with pytest.raises(ValueError, match="season"):
        load_matches(write_rows(tmp_path, [minimal]))


@pytest.mark.parametrize("extra", [row(gf=3), row(venue="Away", team="Wolves", opponent="Arsenal", gf=0, ga=2)])
def test_conflicting_duplicate_or_reverse_scores_are_rejected(tmp_path, extra):
    from football_predictor.data import load_matches
    with pytest.raises(ValueError, match="[Cc]onflict|[Ii]nconsistent"):
        load_matches(write_rows(tmp_path, [row(), extra]))


def test_standings_points_ranking_seasons_and_exclusive_cutoff():
    from football_predictor.data import standings
    matches = pd.DataFrame([
        dict(date=pd.Timestamp("2024-01-01"), season=2024, home_team="A", away_team="B", home_goals=9, away_goals=0),
        dict(date=pd.Timestamp("2025-01-01"), season=2025, home_team="A", away_team="B", home_goals=1, away_goals=2),
        dict(date=pd.Timestamp("2025-01-02"), season=2025, home_team="B", away_team="A", home_goals=0, away_goals=0),
    ])
    table = standings(matches, season=2025, before="2025-01-02")
    assert table.team.tolist() == ["B", "A"]
    assert table.points.tolist() == [3, 0]
    assert table.played.tolist() == [1, 1]
    assert table.gf.tolist() == [2, 1]
    assert table.ga.tolist() == [1, 2]
    assert table.gd.tolist() == [1, -1]
    assert table.position.tolist() == [1, 2]
    assert standings(matches).points.tolist() == [4, 1]
    assert standings(matches, before="2024-02-01").points.tolist() == [3, 0]
    assert standings(matches, before="2020-01-01").empty
    assert standings(matches, season=2030).columns.tolist() == ["team", "played", "points", "gf", "ga", "gd", "position"]
