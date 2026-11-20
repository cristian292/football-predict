"""Offline, synthetic tests for MatchHistory ingestion."""
import pandas as pd
import pytest

from football_predictor import acquisition


def game(**changes):
    return dict(season="2526", date=pd.Timestamp("2025-08-16 15:00"),
                home_team="Man City", away_team="Nott'm Forest", FTHG=2, FTAG=1,
                FTR="H", HS=14, AS=8, HST=6, AST=3, referee="A Ref") | changes


@pytest.mark.parametrize("changes", [dict(FTHG=-1), dict(FTAG=1.5), dict(FTHG="oops"), dict(FTR="A"), dict(date=None), dict(date="bad"), dict(date="2099-01-01"), dict(home_team="Nott'm Forest"), dict(home_team=" "), dict(season="bad")])
def test_normalize_rejects_invalid_observations(changes):
    with pytest.raises(ValueError):
        acquisition.normalize_matchhistory(pd.DataFrame([game(**changes)]))


def test_normalize_accepts_soccerdata_index_skips_unplayed_preserves_real_xg():
    games = pd.DataFrame([game(HxG=1.7, AxG=0.4), game(date="2025-08-17", FTHG=None, FTAG=None)])
    games = games.assign(league="ENG-Premier League", game=["one", "two"]).set_index(["league", "season", "game"])
    result = acquisition.normalize_matchhistory(games)
    assert len(result) == 2
    assert result.xg.tolist() == [1.7, 0.4]
    assert result.xga.tolist() == [0.4, 1.7]
    assert acquisition.normalize_matchhistory(pd.DataFrame([game(HxG=float("inf"))])).xg.isna().all()


@pytest.mark.parametrize("season", ["2025-2026", "2025/2026", 2526, "2526"])
def test_normalize_understands_season_encodings(season):
    result = acquisition.normalize_matchhistory(pd.DataFrame([game(season=season)]))
    assert result.season.eq(2026).all()


def test_merge_preserves_original_metadata_and_source_deduplicates_stably():
    new = acquisition.normalize_matchhistory(pd.DataFrame([game(), game(date="2025-08-17")]))
    existing = new.iloc[:2].assign(**{"data_source": "original", "captain": "Original captain", "Unnamed: 0": [99, 100]})
    before = existing.copy(deep=True)
    result = acquisition.merge_history(existing, new)
    assert len(result) == 4
    assert "Unnamed: 0" not in result
    assert result.loc[result.date.eq("2025-08-16"), "captain"].eq("Original captain").all()
    assert result.loc[result.date.eq("2025-08-16"), "data_source"].eq("original").all()
    pd.testing.assert_frame_equal(existing, before)
    pd.testing.assert_frame_equal(result, acquisition.merge_history(result, new))
    assert result.equals(result.sort_values(["season", "team", "date"], ascending=[False, True, False], kind="stable").reset_index(drop=True))


def test_merge_rejects_conflicting_scores():
    rows = acquisition.normalize_matchhistory(pd.DataFrame([game()]))
    changed = rows.copy()
    changed.loc[0, "gf"] = 99
    with pytest.raises(ValueError, match="[Cc]onflict"):
        acquisition.merge_history(rows, changed)


def test_cache_loader_uses_real_soccerdata_parser_without_network(tmp_path):
    pytest.importorskip("soccerdata")
    (tmp_path / "E0_2526.csv").write_text("Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HS,AS,HST,AST,Referee\nE0,16/08/2025,15:00,Man City,Nott'm Forest,2,1,H,14,8,6,3,A Ref\n")
    games = acquisition.load_matchhistory(["2025-2026"], tmp_path)
    result = acquisition.normalize_matchhistory(games)
    assert len(result) == 2 and result.gf.tolist() == [2, 1]
    with pytest.raises(FileNotFoundError):
        acquisition.load_matchhistory(["2026-2027"], tmp_path)


def test_cli_merges_offline_to_separate_validated_output(tmp_path):
    from football_predictor.data import load_matches
    pytest.importorskip("soccerdata")
    (tmp_path / "E0_2526.csv").write_text("Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR\nE0,16/08/2025,15:00,Man City,Nott'm Forest,2,1,H\n")
    input_path, output_path = tmp_path / "matches.csv", tmp_path / "updated.csv"
    acquisition.normalize_matchhistory(pd.DataFrame([game(date="2025-08-15")])).to_csv(input_path, index=False)
    before = input_path.read_bytes()
    args = ["--input", str(input_path), "--output", str(output_path), "--cache-dir", str(tmp_path), "--seasons", "2025-2026"]
    assert acquisition.main(args) == 0
    assert input_path.read_bytes() == before
    assert len(load_matches(output_path)) == 2
    with pytest.raises(ValueError, match="[Ss]eparate"):
        acquisition.main(["--input", str(input_path), "--output", str(input_path), "--cache-dir", str(tmp_path), "--seasons", "2025-2026"])


def test_normalize_rejects_other_league_instead_of_relabeling_it():
    with pytest.raises(ValueError, match="[Ll]eague"):
        acquisition.normalize_matchhistory(pd.DataFrame([game(league="ESP-La Liga")]))


@pytest.mark.parametrize("season", ["bad26", "2527", "2025-2027"])
def test_normalize_rejects_malformed_or_nonconsecutive_season(season):
    with pytest.raises(ValueError, match="[Ss]eason"):
        acquisition.normalize_matchhistory(pd.DataFrame([game(season=season)]))


def test_normalize_produces_reciprocal_rows_without_inventing_stats():
    result = acquisition.normalize_matchhistory(pd.DataFrame([game()]))
    assert len(result) == 2
    home, away = result.iloc[0], result.iloc[1]
    assert (home.team, away.team) == ("Manchester City", "Nottingham")
    assert (home.opponent, away.opponent) == (away.team, home.team)
    assert (home.gf, home.ga, away.gf, away.ga) == (2, 1, 1, 2)
    assert (home.result, away.result, home.venue, away.venue) == ("W", "L", "Home", "Away")
    assert (home.sh, home.sot, away.sh, away.sot) == (14, 6, 8, 3)
    assert home.date == "2025-08-16" and home.time == "15:00" and home.season == 2026
    assert home.comp == "Premier League" and home.referee == "A Ref"
    assert result[["poss", "attendance", "captain", "formation", "dist", "fk", "pk", "pkatt", "xg", "xga"]].isna().all().all()
    assert result.data_source.eq("football-data.co.uk via soccerdata.MatchHistory").all()
    assert result.columns.tolist() == ["date", "time", "comp", "round", "day", "venue", "result", "gf", "ga", "opponent", "xg", "xga", "poss", "attendance", "captain", "formation", "referee", "match report", "notes", "sh", "sot", "dist", "fk", "pk", "pkatt", "season", "team", "data_source"]
