"""Convert soccerdata MatchHistory fixtures to the historical CSV schema."""
import numpy as np
from pathlib import Path
import pandas as pd

SOURCE = "football-data.co.uk via soccerdata.MatchHistory"
COLUMNS = ["date", "time", "comp", "round", "day", "venue", "result", "gf", "ga", "opponent", "xg", "xga", "poss", "attendance", "captain", "formation", "referee", "match report", "notes", "sh", "sot", "dist", "fk", "pk", "pkatt", "season", "team", "data_source"]
ALIASES = {"Man City": "Manchester City", "Man United": "Manchester Utd", "Nott'm Forest": "Nottingham", "Leeds": "Leeds United", "Ipswich": "Ipswich Town"}


def normalize_matchhistory(games: pd.DataFrame) -> pd.DataFrame:
    """Return two observed team perspectives per completed fixture."""
    if "season" in games.index.names:
        games = games.reset_index()
    required = {"season", "date", "home_team", "away_team", "FTHG", "FTAG"}
    if required.difference(games.columns):
        raise ValueError(f"Missing required MatchHistory columns: {sorted(required.difference(games.columns))}")
    rows = []
    for game in games.to_dict("records"):
        if pd.isna(game["FTHG"]) or pd.isna(game["FTAG"]):
            continue
        if "league" in game and game["league"] not in ("ENG-Premier League", "Premier League", "E0"):
            raise ValueError("Only Premier League observations are supported")
        import re
        season_text = str(game["season"])
        if re.fullmatch(r"\d{4}", season_text):
            start, end = int(season_text[:2]), int(season_text[2:])
            if (start + 1) % 100 != end:
                raise ValueError("Season must contain consecutive start/end years")
            ending_year = 2000 + end
        elif re.fullmatch(r"\d{4}[-/]\d{4}", season_text):
            start, ending_year = int(season_text[:4]), int(season_text[5:])
            if ending_year != start + 1:
                raise ValueError("Season must contain consecutive start/end years")
        else:
            raise ValueError("Invalid MatchHistory season encoding")
        date = pd.Timestamp(game["date"])
        if pd.isna(date) or date.date() > pd.Timestamp.today().date():
            raise ValueError("Invalid or future observed date")
        for field in ("home_team", "away_team"):
            if pd.isna(game[field]) or not str(game[field]).strip():
                raise ValueError("Missing team identity")
            name = str(game[field]).strip()
            game[field] = ALIASES.get(name, name)
        if game["home_team"] == game["away_team"]:
            raise ValueError("A team cannot play itself")
        goals = [float(game[key]) for key in ("FTHG", "FTAG")]
        if any(not np.isfinite(value) or value < 0 or not value.is_integer() for value in goals):
            raise ValueError("Goals must be nonnegative integers")
        home_goals, away_goals = map(int, goals)
        expected = "H" if home_goals > away_goals else "A" if home_goals < away_goals else "D"
        if pd.notna(game.get("FTR")) and game["FTR"] != expected:
            raise ValueError("FTR conflicts with score")
        for home in (True, False):
            gf, ga = (home_goals, away_goals) if home else (away_goals, home_goals)
            rows.append(dict(date=date.strftime("%Y-%m-%d"), time=date.strftime("%H:%M"),
                comp="Premier League", day=date.strftime("%a"), season=ending_year,
                team=ALIASES.get(game["home_team" if home else "away_team"], game["home_team" if home else "away_team"]),
                opponent=ALIASES.get(game["away_team" if home else "home_team"], game["away_team" if home else "home_team"]),
                venue="Home" if home else "Away", gf=gf, ga=ga, result="W" if gf > ga else "L" if gf < ga else "D",
                sh=game.get("HS" if home else "AS", np.nan), sot=game.get("HST" if home else "AST", np.nan),
                xg=pd.to_numeric(game.get("HxG" if home else "AxG", np.nan), errors="coerce"),
                xga=pd.to_numeric(game.get("AxG" if home else "HxG", np.nan), errors="coerce"),
                referee=game.get("referee", np.nan), data_source=SOURCE))
    result = pd.DataFrame(rows).reindex(columns=COLUMNS)
    result[["xg", "xga"]] = result[["xg", "xga"]].replace([np.inf, -np.inf], np.nan)
    return result


def load_matchhistory(seasons, cache_dir, refresh=False) -> pd.DataFrame:
    """Parse already acquired Premier League CSVs with soccerdata, offline.

    Current-season caches are intentionally reused. Missing caches and refresh
    requests fail explicitly rather than making implicit network requests.
    """
    if refresh:
        raise ValueError("Offline updater cannot refresh; acquire approved source files separately")
    from soccerdata import MatchHistory

    class CachedMatchHistory(MatchHistory):
        @classmethod
        def _all_leagues(cls):
            return MatchHistory._all_leagues()

        def _init_session(self, headers=None):
            # No transport is constructed: get() can only open existing files.
            return None

        def get(self, url, filepath, **kwargs):
            path = Path(filepath)
            if not path.is_file():
                raise FileNotFoundError(f"Missing offline MatchHistory cache: {path}")
            header = path.open(encoding="utf-8-sig").readline().strip().split(",")
            if not {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"}.issubset(header):
                raise ValueError(f"Not a MatchHistory CSV: {path}")
            return path.open("rb")

    if not seasons:
        raise ValueError("At least one season is required")
    return CachedMatchHistory(leagues="ENG-Premier League", seasons=list(seasons),
                              data_dir=Path(cache_dir)).read_games()


def merge_history(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Merge without replacing original metadata; conflicting scores are errors.

    Existing rows win whole-row precedence, including their missing values and
    source labels. Unnamed exported index columns are not match data.
    """
    keys = ["season", "team", "opponent", "date", "venue"]
    frames = [frame.loc[:, ~frame.columns.astype(str).str.startswith("Unnamed:")].copy()
              for frame in (existing, new)]
    combined = pd.concat(frames, ignore_index=True)
    if combined.empty:
        return combined
    if set(keys + ["gf", "ga"]).difference(combined.columns):
        raise ValueError("Missing fixture identity or scores")
    combined["date"] = pd.to_datetime(combined.date, errors="raise").dt.strftime("%Y-%m-%d")
    combined["season"] = pd.to_numeric(combined.season, errors="raise")
    if combined[keys].isna().any().any():
        raise ValueError("Missing fixture identity")
    for col in ("gf", "ga"):
        combined[col] = pd.to_numeric(combined[col], errors="raise")
    if (combined.groupby(keys)[["gf", "ga"]].nunique(dropna=True) > 1).any().any():
        raise ValueError("Conflicting duplicate fixture scores")
    return (combined.drop_duplicates(keys, keep="first")
            .sort_values(["season", "team", "date"], ascending=[False, True, False], kind="stable")
            .reset_index(drop=True))


def main(argv=None):
    """Build and validate a separate updated CSV from offline source caches."""
    import argparse
    from .data import load_matches

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("matches.csv"))
    parser.add_argument("--output", type=Path, default=Path("matches_updated.csv"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/raw/matchhistory"))
    parser.add_argument("--seasons", nargs="+", default=["2025-2026", "2026-2027"])
    parser.add_argument("--refresh", action="store_true", help="Rejected in offline mode; acquire approved source files separately")
    args = parser.parse_args(argv)
    if args.input.resolve() == args.output.resolve():
        raise ValueError("Output must be separate from input; original history is never overwritten")
    existing = pd.read_csv(args.input)
    games = load_matchhistory(args.seasons, args.cache_dir, args.refresh)
    merged = merge_history(existing, normalize_matchhistory(games))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Validate staging before publishing so invalid data never replaces output.
    import tempfile
    import os
    handle, name = tempfile.mkstemp(prefix=".matches-", suffix=".csv", dir=args.output.parent)
    os.close(handle)
    staging = Path(name)
    try:
        merged.to_csv(staging, index=False)
        fixtures = load_matches(staging)
        staging.replace(args.output)
    finally:
        staging.unlink(missing_ok=True)
    print(f"Wrote {len(merged)} team rows / {len(fixtures)} fixtures to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
