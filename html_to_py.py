import re
import sys
from io import StringIO
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

SHOOTING_COLS = ["Sh", "SoT", "Dist", "FK", "PK", "PKatt"]
# Aceeasi schema si ordine ca football_matches/matches.csv de la Dataquest.
CSV_COLUMNS = [
    "date", "time", "comp", "round", "day", "venue", "result", "gf", "ga",
    "opponent", "xg", "xga", "poss", "attendance", "captain", "formation",
    "referee", "match report", "notes", "sh", "sot", "dist", "fk", "pk",
    "pkatt", "season", "team",
]


def read_table(html: str, caption: str):
    """Prima tabela care contine textul `caption`, sau None."""
    try:
        return pd.read_html(StringIO(html), match=caption, flavor="lxml")[0]
    except ValueError:
        return None


def flatten(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina nivelul de sus al antetului (ca droplevel din notebook)."""
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(c[-1]).strip() for c in df.columns]
    else:
        df.columns = [str(c).strip() for c in df.columns]
    return df.loc[:, ~df.columns.duplicated()]


def only_premier_league(df: pd.DataFrame) -> pd.DataFrame:
    if "Comp" in df.columns:
        df = df[df["Comp"] == "Premier League"]
    return df[df["Date"].notna()]


def team_and_season(html: str, path=None):
    """Echipa si anul final: din titlu sau numele cache-ului soccerdata."""
    soup = BeautifulSoup(html, "lxml")
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    m = re.search(r"^(.*?)\s+(\d{4})-(\d{4})", title)
    if m:
        return m.group(1).strip(), int(m.group(3))
    if path is not None:
        cached = re.fullmatch(r"matchlogs_(.+)_(\d{4})_(schedule|shooting)", Path(path).stem)
        if cached:
            team, code, _ = cached.groups()
            return team.replace("_", " "), 2000 + int(code[2:])
    return None, None


def main():
    folder = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else Path.home() / "soccerdata/data/FBref"
    if not folder.is_dir():
        raise NotADirectoryError(f"Folderul nu exista: {folder}")
    out_file = sys.argv[2] if len(sys.argv) > 2 else "matches.csv"

    files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in (".html", ".htm"))
    print(f"{len(files)} fisiere HTML gasite in {folder}")

    schedules, shootings = {}, {}  # (team, season) -> DataFrame
    for path in files:
        html = path.read_text(encoding="utf-8", errors="ignore")
        team, season = team_and_season(html, path)
        if team is None:
            continue

        sh = read_table(html, "Shooting")
        if sh is not None:
            sh = only_premier_league(flatten(sh))
            if "Sh" in sh.columns:
                shootings[(team, season)] = sh

        sc = read_table(html, "Scores & Fixtures")
        if sc is not None:
            sc = flatten(sc)
            if "Date" in sc.columns:
                schedules[(team, season)] = only_premier_league(sc)

    print(f"{len(schedules)} tabele de meciuri, {len(shootings)} tabele de sut")

    all_matches = []
    for key, matches in schedules.items():
        shooting = shootings.get(key)
        if shooting is None:
            print(f"  lipseste shooting pentru {key}, il sar")
            continue
        cols = ["Date"] + [c for c in SHOOTING_COLS if c in shooting.columns]
        shooting = shooting[cols].drop_duplicates("Date")
        team_data = matches.merge(shooting, on="Date")
        team_data["Season"] = key[1]
        team_data["Team"] = key[0]
        all_matches.append(team_data)

    if not all_matches:
        print("Nu am gasit nimic de unit. Verifica folderul.")
        return

    match_df = pd.concat(all_matches)
    match_df.columns = [c.strip().lower() for c in match_df.columns]
    # Coloanele absente devin NaN; valorile existente nu sunt inlocuite.
    match_df = match_df.reindex(columns=CSV_COLUMNS)
    match_df = match_df.sort_values(["season", "team", "date"], ascending=[False, True, True])
    match_df = match_df.reset_index(drop=True)
    for col in CSV_COLUMNS:
        missing = int(match_df[col].isna().sum())
        if missing:
            print(f"Atentie: '{col}' are {missing}/{len(match_df)} valori lipsa in datele sursa.")
    # GitHub include o prima coloana fara nume pentru index.
    match_df.to_csv(out_file, index=True, index_label="")
    print(f"Gata: {len(match_df)} randuri, {match_df['team'].nunique()} echipe -> {out_file}")


if __name__ == "__main__":
    main()