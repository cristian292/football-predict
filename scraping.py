"""
Scraping FBref cu soccerdata (inlocuieste requests + BeautifulSoup).

Produce matches.csv cu aceeasi structura ca notebook-ul original:
meciuri de Premier League + statistici de sut (sh, sot, dist, fk, pk, pkatt),
cu coloanele season (anul de final al sezonului) si team.

Instalare:  pip install soccerdata pandas
Rulare:     python scrape_fbref.py
"""
import pandas as pd
import soccerdata as sd

# ---------------- Setari ----------------
LEAGUE = "ENG-Premier League"
# Sezoane complete. In notebook: 2021 = sezonul 2020-21, 2022 = sezonul 2021-22
SEASONS = ["2024-2025"]
OUT_FILE = "matches.csv"

SHOOTING_COLS = ["sh", "sot", "dist", "fk", "pk", "pkatt"]
KEYS = ["league", "season", "team", "game"]


def flatten(df: pd.DataFrame) -> pd.DataFrame:
    """Aplatizeaza coloanele (MultiIndex -> nume simple, lowercase)
    si elimina coloanele duplicate."""
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        names = []
        for col in df.columns:
            parts = [str(p) for p in col if str(p) and not str(p).startswith("Unnamed")]
            names.append(parts[-1] if parts else "")
        df.columns = names
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df.loc[:, ~df.columns.duplicated()]


def find_col(df: pd.DataFrame, name: str):
    """Gaseste coloana 'name', chiar daca are prefix (ex. standard_sh)."""
    for c in df.columns:
        if c == name or c.endswith("_" + name):
            return c
    return None


def season_label(code) -> int:
    """'2122' -> 2022 (anul de final al sezonului, ca in notebook)."""
    s = str(code)
    if len(s) == 4 and s.isdigit():
        return 2000 + int(s[2:])
    return int(s[-4:])


def main():
    # soccerdata respecta singur limita FBref (pauze intre cereri) si
    # tine cache local in ~/soccerdata, deci a doua rulare e instantanee.
    fbref = sd.FBref(leagues=LEAGUE, seasons=SEASONS)

    print("Descarc meciurile (schedule)...")
    sched = flatten(fbref.read_team_match_stats(stat_type="schedule").reset_index())

    print("Descarc statisticile de sut (shooting)...")
    shoot = flatten(fbref.read_team_match_stats(stat_type="shooting").reset_index())

    # Pastram doar coloanele de sut de care avem nevoie
    keep = {}
    for name in SHOOTING_COLS:
        col = find_col(shoot, name)
        if col is not None:
            keep[col] = name
        else:
            print(f"Atentie: coloana '{name}' lipseste din FBref pentru aceste sezoane.")
    shoot_sel = shoot[KEYS + list(keep.keys())].rename(columns=keep)

    # Unim pe (liga, sezon, echipa, meci), la fel ca merge-ul pe Date din notebook
    df = sched.merge(shoot_sel, on=KEYS, how="left")

    # Coloane ca in notebook
    df["comp"] = "Premier League"
    df["season"] = df["season"].map(season_label)
    df = df.drop(columns=["league", "game"], errors="ignore")

    # Ordine coloane: cele din notebook primele
    first = ["date", "time", "comp", "round", "day", "venue", "result",
             "gf", "ga", "opponent"]
    ordered = [c for c in first if c in df.columns]
    rest = [c for c in df.columns if c not in ordered + ["season", "team"]]
    df = df[ordered + rest + ["season", "team"]]

    df = df.sort_values(["season", "team", "date"], ascending=[False, True, True])
    df.to_csv(OUT_FILE, index=False)
    print(f"Gata: {len(df)} randuri, {df['team'].nunique()} echipe -> {OUT_FILE}")


if __name__ == "__main__":
    main()