# Football Predictor — GTK4

Aplicație desktop în Python pentru **analiza și predicția rezultatelor meciurilor de fotbal**, proiect universitar. Interfață în română, separată de prelucrarea datelor și de algoritmi.

## Pornire în mediul local existent

```bash
cd ~/Documents/py/gambling
./run.sh
```

`run.sh` folosește `.venv/bin/python`. Pe Debian/Parrot, dacă este compatibil, adaugă la finalul căii de import legăturile GI instalate de sistem; nu înlocuiește pachetele științifice ale mediului virtual. Pentru alt interpretor: `FOOTBALL_PYTHON=/cale/python ./run.sh`.

## Instalare curată pe Linux Debian/Parrot/Ubuntu

Dependențele native GTK nu sunt incluse în `requirements.txt`:

```bash
sudo apt install python3-venv python3-gi gir1.2-gtk-4.0
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -e '.[dev]'
./run.sh --data /cale/catre/matches.csv
```

Nu recrea peste un mediu existent fără backup. Python din venv trebuie să fie compatibil cu PyGObject al distribuției. Instrucțiunile de mai sus sunt pentru Linux; distribuirea pentru Windows/macOS necesită împachetare și dependențe native separate.

## Structură

```text
src/football_predictor/
  cli.py                 # lansare desktop / evaluare fără display
  data.py                # validare CSV, normalizare nume, clasament
  features.py            # statistici exclusiv pre-meci
  models/                # KNN, Random Forest, Poisson
  evaluation.py          # împărțire cronologică și metrici
  service.py             # orchestrare antrenare, predicții, export
  ui/                    # interfață GTK4
 tests/                  # teste automate pentru noua aplicație
 docs/                   # metodologie și pregătire GitHub
 matches.csv             # date locale; ignorate implicit de Git
 scraping.py             # script existent, păstrat
 download_fbref.py        # script existent, păstrat
 html_to_py.py            # convertor existent, păstrat
```

## Date

Fișierul principal `matches.csv` a fost completat prin `soccerdata.MatchHistory` și conține **5.520 de perspective de echipă**, între **2019-08-09 și 2026-11-08**, reprezentând **2.760 de meciuri distincte**. Cele 480 de meciuri noi provin din football-data.co.uk; statisticile indisponibile au rămas nule. Operațiile, sursele, backupul și verificările sunt documentate în **`DOCUMENTARE_DATE.md`**. Un meci apare de două ori în sursă: o dată pentru gazde și o dată pentru oaspeți. Aplicația construiește o singură observație pe meci, orientată gazde/oaspeți. Nu interpreta numărul de rânduri brute drept număr de meciuri distincte.

Coloane de bază în sursă: `date`, `season`, `team`, `opponent`, `venue`, `gf`, `ga`. Statisticile `sh`, `sot`, `poss` sunt utilizate numai din meciurile anterioare. Coloanele complet goale, de exemplu unele valori xG din această extracție, nu sunt prezentate drept informație disponibilă. Nu se descarcă automat date de pe internet.

**Datele se opresc la 8 noiembrie 2026.** Selectarea unei date ulterioare nu actualizează loturile sau forma echipelor. Predicțiile sunt simulări bazate pe istoricul disponibil, nu o descriere la zi a fotbalului.

## Algoritmi și experimente

- **KNN:** caracteristici de bază, normalizare și selecție de hiperparametri.
- **Random Forest:** caracteristici istorice mai complexe și selecție de hiperparametri.
- **Poisson:** intensități de gol și distribuție de scor; optimizare numerică a parametrilor.
- **Reper simplu:** pentru a verifica dacă modelele aduc un avantaj real.

Ținta este cu trei clase: `0 = victorie gazde`, `1 = egal`, `2 = victorie oaspeți`. Procentele sunt estimări ale modelelor, nu certitudini și nici garanții de profit.

Împărțirea datelor este cronologică, fără amestecare aleatorie; o zi de meci nu este împărțită între două seturi. Hiperparametrii se selectează pe validare, nu pe test. Statisticile istorice din test pot include rezultatele observate ale meciurilor de test anterioare: aceasta este evaluare secvențială pre-meci, nu prognoza unui sezon întreg dintr-o singură dată.

```bash
./run.sh --evaluate --output reports
```

Exporturile includ tabelele cu metrici și căutarea hiperparametrilor, predicțiile de test și un raport JSON. Rezultatele experimentale trebuie generate din execuție, nu copiate din notebook ca performanță a noii aplicații.

## Evaluare pe sezoane și analiza erorilor

```bash
./run.sh --benchmark --output reports/optimization
```

Compară configurațiile `legacy` și `optimized` pe aceleași meciuri, cu selecție numai pe sezonul anterior de validare. Generează tabele pe sezoane, predicții, metrici pe clase, calibrare și erori pe echipe. Rezultatele verificate și limitele sunt în `DOCUMENTARE_DATE.md`, secțiunea 12. KNN și Poisson au câștiguri agregate mici; căutarea extinsă nu îmbunătățește Random Forest. Interfața desktop nu a fost trecută automat pe configurațiile experimentale. Identificarea egalurilor rămâne o limită importantă.

## Testare

```bash
.venv/bin/python -m pytest tests -q
# GTK necesită display; pe un sistem fără display:
sudo apt install xvfb
PYTHONPATH=src xvfb-run -a .venv/bin/python -c "import sys; sys.path.append('/usr/lib/python3/dist-packages'); import pytest; raise SystemExit(pytest.main(['tests', '-q']))"
```

Testele vechilor scripturi din rădăcină pot necesita cache HTML local; acestea nu sunt condiții pentru instalarea aplicației de predicție.

## Notebookul inițial

Referință locală analizată: `~/Documents/projects/football_predict/main.ipynb`. Puncte reutilizate conceptual: Random Forest, medii mobile deplasate cu un meci și separare temporală. Notebookul inițial prezice victorie/non-victorie; aplicația extinde problema la trei clase, modele distincte și probabilități de scor. Notebookul și datele din celălalt director nu sunt modificate.

## Rezultate și metodologie

Rezultatele **actualizării prin soccerdata** sunt în `DOCUMENTARE_DATE.md` și `reports/soccerdata_update/`. Tabelele din `docs/RESULTS.md` descriu experimentul istoric, înaintea actualizării; explicațiile algoritmilor și limitele evaluării sunt în `docs/METHODOLOGY.md`. Workflow-ul `.github/workflows/tests.yml` este pregătit pentru teste automate pe GitHub, dar nu a fost rulat acolo.

## GitHub

Vezi `docs/GITHUB.md`. Mediul virtual, cache-urile și rapoartele generate sunt ignorate. Datele sunt ignorate implicit până la verificarea dreptului de redistribuire. Nu s-a ales o licență în numele autorului și nu se publică automat în niciun cont.

## Documentație tehnică primară

- GTK Python: <https://www.gtk.org/docs/language-bindings/python>
- PyGObject instalare: <https://pygobject.gnome.org/getting_started.html>
- KNN: <https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html>
- Random Forest: <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html>
- Optimizare SciPy: <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>
- Poisson SciPy: <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.poisson.html>
