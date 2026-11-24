# Football Predictor — aplicație web

Aplicație în Python pentru **analiza și predicția rezultatelor meciurilor de fotbal**, cu interfață în română, direct în browser.

Versiunea web nu folosește GTK, GDK, noVNC sau un display virtual. Modelele și prelucrarea datelor sunt separate de prezentare. Containerizarea va fi pregătită ulterior, împreună cu autorul proiectului; în această etapă nu se construiește o imagine Docker.

## Pornire rapidă pe Linux/macOS

Ai nevoie de Python **3.11 sau mai nou** și de acces la repository-ul privat. Din directorul proiectului:

```bash
bash install.sh
bash run.sh --doctor
bash run.sh
```

Apoi deschide în browser:

```text
http://127.0.0.1:8080
```

Instalarea nouă creează o `.venv` izolată. Nu instalează pachete de sistem automat și nu descarcă meciuri. Nu sunt necesare GTK sau PyGObject. Pentru verificarea prerechizitelor fără instalare: `bash install.sh --check-only`.

### Instalare manuală, inclusiv Windows

Linux/macOS:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m football_predictor
```

Windows, în PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m football_predictor
```

Aceste instrucțiuni folosesc Python și browserul, nu un server grafic Linux. Verificările locale executate și limitele testării pe alte sisteme sunt consemnate în `DOCUMENTARE_DATE.md`.

### Port, date și directorul de lucru

```bash
bash run.sh --port 8081
bash run.sh --data /cale/catre/matches.csv
bash run.sh --work-dir /cale/catre/date-web
```

Implicit, serverul ascultă numai pe `127.0.0.1`. Rulează cu Waitress, fără debugger sau reloader Flask. Oprește-l cu Ctrl+C.

## Utilizarea interfeței

1. Verifică numele fișierului și intervalul datelor încărcate.
2. Apasă **Antrenează modelele**. Operația rulează în fundal; progresul este afișat în pagină.
3. În **Predicții**, selectează două echipe diferite și o dată ulterioară ultimei partide din date.
4. Compară probabilitățile 1/X/2, scorurile Poisson și matricea de scor.
5. În **Experimente**, consultă metricile, separarea cronologică, configurațiile și matricile de confuzie.
6. În **Date**, verifică clasamentele și istoricul echipelor.
7. Importă un CSV pentru alt istoric sau descarcă rapoartele CSV/JSON din interfață.

Selectarea echipelor și a datei este o **simulare**, nu confirmarea unui meci programat. Datele nu sunt actualizate automat de pe internet.

Importul valid resetează modelele și rezultatele anterioare. Un fișier invalid nu trebuie să înlocuiască istoricul deja încărcat. Fișierul original `matches.csv` nu este suprascris de importurile web.

## Structură modulară

```text
src/football_predictor/
  web/
    app.py                # rute Flask, validare HTTP și descărcări
    workspace.py          # stare locală și operații în fundal
    templates/index.html  # interfața web
    static/app.css        # stiluri responsive
    static/app.js         # interacțiuni și cereri către API
  cli.py                  # server web și experimente fără interfață
  doctor.py               # verificarea mediului și antetului CSV
  data.py                 # validare date și clasamente
  features.py             # caracteristici exclusiv pre-meci
  models/                 # KNN, Random Forest și Poisson
  service.py              # antrenare, predicție și export reutilizabile
  evaluation.py           # separare cronologică și metrici
  selection.py            # selecția hiperparametrilor pe validare
  experiments.py          # evaluare retrospectivă pe sezoane
  diagnostics.py          # analiza erorilor
```

Directorul vechi `ui/` și unele configurații Docker GTK sunt păstrate ca istoric, nu sunt folosite de pornirea implicită. Instrucțiunile istorice Docker/GTK nu reprezintă distribuirea noii aplicații web.

## Date și algoritmi

`matches.csv`, inclus intenționat în repository-ul privat, are **5.520 de perspective de echipă**, reprezentând **2.760 de meciuri distincte**, între **2019-08-09 și 2026-11-08**. Două perspective ale aceluiași meci nu sunt tratate ca meciuri diferite.

- **KNN:** statistici istorice de bază și standardizare ajustată numai pe antrenare.
- **Random Forest:** setul de caracteristici istorice mai complex.
- **Poisson:** probabilități de gol/scor și ajustare numerică a intensităților.
- **Reper majoritar:** comparație simplă pentru evaluarea câștigului real.

Clasele sunt `0 = gazde`, `1 = egal`, `2 = oaspeți`. Hiperparametrii sunt selectați pe validare, nu pe test. Caracteristicile meciurilor testate folosesc numai rezultatele observate anterior; protocolul este secvențial pre-meci, nu prognoza unui sezon întreg dintr-o singură dată.

**Nu există garanții asupra rezultatului sau profitului.** Identificarea egalurilor este o limită măsurată a modelelor actuale. Nu au fost promovate automat configurațiile experimentale care au obținut rezultate mai slabe.

## Experimente din terminal

```bash
bash run.sh --evaluate --output reports
bash run.sh --benchmark --output reports/optimization
```

Tabelele agregate, explicațiile și rezultatele istorice sunt în `DOCUMENTARE_DATE.md`, `docs/METHODOLOGY.md` și `docs/RESULTS.md`. Directoarele `reports/` sunt generate local, nu vin automat cu o clonare nouă.

## Testare

```bash
bash install.sh --dev
PYTHONNOUSERSITE=1 .venv/bin/python -m pytest tests -q
```

Testele web nu necesită Xvfb sau GTK. Pentru testele de interacțiune JavaScript sunt folosite Node.js și jsdom numai în dezvoltare; pe Debian/Ubuntu: `sudo apt install nodejs node-jsdom`. Nu sunt necesare pentru rularea site-ului și nu există un pas de build npm. CI verifică prezența lor înainte de teste. Testele vechii interfețe GTK și ale schiței Docker sunt arhivate în `legacy-tests/`, în afara colectării implicite; patru eșecuri X11 cunoscute ale vechiului launcher nu sunt prezentate drept remediate. Testele scripturilor vechi de colectare din rădăcină pot necesita un cache HTML local și nu fac parte din pornirea site-ului.

## Stare verificată

Pe Linux, în mediul izolat fără GTK: **224 de teste trecute, 2 integrări opționale soccerdata omise**. Fluxul real în browser, importurile invalide/valide, descărcările și afișarea desktop/mobil au fost verificate. Wheel-ul a fost construit și instalat separat, cu resursele web incluse. Detalii: `DOCUMENTARE_DATE.md`; dovezile locale generate sunt în `reports/web-qa/report.md`. Windows/macOS și imaginea Docker nu sunt prezentate drept testate.

## Stare, importuri și securitate

Această versiune este un **spațiu de lucru local, pentru un singur utilizator**, într-un singur proces Waitress. Nu include conturi, autentificare sau separarea datelor între utilizatori.

- Antrenarea și celelalte operații grele sunt serializate într-un worker; cererile HTTP nu așteaptă antrenarea completă.
- Starea și modelele sunt în memorie. Reîmprospătarea paginii nu oprește operația; repornirea serverului necesită reantrenare.
- Importurile și exporturile web sunt salvate în directorul de lucru, implicit `data/webapp/`, ignorat de Git.
- CSV-ul importat este limitat la 10 MiB și validat înainte de înlocuirea sursei active.
- Rutele care schimbă starea cer un token CSRF. Acesta nu reprezintă autentificare.
- Descărcările sunt limitate la rapoartele generate de aplicație; utilizatorul nu poate cere arbitrar fișiere de pe server.
- Resursele interfeței sunt locale; nu se folosesc CDN-uri, tracking sau date demonstrative prezentate drept rezultate reale.

**Nu expune serverul direct pe internet.** Pentru publicare externă sunt necesare autentificare, HTTPS și o arhitectură multi-utilizator. `--host 0.0.0.0` este o opțiune de configurare pentru etapa viitoare de distribuire, nu o recomandare de expunere publică.

## GitHub și Docker

Repository: **privat**, `cristian292/football-predict`. Accesul colegilor se acordă prin conturile lor GitHub, fără partajarea credențialelor.

Copia activă de lucru este `~/Documents/football_predict_github`. Datele personale, `.venv`, cache-urile, importurile și rapoartele nu se publică accidental. Nu s-a ales o licență pentru cod în numele autorului; drepturile asupra datelor trebuie verificate separat înaintea oricărei publicări publice.

**Docker este etapa următoare, făcută împreună cu utilizatorul.** Configurațiile Docker vechi din proiect sunt pentru versiunea GTK și trebuie revizuite pentru serverul web. Nu este necesar noVNC sau acces la Wayland/X11 pentru noua aplicație.

## Documentație tehnică primară

- Flask: <https://flask.palletsprojects.com/>
- Waitress: <https://docs.pylonsproject.org/projects/waitress/en/stable/>
- scikit-learn: <https://scikit-learn.org/stable/>
- SciPy: <https://docs.scipy.org/doc/scipy/>
