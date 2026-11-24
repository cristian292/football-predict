# Documentarea actualizării datelor — 19 noiembrie 2026

**Stare: finalizat.** Datele noi sunt adăugate în fișierul principal `/home/hxh/Documents/py/gambling/matches.csv`, conform solicitării explicite a utilizatorului. Acest document centralizează operațiile, sursele, limitările și verificările acestei actualizări.

## 1. Situația inițială și obiectivul

Fișierul inițial avea 4.560 de rânduri de echipă, reprezentând 2.280 de meciuri Premier League din sezoanele 2019–2020 până la 2024–2025. Ultimul rezultat era din 25 mai 2025.

Obiectiv: completarea cu rezultatele disponibile pentru 2025–2026 și 2026–2027, folosind biblioteca `soccerdata`, fără inventarea statisticilor indisponibile. Nu au fost adăugate alte campionate.

## 2. Surse și încercări de acces

### FBref

- S-a verificat biblioteca instalată în mediul virtual al proiectului: `soccerdata 2026.43.0`.
- S-au inspectat API-ul și cache-ul local. Acesta conținea câte 20 de jurnale de echipă pentru fiecare dintre cele șase sezoane deja prezente.
- O singură cerere nouă către `https://fbref.com/en/comps/` a primit **HTTP 403**.
- Cererile către FBref au fost oprite: fără reîncercări repetate, rezolvare CAPTCHA, rotație de IP sau proxy.
- Rezultatul este păstrat în `data/raw/matchhistory/fbref_stopped_manifest.json`.

### football-data.co.uk prin soccerdata.MatchHistory

Utilizatorul a cerut biblioteca soccerdata, fără a limita achiziția exclusiv la FBref. Sursa alternativă a fost anunțată explicit în conversație.

Fișierele efectiv obținute:

| Sezon | Fișier sursă | Rezultate | Interval observat |
|---|---|---:|---|
| 2025–2026 | `https://football-data.co.uk/mmz4281/2526/E0.csv` | 380 | 2025-08-15 – 2026-05-24 |
| 2026–2027 | `https://football-data.co.uk/mmz4281/2627/E0.csv` | 100 | 2026-08-21 – 2026-11-08 |

Ambele răspunsuri finale au fost HTTP 200. Adresele cu `www` au redirecționat către adresele fără `www` de mai sus.

Transportul TLS implicit al versiunii instalate a produs `Too many redirects`. S-a folosit un subclass local al `soccerdata.MatchHistory` cu o sesiune HTTP standard `requests.Session`; parsarea a rămas cea reală din `MatchHistory.read_games()`. Nu s-au schimbat proxy-uri și nu s-au utilizat mecanisme de ocolire a protecțiilor. Cererile de achiziție au fost seriale, cu pauze de 15 secunde și timeout de 40 de secunde. Un test de diagnostic standard HTTPS confirmase anterior accesul la fișierul 2025–2026.

**Nu pretindem că există date până la 19 noiembrie:** ultima partidă efectiv prezentă în sursa descărcată este din **8 noiembrie 2026**. Sezonul 2026–2027 este incomplet.

## 3. Fișiere locale și proveniență

În `data/raw/matchhistory/` sunt păstrate:

- `E0_2526.csv`, `E0_2627.csv`: fișierele brute folosite de soccerdata;
- `2025-2026.csv`, `2026-2027.csv`: rezultatul parsării prin soccerdata;
- `manifest.json`: adresele, statusurile HTTP, hashurile, coloanele și intervalele observate;
- `fbref_stopped_manifest.json`: dovada opririi accesului FBref.

Fișierele au fost copiate din spațiul temporar în proiect, iar copiile au fost verificate prin SHA-256. Nu au fost modificate pachetele instalate soccerdata pentru a altera parsarea.

Hashuri SHA-256 ale fișierelor brute:

```text
E0_2526.csv
3e3a8352f9ada6789c508d6ca184424421fed56a30400904a4a327c583407e62

E0_2627.csv
d598a5f681e1a9a4acf9dad9fb203b68da6f67e2d195678dbe99e95fbba14593
```

Coloana nouă `data_source` din `matches.csv` distinge:

- `existing_local_extraction`: cele 4.560 de rânduri existente;
- `football-data.co.uk via soccerdata.MatchHistory`: cele 960 de rânduri noi.

Eticheta istorică identifică fișierul local păstrat; nu pretinde o nouă verificare online a fiecărui rând vechi.

## 4. Copia de siguranță înainte de modificare

Fișier original salvat și verificat înainte de îmbinare:

`data/backups/matches.before_soccerdata_2026-11-19.csv`

Dimensiune: **770.557 de octeți**.

SHA-256 original și backup identic:

```text
1282b370fc9dc7043148607c06b588e57020e53a79427e8fb022f9cd5cf33b79
```

Pentru revenire manuală, dacă este necesar:

```bash
cp data/backups/matches.before_soccerdata_2026-11-19.csv matches.csv
```

Această comandă este doar documentată; nu a fost executată după actualizare.

## 5. Transformările aplicate

S-a adăugat modulul `src/football_predictor/acquisition.py` și fișierul de teste `tests/test_acquisition.py`.

1. Fiecare rezultat a fost convertit în două perspective: `Home` și `Away`, în aceeași schemă ca datele existente.
2. `FTHG`/`FTAG` au devenit `gf`/`ga`; rezultatul `W`, `D` sau `L` a fost derivat din scor și verificat împotriva `FTR` când acesta exista.
3. `HS`/`AS` au devenit `sh`; `HST`/`AST` au devenit `sot` pentru perspectiva corectă.
4. `HxG`/`AxG`, unde erau efectiv prezente, au devenit `xg` și `xga` reciproc.
5. Data a fost exportată `YYYY-MM-DD`; ora disponibilă în sursă a fost păstrată separat. Fusurile orare nu au fost armonizate între furnizori; modelele folosesc data, nu ora de start.
6. Codurile soccerdata `2526` și `2627` au devenit anii de încheiere `2026` și `2027`, compatibili cu fișierul istoric.
7. Au fost normalizate numele pentru a evita crearea unor echipe duplicate:

| Sursă nouă | Nume în proiect |
|---|---|
| Man City | Manchester City |
| Man United | Manchester Utd |
| Nott'm Forest | Nottingham |
| Leeds | Leeds United |
| Ipswich | Ipswich Town |

8. Cheia de deduplicare este `(season, team, opponent, date, venue)`. Scorurile contradictorii sunt respinse; rândurile existente au prioritate la suprapunere.
9. S-au păstrat coloanele istorice și antetul de index fără nume; s-a adăugat `data_source`. Indexul a fost regenerat, deci nu este identificator de meci.
10. Datele au fost sortate determinist. Valorile tuturor coloanelor istorice ale celor 4.560 de observații vechi au fost comparate cu originalul și au rămas neschimbate, independent de ordinea rândurilor și de reprezentarea numerică CSV.
11. Fișierul îmbinat a fost validat într-un fișier intermediar; abia apoi a înlocuit atomic `matches.csv`.

## 6. Statistici lipsă — fără valori fabricate

Pentru cele **960 de rânduri noi**:

| Câmp | Rânduri fără valoare | Explicație |
|---|---:|---|
| `gf`, `ga` | 0 | Scoruri finale disponibile |
| `sh`, `sot` | 0 | Șuturi și șuturi pe poartă disponibile |
| `poss` | 960 | Posesia nu este furnizată de aceste fișiere |
| `xg`, `xga` | 760 fiecare | Lipsesc în 2025–2026; există pentru cele 200 de perspective din 2026–2027 |
| `dist`, `fk`, `pk`, `pkatt` | 960 fiecare | Câmpuri indisponibile în sursa utilizată |

Au rămas nule și celelalte metadate nesupuse recuperării, precum căpitan, formație sau participare, dacă nu existau în sursă. Cotele din fișierele brute nu au fost introduse ca predictori în aplicație.

Lipsa posesiei poate afecta comparabilitatea predictorilor complecși. Prelucrarea existentă folosește numai observațiile istorice disponibile și priorul fix documentat când fereastra recentă nu conține observații. Valorile de posesie nu au fost inventate în CSV. Modelul actual nu utilizează xG ca predictor; păstrarea noilor valori nu înseamnă că acesta a fost extins automat să le folosească.

## 7. Conținutul final al matches.csv

| Sezon | Valoarea `season` | Rânduri de echipă | Meciuri distincte |
|---|---:|---:|---:|
| 2019–2020 | 2020 | 760 | 380 |
| 2020–2021 | 2021 | 760 | 380 |
| 2021–2022 | 2022 | 760 | 380 |
| 2022–2023 | 2023 | 760 | 380 |
| 2023–2024 | 2024 | 760 | 380 |
| 2024–2025 | 2025 | 760 | 380 |
| 2025–2026 | 2026 | 760 | 380 |
| 2026–2027, parțial | 2027 | 200 | 100 |
| **Total** | | **5.520** | **2.760** |

Au fost adăugate **480 de meciuri**, adică **960 de perspective de echipă**. Nu există duplicate pe cheia de mai sus. `load_matches()` a recitit și validat fișierul final, inclusiv consistența scorurilor reciproce. Intervalul final: **9 august 2019 – 8 noiembrie 2026**.

SHA-256 al `matches.csv` după actualizare:

```text
e5956c0636b35315346f721e1aa2688c576c089fb6ec7f4276f211b86f5f3a47
```

Manifestul final verificabil: `data/update_manifest.json`.

## 8. Teste și reantrenare executate

- **71 de teste trecute**, inclusiv testele de achiziție/conversie și GTK sub Xvfb/X11.
- Au fost observate avertismente neblocante privind acceleratoarele pandas opționale și API-uri depreciate pandas/GTK. Nu au fost mascate drept rezultate de succes ale unor operații eșuate.
- S-a rulat efectiv:

```bash
./run.sh --evaluate --output reports/soccerdata_update
```

Împărțirea cronologică nouă, pe zile întregi:

| Set | Meciuri | Interval |
|---|---:|---|
| Antrenare | 1.611 | 2019-08-09 – 2023-10-27 |
| Validare | 587 | 2023-10-28 – 2025-04-02 |
| Test | 562 | 2025-04-03 – 2026-11-08 |

| Model | Acuratețe test | Macro-F1 | Log-loss ↓ |
|---|---:|---:|---:|
| KNN | 42,88% | 0,3249 | 1,0972 |
| Random Forest | 48,93% | 0,3603 | 1,0362 |
| Poisson | 45,73% | 0,3437 | 1,0597 |
| Reper majoritar | 42,53% | 0,1989 | 1,0836 |

Cele trei ajustări numerice Poisson au raportat convergență. Rezultatele nu sunt direct comparabile cu vechea evaluare: setul de test și intervalul temporal sunt diferite. Mai multe date nu garantează procente mai mari și nu transformă predicțiile în certitudini.

Rapoartele noi sunt în `reports/soccerdata_update/`. Rapoartele vechi din `reports/` și `docs/RESULTS.md` descriu experimentul anterior, pe datele până în mai 2025; nu au fost prezentate drept rezultatele noului set.

## 9. Pornire, reproducere și limite

Aplicația folosește implicit chiar `matches.csv`, nu un fișier intermediar:

```bash
./run.sh
```

Dacă interfața era deschisă înaintea actualizării, trebuie repornită sau reimportat `matches.csv`, apoi apăsat „Antrenează modele”. Modelele nu sunt persistate automat între porniri.

Reconstituirea într-un fișier separat din backup și cache, fără cereri online:

```bash
.venv/bin/python -m football_predictor.acquisition \
  --input data/backups/matches.before_soccerdata_2026-11-19.csv \
  --output matches_updated.csv \
  --cache-dir data/raw/matchhistory \
  --seasons 2025-2026 2026-2027
```

Modulul de conversie este intenționat **offline**: citește cache-ul real prin soccerdata, iar `--refresh` este respins explicit. Nu este un buton de actualizare online a datelor viitoare. Reconstituirea reproduce observațiile; eticheta `existing_local_extraction` pentru rândurile vechi a fost completată în publicarea finală descrisă mai sus.

Fișierele brute, backupurile, manifestele generate și CSV-urile locale sunt ignorate de Git pentru a evita publicarea accidentală înaintea verificării drepturilor de redistribuire. Configurația de teste GitHub instalează și extra-ul `scraping` necesar testelor soccerdata; nu s-a executat un workflow pe GitHub și nu s-a publicat nimic online.

## 10. Verificarea după reluarea lucrului

La reluare s-au recitit `matches.csv`, documentația și raportul. Hashul CSV a rămas identic cu cel din secțiunea 7; datele nu au fost pierdute sau modificate.

S-a executat suplimentar fluxul complet cu interfața GTK nativă și serviciul real, pe datele actualizate:

1. Încărcarea celor **2.760 de meciuri** și verificarea avertismentului de actualitate **2026-11-08**.
2. Antrenare prin butonul interfeței, în worker, fără blocarea firului GTK.
3. Simulare arbitrară Arsenal–Chelsea pentru **2026-11-20**. Aceasta verifică aplicația, **nu afirmă existența unui meci programat la această dată**.
4. Verificarea celor trei rânduri de probabilități și a matricei de scor.
5. Export și recitire a raportului: **562 de meciuri de test**, respectiv **2.248 de rânduri de predicție**, câte unul pentru fiecare dintre cele patru modele/reper.
6. Reverificarea hashului pentru a demonstra că antrenarea și exportul nu au modificat CSV-ul.
7. Reluarea suitei: **71 de teste trecute**; avertismentele neblocante sunt cele deja documentate.
8. Construirea unui wheel nou și inspectarea arhivei: stylesheet-ul GTK, modulul de achiziție și serviciul sunt incluse.

Dovezi salvate:

- `reports/soccerdata_update/gtk_verification.json` — verificările și numerele reconciliate;
- `reports/soccerdata_update/gtk_updated_prediction.png` — captură a ferestrei reale, inspectată vizual;
- rapoartele CSV/JSON din același director, regenerate prin interfață.

Interfața a fost închisă la încheierea verificării automate. Nu s-a lăsat pornită o fereastră ascunsă și nu s-au făcut publicări pe GitHub.

## 11. Etapa de optimizare — protocol stabilit înaintea experimentului

La solicitarea de continuare, utilizatorul a ales explicit îmbunătățirea modelelor, evaluarea pe sezoane și analiza erorilor. **Etapa este finalizată și verificată pe date reale; rezultatele sunt în secțiunea 12.** CSV-ul nu a fost modificat de experimente.

Protocolul este salvat și în `reports/optimization/protocol.json`:

- comparație între configurația anterioară (`legacy`) și o căutare de hiperparametri extinsă (`optimized`);
- sezoane de test cu ani finali 2023, 2024, 2025, 2026 și 2027, ultimul incomplet;
- pentru fiecare sezon testat: antrenare pe sezoanele mai vechi, validare pe sezonul imediat anterior, apoi reajustare pe întregul trecut;
- hiperparametrii sunt aleși după log-loss de validare, nu după scorul pe sezonul de test;
- ambele variante sunt evaluate pe aceleași meciuri; un rezultat mai slab este raportat, nu ascuns;
- analiza erorilor va separa clasele, echipele și nivelurile de încredere, pentru a evidenția inclusiv egalurile ratate;
- aceste date au fost deja inspectate în etapele precedente: este un **benchmark retrospectiv**, nu un test viitor complet neatins. Orice concluzie de îmbunătățire rămâne de verificat pe rezultate viitoare.

Inițial s-a adăugat ruta CLI `--benchmark`, verificată prin teste de rutare. Ulterior au fost integrate modulele de selecție și diagnostic și s-a executat benchmarkul real de mai jos.

## 12. Rezultatele optimizării și analiza erorilor

### Execuție și integritate

Comanda executată efectiv:

```bash
./run.sh --benchmark --output reports/optimization
```

S-au verificat **5 sezoane de test**, corespunzătoare anilor finali 2023–2027: patru sezoane complete cu câte 380 de meciuri și sezonul parțial 2026–2027 cu 100. Totalul este **1.620 de meciuri distincte de test** pentru fiecare model și variantă. Nu trebuie confundat cu totalul de 2.760 de meciuri din întregul istoric.

Exporturile au fost recitite și reconciliate programatic:

- **40 de rânduri de metrici**: 5 sezoane × 2 variante × 4 modele/reper;
- **12.960 de rânduri de predicție**: 1.620 de meciuri × 2 variante × 4 modele/reper;
- **320 de rânduri de selecție** în tabelul de tuning; acestea includ și subseturile legacy reutilizate, nu reprezintă 320 de ajustări distincte;
- fără duplicate de meci în interiorul unui grup variantă/model;
- toate ajustările Poisson de selecție și evaluare au raportat convergență;
- hashul sursei este identic cu cel din secțiunea 7.

Configurațiile extinse au 24 de candidați KNN, 18 Random Forest și 9 Poisson pe fiecare fold real. Configurațiile legacy sunt incluse integral și reutilizează scorurile de validare, fără antrenări duplicate pentru aceiași candidați.

### Rezultate agregate pe cele 1.620 de meciuri

Metricile sunt calculate din toate predicțiile reunite, ponderate implicit după numărul de meciuri, nu prin media neponderată a procentelor sezoanelor. Pentru fiecare sezon există un model reajustat separat, pe trecutul acelui sezon.

| Variantă | Model | Acuratețe | Macro-F1 | Log-loss ↓ | Recall egaluri |
|---|---|---:|---:|---:|---:|
| legacy | Reper majoritar | 44,32% | 0,2047 | 1,0701 | 0,00% |
| legacy | KNN | 45,19% | 0,3386 | 1,0763 | 2,29% |
| legacy | Poisson | 51,05% | 0,3779 | 1,0129 | 0,00% |
| legacy | Random Forest | 53,27% | 0,3976 | 1,0001 | 0,51% |
| optimized | Reper majoritar | 44,32% | 0,2047 | 1,0701 | 0,00% |
| optimized | KNN | 47,04% | 0,3375 | 1,0468 | 0,25% |
| optimized | Poisson | 51,30% | 0,3799 | 1,0082 | 0,00% |
| optimized | Random Forest | 52,16% | 0,3935 | 1,0022 | 1,27% |

Interpretare:

- KNN câștigă aproximativ **1,85 puncte procentuale** la acuratețe și are log-loss mai bun, dar Macro-F1 este ușor mai slab și identifică mai puține egaluri. Nu este o îmbunătățire pe toate criteriile.
- Poisson câștigă aproximativ **0,25 puncte procentuale** la acuratețe, cu log-loss și Macro-F1 ușor mai bune. Diferența este mică și nu este însoțită de un test de semnificație statistică.
- Random Forest pierde aproximativ **1,11 puncte procentuale** după extinderea căutării și are log-loss ușor mai slab. Căutarea mai largă pe validare nu garantează un sezon de test mai bun.
- Random Forest legacy rămâne cel mai bun la acuratețe și log-loss în această comparație agregată. **Nu s-a înlocuit automat configurația desktop cu varianta optimized.** Serviciul aplicației păstrează antrenarea existentă; căutarea extinsă este disponibilă prin modulul de selecție și benchmarkul reproductibil.

### Analiza erorilor și limite constatate

În meciurile evaluate au existat **393 de egaluri reale**. Modelele aproape nu aleg egalul ca rezultat final: recall-ul maxim din tabel este numai 2,29%. Poisson nu a ales niciun egal prin argmax. Aceasta nu înseamnă că probabilitatea calculată pentru egal este zero, ci că ea nu a fost cea mai mare dintre cele trei în aceste cazuri.

Pentru KNN optimized, precizia clasei egal este 100%, dar provine din numai **un egal identificat** din 393: o precizie mare pe un suport de predicții foarte mic nu demonstrează un model bun. Din acest motiv se raportează împreună suportul, precizia, recall-ul și F1.

Diagnosticul include:

- indicatori separați pentru victorie gazde, egal și victorie oaspeți;
- erorile ordonate după încrederea maximă, pentru inspectarea eșecurilor prea încrezătoare;
- distribuții pe echipe și pe sezoane;
- calibrare top-label în zece intervale și ECE. ECE mic nu demonstrează calibrarea tuturor celor trei probabilități; reperul constant poate avea ECE mic fără putere predictivă individuală.

Benchmarkul rămâne retrospectiv, cu etichete istorice deja inspectate. Datele noi provin din alt furnizor și nu conțin posesie; aceasta este o posibilă sursă de schimbare a distribuției, dar nu a fost demonstrată drept cauza unei scăderi de performanță. Nu se aleg noi hiperparametri pe baza acestor scoruri de test și nu se ascund sezoanele cu rezultate mai slabe.

### Module și fișiere livrate

Module noi:

- `src/football_predictor/selection.py`: selecție deterministă numai pe validare;
- `src/football_predictor/experiments.py`: folduri cronologice pe sezoane și comparație paired;
- `src/football_predictor/diagnostics.py`: validarea predicțiilor și analiza erorilor.

În `reports/optimization/`:

- `metrics_by_season.csv`, `predictions_by_season.csv`, `tuning_by_season.csv`;
- `benchmark.json`: protocol, folduri, selecții, matrici de confuzie și convergență;
- `aggregate_metrics.csv`, `verified_summary.json`;
- `diagnostics/per_class.csv`, `calibration.csv`, `confident_errors.csv`, `errors_by_team.csv`, `errors_by_season.csv`, `summary.csv`, `diagnostics.json`.

Verificare finală: **129 de teste trecute**, inclusiv GTK sub Xvfb. Persistă avertismentele neblocante deja documentate. S-a reconstruit wheel-ul și s-a verificat că include toate cele trei module noi și stylesheet-ul GTK. Nu au fost modificate datele și nu s-a publicat nimic pe GitHub.

## 13. Instalare ușoară pe alt calculator — finalizată local

După publicarea manuală de către utilizator, s-a verificat că repository-ul `cristian292/football-predict` este privat și conține codul și `matches.csv`. Copia de lucru activă pentru această etapă este `~/Documents/football_predict_github`, nu vechiul director din repository-ul părinte pentru hotel.

S-a preluat prin fast-forward commitul utilizatorului `a7e289c` (modificări README), apoi s-a creat branchul local `feat/easy-linux-install`. Modificările README ale utilizatorului au fost păstrate. Workflow-ul GitHub al acelui commit, run `44772971211`, a fost verificat ca `completed/success`; acest rezultat nu acoperă schimbările locale ulterioare.

Obiectivul etapei: instalare ghidată pe Linux Debian/Ubuntu/Parrot și verificare într-o copie curată, fără mediul virtual, cache-urile sau pachetele Python personale ale autorului. Nu se schimbă modelele ori datele și nu se promite un pachet Windows/macOS.

Modificări implementate:

- `install.sh`: verificarea prerechizitelor și instalarea într-un mediu virtual local, fără apt/sudo automat;
- modul `doctor.py` și ruta CLI `--doctor`: verificarea dependențelor, GTK4 și antetului CSV, fără antrenare sau scraping;
- `run.sh`: ignorarea pachetelor personale `~/.local` prin `PYTHONNOUSERSITE=1`;
- README și instrucțiunile GitHub adaptate unui utilizator care descarcă repository-ul privat;
- workflow-ul CI folosește același instalator, nu o cale de instalare separată.

### Verificarea efectivă

1. `install.sh --check-only` a reușit înainte să existe `.venv`, fără să creeze mediul sau să acceseze surse sportive.
2. S-a construit o copie nouă din fișierele proiectului, fără `.git`, `.venv`, `__pycache__`, rapoarte, cache brut sau backupuri; s-a folosit și un director HOME separat, cu `PYTHONNOUSERSITE=1` și fără PYTHONPATH personal.
3. S-a executat instalatorul normal cu pip real, nu un serviciu simulat. A creat mediul virtual și a instalat aplicația plus dependențele necesare. `--system-site-packages` permite folosirea bibliotecilor distribuției; aceasta este o instalare nouă a proiectului, **nu o mașină virtuală cu un sistem de operare proaspăt**.
4. `run.sh --doctor` a reușit în copia nouă. A fost verificat și un CSV inexistent: mesaj clar și cod de ieșire 1, fără traceback.
5. A rulat fluxul GTK real pe instalarea normală, fără soccerdata: **2.760 de meciuri → antrenare → simulare Arsenal–Chelsea → export**. Raportul are 562 de meciuri de test și 2.248 de rânduri de predicție pentru cele patru modele/reper.
6. Importurile științifice au fost verificate: niciunul nu provine din `~/.local` sau din vechiul mediu `~/Documents/py/gambling/.venv`. Versiunile din această verificare: Python 3.13.5, NumPy 2.2.4, pandas 2.2.3, SciPy 1.15.3, scikit-learn 1.9.1, GTK 4.18.6. Bibliotecile NumPy/pandas/SciPy provin din pachetele distribuției; scikit-learn din noua `.venv`.
7. `install.sh --dev` a fost executat și în copia Git activă. Suita completă a trecut: **174 de teste, fără teste omise**, cu 3 avertismente de depreciere GTK.
8. S-a construit wheel-ul și s-au verificat `doctor.py` și resursa CSS GTK în arhivă. `bash -n` și `git diff --check` au reușit.
9. Hashul `matches.csv` a rămas cel din secțiunea 7. Nu au fost antrenate sau promovate alte configurații de modele în această etapă; simularea doar exercită antrenarea existentă.

### Avertismente și limite

Instalarea opțională `--dev` pe Parrot a afișat și conflicte pip pentru pachete ale altor aplicații, vizibile prin `--system-site-packages`. Instalarea s-a încheiat cu exit 0, importurile aplicației și toate testele au reușit. Nu s-au dezinstalat ori modificat pachetele distribuției: noile pachete au fost instalate în `.venv`. Nu se pretinde că întregul set de pachete al distribuției trece `pip check`.

Versiunile dependențelor sunt încă intervale din `pyproject.toml`, nu un lockfile complet pe platforme. Rezultatele numerice pot varia față de experimentele mai vechi cu alte versiuni. Windows/macOS, pachetele instalabile independente și testarea pe un calculator fizic diferit rămân în afara verificării locale actuale.

Modificările sunt pe branchul local `feat/easy-linux-install`; **nu s-a executat commit, push sau merge** pentru această etapă. Workflow-ul modificat este verificat local prin aceleași comenzi de instalare/doctor/teste, dar nu a fost încă rulat de GitHub pentru acest branch. Pentru distribuire, trebuie publicat și verificat în Actions.

### Comenzi pentru utilizator

După instalarea dependențelor native descrise în README:

```bash
bash install.sh
bash run.sh --doctor
bash run.sh
```

Instalatorul poate fi rulat din alt director și tratează căile cu spații. Păstrează mediile existente și refuză suprascrierea unei `.venv` care nu corespunde verificărilor. Pentru testare/dezvoltare se folosește `bash install.sh --dev`.

## 14. Corectarea direcției: distribuire prin Docker

Utilizatorul a precizat explicit că dorește **Docker**, nu instalarea Python pe gazdă. Calea principală din README a fost schimbată în consecință. Lucrările anterioare de instalare directă nu au fost șterse: sunt păstrate ca alternativă și ca istoric verificabil. Nu sunt prezentate drept soluția Docker.

Branchul local a fost redenumit din `feat/easy-linux-install` în `feat/docker-distribution`, fără pierderea schimbărilor și fără commit/push. Interfața rămâne GTK4 nativă.

### Descoperire și probă inițială

- Pe gazdă, `/bin/docker` este wrapper pentru **Podman 5.4.2**, nu Docker Engine.
- `podman info` a confirmat rularea rootless și un storage local funcțional.
- Providerul Compose disponibil raportează Docker Compose 2.26.1.
- Sesiunea este Wayland; există și DISPLAY/XAUTHORITY pentru X11. Socketul Wayland verificat este `/run/user/1000/wayland-0`.
- `podman run --rm docker.io/library/ubuntu:24.04 /bin/true` a descărcat imaginea de bază și a reușit cu exit 0. Aceasta dovedește funcționarea runtime-ului, **nu încă a imaginii aplicației**.
- Nu s-a instalat Docker Engine, nu s-au modificat grupurile utilizatorului sau permisiunile sesiunii grafice și nu s-au folosit privilegii suplimentare.

### Implementare în curs

Imagine runtime și target de teste, context `.dockerignore` restrâns, Compose pentru moduri fără display și un wrapper pentru GUI Wayland/X11 cu autentificare. Datele și codul vor fi copiate în imagine; rapoartele vor persista prin bind mount. Nu se montează întregul HOME sau socketul motorului Docker și nu se folosește `xhost +`.

README, `docs/DOCKER.md` și instrucțiunile GitHub au fost adaptate. Workflow-ul CI include un job pentru construirea targetului de teste și rularea lui fără rețea, cu rootfs read-only. Acest workflow nu a fost publicat sau executat încă pentru noul branch.

**Etapa nu este încă declarată finalizată:** trebuie construite și rulate imaginile reale, verificată evaluarea/doctor în container, exercitate testele interne și verificat accesul GTK nativ disponibil pe gazdă. Docker Engine, Podman și Xvfb vor fi raportate distinct; succesul unuia nu este prezentat drept verificarea celorlalte.

## 15. Migrarea solicitată la aplicație web reală

Utilizatorul a renunțat explicit la fereastra GTK/GDK și la afișarea ei prin noVNC. Solicitarea curentă este implementarea unui site web, iar imaginea Docker va fi creată **ulterior, împreună cu utilizatorul**, după verificarea aplicației. În această etapă nu se execută build, run sau push de imagini Docker.

Copia de lucru activă este `~/Documents/football_predict_github`, pe branchul local `feat/web-application`, cu modificările anterioare nesuprascrise. La reluare au fost găsite fișiere web parțiale, inspectate înainte de continuare. Rutele și interacțiunile lipsă sunt completate separat, prin teste incremental RED→GREEN; nu se prezintă o interfață statică drept aplicație funcțională.

### Arhitectură

- Flask pentru API și resurse locale HTML/CSS/JavaScript, servite cu Waitress;
- același `ProjectService` pentru KNN, Random Forest, Poisson, evaluare și export;
- operații grele serializate într-un worker, cu progres și rezultate accesibile prin API;
- spațiu de lucru local unic, fără conturi sau separare între utilizatori; nu este un serviciu public multi-user;
- pornirea implicită și diagnosticul nu mai importă GTK sau biblioteci de display;
- importurile și exporturile web sunt separate de `matches.csv` și ignorate de Git.

### Verificări deja executate în această etapă

- mediul web de bază, instalatorul, CLI și diagnosticul au trecut testele focalizate;
- s-a creat și instalat efectiv un mediu virtual complet separat, fără system-site-packages și cu HOME separat;
- în acel mediu `gi` nu este instalat, importul aplicației web nu încarcă GTK, iar `--doctor` raportează OK;
- cele **38 de teste ale datelor, caracteristicilor, modelelor și serviciului** au trecut în mediul fără GTK;
- Flask 3.1.3 și Waitress 3.0.2 au fost instalate în mediul curat, alături de dependențele științifice;
- versiunea pachetului a fost sincronizată la `0.2.0` și declararea resurselor web este testată;
- închiderea workspace-ului la ieșirea serverului a fost verificată inclusiv pentru întreruperi și excepții;
- hashul CSV este în continuare `e5956c0636b35315346f721e1aa2688c576c089fb6ec7f4276f211b86f5f3a47`, iar modulele algoritmice nu au fost modificate.

README și instrucțiunile de colaborare descriu acum serverul web. CI a fost adaptat pentru teste Python și DOM JavaScript fără GTK; noul workflow nu a fost publicat sau verificat pe GitHub. Configurațiile vechi Docker/GTK sunt păstrate ca istoric și nu trebuie folosite drept imagine web verificată.

**Migrarea web este finalizată și verificată pe Linux.** În mediul izolat fără GTK, suita activă a trecut cu **224 de teste reușite și 2 omise**, cele omise fiind integrări opționale `soccerdata`, neinstalat în acel mediu. Testele vechi GTK și cele ale schiței Docker au fost păstrate în `legacy-tests/`, separat de colectarea implicită; cele patru eșecuri X11 cunoscute ale vechiului launcher nu sunt declarate remediate.

### Verificarea finală executată la 20 noiembrie 2026

- Serverul web a fost pornit efectiv cu Waitress, pe `http://127.0.0.1:8080`, într-un mediu fără `gi`.
- Au trecut **11 verificări de interacțiune și stare în browser**, **18 verificări HTTP reale** și **7 verificări Chromium headless**, inclusiv navigare de tastatură.
- Au fost antrenate efectiv cele trei modele pe cele 2.760 de meciuri; s-au verificat refreshul în timpul antrenării și afișarea raportului final.
- Simularea Arsenal–Chelsea, 2026-11-21, a produs rezultate reale ale algoritmilor; nu reprezintă confirmarea unui meci programat. Probabilitățile brute însumează 1 pe model, iar afișarea rotunjită corespunde API-ului.
- Clasamentul sezonier, istoricul echipei, cele patru pagini și validarea formularului au fost exercitate în browser.
- Un CSV invalid a păstrat istoricul și modelele active; un CSV valid a invalidat rezultatele și modelele; resetarea și reantrenarea au funcționat.
- Au fost descărcate efectiv `metrics.csv`, `tuning.csv`, `predictions_test.csv`, `report.json` și `report.zip`; conținutul arhivei coincide cu descărcările individuale. Predicțiile test au 2.248 de rânduri, pentru 562 de meciuri și patru modele/reper, nu 2.248 de meciuri distincte.
- Pe desktop 1280×1000 și mobil 390×844 nu s-a găsit overflow al paginii în cele patru secțiuni; tabelele late folosesc derulare internă.
- QA a găsit o cerere 404 pentru iconița absentă; a fost adăugat un SVG local și un test de regresie RED→GREEN. Rularea finală Chromium nu are erori JavaScript sau de consolă.
- A fost construit și instalat separat wheel-ul `football_predictor_web-0.2.0-py3-none-any.whl`; resursele HTML/CSS/JS/SVG și API-ul de stare sunt funcționale din pachetul instalat, fără GTK și fără a importa codul din copia editable.
- `bash run.sh --doctor`, verificarea sintaxei JavaScript și `git diff --check` au trecut. Fișierul original și modulele algoritmice au rămas neschimbate.

Raportul local detaliat, capturile reale, logurile și dovezile structurate sunt în `reports/web-qa/report.md` și `reports/web-qa/manifest.json`; directorul de rapoarte este generat local și ignorat de Git. Unele capturi CDP interactive au expirat; verificarea vizuală a fost executată ulterior cu Chromium headless/Playwright, nu înlocuită cu imagini fabricate. Selenium-ul sistemului nu avea driverul necesar și nu a fost raportat ca verificare reușită.

**Limite păstrate explicit:** spațiu de lucru local unic, fără conturi; starea modelelor se pierde la repornire; URL-urile de export expiră la repornire sau după 100 de exporturi mai noi. Nu s-a testat fizic pe Windows/macOS și nu s-a publicat noul workflow pe GitHub. Nu s-a făcut commit/push și **nu s-a construit nicio imagine Docker**. Următoarea etapă este construirea imaginii web împreună cu utilizatorul, în modul de învățare solicitat.

## 16. Imaginea web construită ulterior de utilizator

După finalizarea interfeței, utilizatorul a construit local imaginea `localhost/football-predict:web`, folosind noul Dockerfile bazat pe Python 3.12 slim, fără GTK și cu utilizatorul `appuser`. Comanda `docker` de pe această gazdă este oferită de Podman 5.4.2. La proba de pornire, endpointul `/healthz` din container a răspuns efectiv HTTP 200.

O eroare de acces a fost identificată prin inspectarea mapării: comanda introdusă inversase porturile (`127.0.0.1:8080:8081`), deși aplicația asculta pe 8080 în container. Maparea corectă este `127.0.0.1:8081:8080`; utilizatorul a confirmat ulterior că funcționează. Imaginea nu a fost publicată într-un registry. Această etapă nu verifică automat viitoarele modificări ale algoritmilor din copia sursă: o imagine existentă trebuie reconstruită pentru a le include.

## 17. Campanie extinsă de îmbunătățire a acurateței — în lucru

La 24 noiembrie 2026, utilizatorul a autorizat o campanie amplă pentru KNN, Random Forest și Poisson. Nu există promisiunea că fiecare familie va obține un scor mai mare. Versiunile actuale rămân reper; experimentele sunt implementate în module separate și nu suprascriu datele sau configurațiile aplicației înainte de măsurare.

### Starea inițială verificată

- Copie activă: `~/Documents/football_predict_github`, branch `feat/web-application`, cu modificările locale anterioare păstrate.
- Date: aceleași 5.520 de perspective / 2.760 de meciuri; SHA-256 `e5956c0636b35315346f721e1aa2688c576c089fb6ec7f4276f211b86f5f3a47`.
- Audit și fingerprinturi: `reports/accuracy_campaign_20261124_174323/initial_snapshot.json`.
- Setul test retrospectiv rămâne cel al aplicației: 562 meciuri, 2025-04-03–2026-11-08. Datele până la 2025-04-02 sunt rezervate dezvoltării și selecției.
- Serviciul web folosește căutarea inițială de 8 configurații KNN și 4 Random Forest, plus Poisson independent. Căutările mai largi din `selection.py` aparțin benchmarkului anterior, nu au fost activate automat în serviciul web.
- Coloanele brute xG/xGA există, dar au aproximativ 96,38% valori lipsă. Nu sunt tratate ca o sursă completă și nu se inventează statistici pentru a promite performanță.

### Protocolul campaniei

Selecția folosește trei folduri cronologice expansive numai în perioada de dezvoltare, fără împărțirea aceleiași date între antrenare și validare. Obiectivul principal este acuratețea cumulată de validare; Macro-F1 și log-loss departajează. Se vor raporta și Brier, performanța pe fiecare clasă și intervale bootstrap paired pe blocuri de date, nu doar cel mai favorabil procent.

Sunt în curs de implementare, cu teste RED→GREEN:

1. caracteristici strict pre-meci: ferestre multiple, formă acasă/deplasare, recență și forță Elo;
2. căutare mai largă pentru KNN și Random Forest, inclusiv familii diferite de caracteristici și opțiuni de recență;
3. model Poisson/Dixon–Coles cu forțe pe echipe, ponderare temporală și corecția scorurilor mici;
4. căutare de hiperparametri Poisson și comparație cu ratele istorice existente;
5. un mic set fix de ajustări ale probabilităților pe clase, acceptate numai dacă sporesc acuratețea de validare și nu degradează log-loss cu mai mult de 0,05. Acestea nu sunt prezentate ca o calibrare garantată.

Grilele concrete și amprenta protocolului vor fi înghețate înainte de căutarea amplă. Alegerile finale vor fi salvate înainte de evaluarea pe test. Etichetele istorice au fost văzute în etape anterioare; comparația este retrospectivă, nu o dovadă prospectivă neatinsă. Nu se aleg noi configurații după rezultatul testului pentru a forța o creștere.

### Componente și prima rulare reală

Componentele au fost implementate separat: `advanced_features.py`, `advanced_selection.py`, `models/dixon_coles.py` și `advanced_poisson_selection.py`. Grila inițială efectivă are **155 configurații KNN, 79 Random Forest și 44 Poisson/Dixon–Coles**. Caracteristicile includ familiile legacy, formă multi-fereastră, forță și setul extins; niciuna nu include scorul partidei curente.

Verificarea părintelui, înaintea adaptorului de integrare, a trecut cu **391 de teste**. Suita de reper neschimbată a trecut separat cu 226 de teste. Există teste de invarianta temporală, gradient numeric/analitic Dixon–Coles, normalizare, date extreme, parametri invalizi, folduri și exporturi.

Prima căutare reală a fost pornită la 18:11:42 pe 24 noiembrie 2026, cu trei workeri și BLAS/OpenMP limitate la câte un fir. Protocolul complet este `reports/accuracy_campaign_20261124_174323/protocol.json`, SHA-256 `95ad12aec629f17ac5d99fd44da129d1f37271430271040c72c207a0962a0591`. Logul este `campaign.log`; selecțiile se salvează separat, fără a cere reluarea tuturor modelelor dacă o etapă se întrerupe.

Înainte ca prima rulare să ajungă la evaluarea testului, la 18:17:01, au fost înregistrate suplimentar reguli fixe de rafinare în `refinement_rules.json`: vecini KNN suplimentari și netezire, păduri de 300/600 arbori și frunze alternative, respectiv valori intermediare de recență/regularizare Poisson. Lista de candidați rafinați va fi derivată numai din clasamentul de validare al primei runde. Raportul/metricile/predicțiile de test ale primei runde nu sunt intrări permise pentru acest pas. Varianta finală rafinată se alege înaintea citirii rezultatelor sale de test, nu prin alegerea celui mai favorabil test dintre runde.

### Recuperarea integrării și starea primei runde

Prima campanie s-a încheiat la **20:20:20, 24 noiembrie 2026**, conform istoricului din `status.json`. Rezultatele de test ale acestei runde nu au fost consultate pentru derivarea candidaților rafinați; selecția rafinată folosește numai regulile înregistrate și rezultatele validării.

Delegarea adaptorului de integrare a fost întreruptă, dar fișierele existente au fost verificate de procesul părinte: **23 de teste ale adaptorului au trecut**, inclusiv antrenare, predicție, export și joburi web reale pe date sintetice. Au fost completate prin RED→GREEN fabrica `service_factory.py` și opțiunea explicită `--model-config` pentru serverul web și `--evaluate`.

Fără rețetă explicită se păstrează exact `ProjectService`. O rețetă validă se folosește numai pentru CSV-ul cu hash identic; un upload cu alte date revine la serviciul original, iar resetarea poate restabili rețeta pentru sursa originală. Configurațiile existente invalide sunt respinse, nu ignorate. Nu se generează rețete din metricile testului.

Suita integrării a trecut cu **423 de teste** în 92,07 secunde (`integration-recovery-suite.log`), excluzând explicit `tests/test_accuracy_refinement.py`, aflat atunci în completare separată. Cele șase teste existente ale rafinării trecuseră separat înainte de adăugarea verificărilor de proveniență. CSV-ul păstrează SHA-256 `e5956c0636b35315346f721e1aa2688c576c089fb6ec7f4276f211b86f5f3a47`; `git diff --check` nu a raportat erori.

### Rezultat final verificat — acuratețe 1/X/2

După prima grilă (278 configurații), runda de rafinare a evaluat 16 variante KNN, 13 Random Forest și 15 Poisson, incluzând câștigătorii inițiali. Comparația de mai jos folosește aceleași 562 de meciuri din 3 aprilie 2025–8 noiembrie 2026. Am recalculat acuratețea din predicțiile individuale și am verificat identitatea meciurilor, etichetele, normalizarea probabilităților și absența duplicatelor.

| Model | Inițial | Selectat pe validare | Diferență în puncte procentuale (interval aproximativ 95%) |
|---|---:|---:|---:|
| KNN | 241/562 (42,88%) | 280/562 (49,82%) | +6,94 [+3,60; +10,51] |
| Random Forest | 275/562 (48,93%) | 266/562 (47,33%) | −1,60 [−4,15; +1,20] |
| Poisson | 257/562 (45,73%) | 258/562 (45,91%) | +0,18 [−0,88; +1,27] |

Numai KNN are un interval al diferenței care nu include zero. Folosește indicatorii de forță și 101 vecini cu ponderi egale. Rezultatele sunt retrospective, nu o garanție pentru meciurile viitoare. Configurația implicită rămâne neschimbată.

Rețeta opțională, generată numai din configurațiile înghețate de validare: `reports/accuracy_campaign_20261124_174323/refined/recipe.json`. S-a corectat în adaptor ordonarea meciurilor din aceeași zi pentru a reproduce ordinea experimentului, fără schimbarea candidaților. Serviciul reproduce toate cele 1.686 de predicții model–meci (diferență maximă de probabilitate sub 1e-15). Suita completă: **464 de teste trecute** (`refined/final-tests.log`).
