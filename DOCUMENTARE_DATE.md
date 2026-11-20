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
