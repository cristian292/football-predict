# Rezultate experimentale verificate

Execuție locală pe `matches.csv`: 4.560 de rânduri sursă transformate în **2.280 de meciuri distincte**. Seed pentru clasificatoare: 42. Evaluarea este secvențială pre-meci; caracteristicile includ numai rezultate observate anterior datei fiecărui meci.

| Set | Meciuri | Prima dată | Ultima dată |
|---|---:|---|---|
| Antrenare | 1.338 | 2019-08-09 | 2023-01-22 |
| Validare | 449 | 2023-01-23 | 2024-03-03 |
| Test | 493 | 2024-03-04 | 2025-05-25 |

Proporțiile 60/20/20 se aplică zilelor distincte, nu exact numărului de meciuri. Hiperparametrii se selectează pe validare; pentru evaluarea finală se reajustează pe antrenare + validare.

| Model | Acuratețe | Macro-F1 | Log-loss ↓ | Brier multiclasă ↓ |
|---|---:|---:|---:|---:|
| KNN | 44,83% | 0,3308 | 1,0892 | 0,6507 |
| Random Forest | 53,75% | 0,4010 | 0,9943 | 0,5932 |
| Poisson | 52,94% | 0,3995 | 0,9962 | 0,5917 |
| Reper majoritar | 42,19% | 0,1978 | 1,0770 | 0,6522 |

Random Forest are cea mai bună acuratețe și cel mai mic log-loss în acest experiment; Poisson are Brier ușor mai mic. KNN depășește reperul în acuratețe, dar are log-loss mai slab. Macro-F1 modest arată că rezultatele nu sunt uniform bune pe toate clasele. Nu există un studiu de semnificație statistică sau o validare externă pe alte campionate; nu deducem superioritate universală.

Toate cele trei ajustări Poisson (selecție, evaluare și model final pentru viitor) au raportat convergență numerică.

## Verificări executate

- 46 de teste automate trecute, inclusiv 5 teste GTK sub Xvfb/X11.
- Flux GTK real, fără serviciu simulat: încărcare CSV → antrenare → predicție Arsenal–Chelsea → export și citirea raportului.
- Pachet wheel construit; `football_predictor/ui/style.css` este inclus.
- Exporturile locale sunt în `reports/`: `metrics.csv`, `tuning.csv`, `predictions_test.csv`, `report.json`. Sunt ignorate implicit de Git.
- Avertismente neblocante: acceleratoare pandas opționale din sistem mai vechi și deprecierea unor apeluri GTK de selecție a fișierelor. Configurația GTK locală poate emite și un avertisment despre `gtk-modules`.
- Workflow-ul GitHub Actions este adăugat ca configurație; **nu a fost executat pe GitHub**.

## Reproducere

```bash
./run.sh --evaluate --output reports
```

Rezultatele depind de versiunea datelor și a bibliotecilor. Valorile de mai sus provin din execuția reală locală, nu din rezultatele notebookului inițial. Datele se opresc la 2025-05-25: pentru estimări actuale este necesar un set de date actualizat.
