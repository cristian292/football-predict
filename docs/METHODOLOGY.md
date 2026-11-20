# Metodologie și corespondența cu obiectivele temei

## 1. Problema și unitatea observației

Se estimează rezultatul unui meci înainte de disputare, în trei clase: victorie gazde, egal, victorie oaspeți. Unitatea observației este **meciul**, nu rândul de echipă din fișierul extras. Perspectivele gazde/oaspeți sunt asociate pentru a evita numărarea de două ori și împărțirea aceluiași meci între seturi.

## 2. Caracteristici și prevenirea scurgerii de informație

Pentru data t sunt permise numai rezultate cu data strict mai mică decât t. Statisticile tuturor meciurilor din aceeași zi se actualizează după calcularea caracteristicilor acelei zile. Medii istorice pentru goluri înscrise/primite și forma echipelor constituie caracteristicile de bază. Statisticile istorice de șuturi, posesie, clasament și confruntări directe extind setul complex.

Clasamentul se calculează din rezultate, nu se introduce retrospectiv clasamentul final al sezonului. Pozițiile calculate sunt sportive, fără penalizări administrative și fără implementarea tuturor regulilor oficiale de departajare. Lipsa datelor xG nu este rezolvată prin inventarea unor valori xG.

## 3. K-Nearest Neighbors

Pentru vectorii standardizați se calculează distanțe către observațiile de antrenare și se combină rezultatele celor k vecini. Standardizarea se ajustează numai pe antrenare. Valoarea k și ponderarea se aleg pe validare. Probabilitățile provin din voturile vecinilor; nu se presupune că sunt perfect calibrate.

## 4. Random Forest

Ansamblu de arbori de clasificare construit pe caracteristicile istorice complexe. Configurațiile candidate sunt comparate pe validare. Modelul captează interacțiuni neliniare, dar o configurație mai complexă nu garantează acuratețe mai mare. Performanța se stabilește empiric pe setul de test păstrat separat.

## 5. Distribuția Poisson și optimizarea numerică

Pentru o intensitate λ pozitivă, probabilitatea a k goluri este:

`P(G=k) = exp(-λ) × λ^k / k!`.

Intensitățile sunt construite din istoricul golurilor înscrise și primite, cu regularizare/neteziere pentru istorice scurte. Parametrii numerici sunt estimați prin minimizarea negativului log-verosimilității. Rapoartele de execuție trebuie consultate pentru configurația efectivă și starea convergenței; existența unui rezultat numeric nu demonstrează automat convergența.

Ipoteza de bază a modelului este independența golurilor celor două echipe condiționat de intensități:

`P(H=i, A=j) = P(H=i) × P(A=j)`.

Suma sub diagonala matricei de scor reprezintă victoria gazdelor; diagonala reprezintă egalul, iar suma deasupra diagonalei victoria oaspeților. Distribuția Poisson are suport infinit: interfața afișează o matrice finită și cele mai probabile scoruri. Interpretarea trebuie să țină cont de masa reziduală/tăierea numerică raportată de implementare. Scorul cel mai probabil poate avea o probabilitate mică chiar când o echipă este favorită.

## 6. Evaluarea experimentală

Datele sunt împărțite cronologic în antrenare, validare și test, păstrând zilele întregi. Configurațiile se aleg fără acces la rezultatele testului. După alegere, modelele pot fi reajustate pe antrenare + validare pentru evaluarea testului. Un model final separat poate folosi toate datele pentru predicții după ultima dată disponibilă.

Protocolul testului este secvențial: pentru fiecare meci sunt disponibile rezultatele efectiv observate anterior. Nu se pretinde evaluarea unei prognoze făcute înaintea întregului interval de test. Un singur set de test nu demonstrează generalizarea în toate campionatele sau sezoanele.

Metrici:

- **Acuratețe:** proporția claselor prezise corect.
- **Macro-F1:** media F1 pe cele trei clase; penalizează ignorarea egalurilor.
- **Log-loss:** penalizează probabilitățile incorecte foarte încrezătoare; mai mic este mai bun.
- **Brier multiclasă:** eroarea pătratică a vectorului de probabilități, însumată pe clase și mediată pe meciuri; mai mic este mai bun.
- **Matrice de confuzie:** rânduri = clase reale, coloane = clase prezise, în ordinea gazde/egal/oaspeți.

Compararea cu un reper simplu este obligatorie. O acuratețe aparent bună nu este suficientă dacă modelul nu depășește reperul sau dă probabilități slabe.

## 7. Acoperirea obiectivelor

| Obiectiv | Componentă |
|---|---|
| Prelucrarea istoricului și clasament | `data.py`, `features.py` |
| KNN de bază | `models/`, serviciul de antrenare |
| Random Forest complex | `models/`, serviciul de antrenare |
| Poisson și scoruri exacte | `models/`, matricea și lista de scoruri din interfață |
| Algoritmi iterativi și minimizarea erorii | căutarea hiperparametrilor și optimizarea numerică Poisson |
| Tabele experimentale | pagina Experimente și exporturile CSV/JSON |
| Evaluare empirică | `evaluation.py`, set de test cronologic |
| Predicții clare | pagina Predicții din interfața GTK4 |

## 8. Limite

Datele locale nu includ informații actualizate despre accidentări, transferuri, loturi și cote. Nu este modelată explicit dependența scorurilor mici de tip Dixon–Coles. Nu sunt implementate pariuri sau tranzacții financiare. Nu există garanție asupra rezultatului unui meci sau asupra profitului. Interfața oferă estimări statistice pentru proiect și analiză, nu informație certă despre viitor.
