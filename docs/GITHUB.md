# GitHub și colaborare

## Repository-ul proiectului

Proiectul este publicat în repository-ul **privat** `cristian292/football-predict`. Copia de lucru Git separată este `~/Documents/football_predict_github`; vechiul director `~/Documents/py/gambling` nu este repository-ul acestui proiect și nu trebuie folosit pentru push către repository-ul aplicației de hotel.

`matches.csv` este inclus intenționat în repository-ul privat. Regula `.gitignore` protejează alte copii accidentale ale datelor, dar nu elimină un fișier deja urmărit de Git. Nu transforma repository-ul în public fără verificarea condițiilor de redistribuire pentru date. Nu s-a ales încă o licență pentru cod în numele autorului.

## Descărcarea de către un coleg

1. Proprietarul acordă acces contului colegului din Settings → Collaborators.
2. Colegul acceptă invitația și se autentifică în GitHub.
3. Descarcă arhiva din Code → Download ZIP sau clonează repository-ul.
4. Urmează pornirea aplicației web din README: instalează pachetul Python, pornește serverul și deschide adresa locală în browser. Nu sunt necesare GTK, Wayland/X11 sau noVNC. Imaginea Docker pentru noul server va fi pregătită ulterior împreună cu utilizatorul; vechile scripturi de containerizare GTK nu sunt calea de pornire a site-ului.

O clonare prin GitHub CLI autentificat:

```bash
gh repo clone cristian292/football-predict
cd football-predict
```

Nu transmite parole/tokenuri colegilor și nu introduce credențiale în URL-uri.

## Schimbări ulterioare

Lucrează într-o singură copie de lucru, cu un branch pentru fiecare funcționalitate:

```bash
git status --short --branch
git fetch origin
git switch main
git merge --ff-only origin/main
git switch -c feat/nume-functionalitate
```

Nu executa fast-forward sau schimbare de branch peste modificări personale nesalvate. Nu folosi force-push pentru a rezolva automat diferențe de istoric.

După implementare și teste, inspectează exact fișierele alese pentru commit:

```bash
git add CALE_FISIER_1 CALE_FISIER_2
git diff --cached --stat
git diff --cached
git commit -m "feat: descrierea modificării"
git push -u origin HEAD
```

Numele `CALE_FISIER_1`/`CALE_FISIER_2` sunt substituenți, nu comenzi gata de executat. Apoi creează un pull request în GitHub și verifică rezultatul Actions înainte de integrare.

## Ce nu se publică

- `.venv`, `__pycache__`, configurații IDE;
- `.env`, parole, tokenuri, chei private;
- cache-urile brute, backupurile și rapoartele generate neinspectate;
- alte proiecte aflate în repository-ul părinte.

Un repository privat nu este un seif de credențiale. Verifică și documentația pentru căi personale înaintea unei distribuiri mai largi.

## Verificare reală înainte de distribuire

- Instalează într-un director nou, fără mediul virtual al autorului.
- Rulează `bash run.sh --doctor` și verifică în browser încărcarea, antrenarea, predicția, importul CSV și descărcarea raportului.
- Rulează testele după `bash install.sh --dev`.
- Verifică rezultatele din Actions pentru commitul/branchul publicat. Un succes pe `main` nu verifică modificările locale nepublicate.
- Confirmă explicit vizibilitatea **PRIVATE** după orice operație de publicare.

Fișierele istorice din `DOCUMENTARE_DATE.md` păstrează ce s-a executat la fiecare etapă; afirmațiile despre starea curentă GitHub trebuie verificate separat, nu deduse dintr-un jurnal vechi.
