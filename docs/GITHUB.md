# Pregătirea pentru GitHub

## Înainte de publicare

- Alege licența pentru cod și completează autorul. Nu adăuga automat o licență la datele extrase.
- Verifică termenii sursei înainte de redistribuirea CSV-ului. `matches.csv` este ignorat implicit.
- Nu încărca `.venv`, `.idea`, chei, `.env`, cache-uri HTML sau rapoarte cu căi personale.
- Rulează testele și experimentele, apoi inspectează manual `git diff --cached`.
- Acest director se află deja sub un depozit Git părinte. Decide explicit dacă vrei proiect separat; nu executa comenzi Git din rădăcina depozitului părinte pentru întregul calculator.

## Variantă sigură: copie separată pentru publicare

Copiază numai fișierele proiectului într-un director nou ales de tine: `src/`, `tests/`, `docs/`, `.github/`, `README.md`, `pyproject.toml`, `requirements.txt`, `run.sh`, `.gitignore`. Scripturile vechi de colectare pot fi adăugate după revizuire. Nu copia datele până la verificarea drepturilor. Apoi:

```bash
cd /cale/catre/copia-pentru-publicare
git init
git add src tests docs README.md pyproject.toml requirements.txt run.sh .gitignore
git diff --cached --stat
git diff --cached
# Numai după inspecție:
git commit -m "Implement modular GTK football prediction application"
```

Creează depozitul în contul tău și configurează remote-ul corect înainte de `push`. Nu au fost create commituri, remote-uri sau publicări automat.

## Date de intrare pentru alți utilizatori

README descrie schema CSV și comanda `--data`. Dacă nu poți publica datele reale, oferă instrucțiunile de colectare și, opțional, un mic exemplu sintetic clar etichetat ca sintetic. Nu prezenta exemplele sintetice drept evaluare empirică.
