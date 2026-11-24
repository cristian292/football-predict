# Teste istorice GTK și schițe Docker

Aplicația activă este acum un site Flask/Waitress. Acest director păstrează testele fostei interfețe GTK și ale schițelor de containerizare oprite la cererea utilizatorului. Nu sunt șterse și nu sunt prezentate ca verificări ale aplicației web.

`pyproject.toml` limitează colectarea implicită la `tests/`. Astfel, `python -m pytest` verifică aplicația web și nucleul științific fără GTK, X11 sau un runtime de containere.

- `test_ui.py`: interfața nativă veche; necesită GTK și display virtual/real.
- `test_container_config.py`, `test_container_launcher.py`: contracte ale schiței GTK/Docker, nu ale viitoarei imagini web. Există patru eșecuri X11 cunoscute în schița launcherului; acestea nu au fost remediate prin migrarea web.

Aceste teste pot fi selectate explicit pentru investigare, de exemplu `python -m pytest legacy-tests/test_container_launcher.py`. Nu rulați toate testele native într-un mediu fără display: bibliotecile GTK pot opri procesul la nivel nativ. Imaginea web va fi proiectată, construită și testată separat, împreună cu utilizatorul.
