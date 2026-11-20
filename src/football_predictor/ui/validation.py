"""Input rules shared by the GTK action and tests."""
from datetime import date


def prediction_error(home, away, when, latest):
    if not home or not away:
        return 'Selectează ambele echipe.'
    if home == away:
        return 'Alege două echipe diferite.'
    try:
        selected = date.fromisoformat(when)
        cutoff = date.fromisoformat(str(latest)[:10])
    except (TypeError, ValueError):
        return 'Data trebuie să respecte formatul AAAA-LL-ZZ.'
    if selected <= cutoff:
        return f'Data trebuie să fie ulterioară ultimei partide ({cutoff}).'
    return None
