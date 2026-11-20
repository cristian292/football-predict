"""Application lifecycle, deliberately independent of scientific imports."""
from pathlib import Path
import gi

gi.require_version('Gtk', '4.0')
from gi.repository import Gtk, Gio
from .window import FootballWindow


class FootballApplication(Gtk.Application):
    def __init__(self, csv_path: str | Path, *, autoload: bool = True):
        super().__init__(application_id='ro.academic.FootballPredictor', flags=Gio.ApplicationFlags.NON_UNIQUE)
        self.csv_path = Path(csv_path)
        self.autoload = autoload

    def do_activate(self):
        window = self.get_active_window()
        if window is None:
            window = FootballWindow(self)
        window.present()
