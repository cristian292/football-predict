"""Single-flight background operations; all GTK writes stay on the main loop."""
import threading
from gi.repository import GLib


class BackgroundJobs:
    def _refresh_actions(self):
        available = not self.busy
        self.import_button.set_sensitive(available)
        self.train_button.set_sensitive(available and self.service is not None)
        self.predict_button.set_sensitive(available and self.trained)
        self.export_button.set_sensitive(available and self.trained)
        for widget in (self.home_team, self.away_team, self.date_entry, self.season_dropdown, self.history_team):
            widget.set_sensitive(available and self.service is not None)

    def set_status(self, text, error=False):
        self.status.set_text(text)
        if error:
            self.status.add_css_class('error')
        else:
            self.status.remove_css_class('error')

    def run_job(self, title, operation, on_success):
        if self.busy or getattr(self, '_closed', False):
            return
        self.busy = True
        self.set_status(title)
        self.spinner.start()
        self.progress.set_fraction(0)
        self._refresh_actions()

        def finish(value, error):
            if getattr(self, '_closed', False):
                return GLib.SOURCE_REMOVE
            self.busy = False
            self.spinner.stop()
            try:
                if error is not None:
                    self.set_status(f'Operația nu a reușit: {error}', error=True)
                else:
                    on_success(value)
                    self.progress.set_fraction(1)
                    if self.status.get_text() == title:
                        self.set_status('Operație finalizată.')
            except Exception as exc:
                self.set_status(f'Nu se pot afișa rezultatele: {exc}', error=True)
            finally:
                self._refresh_actions()
            return GLib.SOURCE_REMOVE

        def work():
            try:
                value = operation()
            except Exception as exc:
                GLib.idle_add(finish, None, exc)
            else:
                GLib.idle_add(finish, value, None)

        threading.Thread(target=work, name='football-worker', daemon=True).start()

    def training_progress(self, *args, **kwargs):
        """Tolerate both message-only and (fraction, message) services."""
        def update():
            if not self.busy or getattr(self, '_closed', False):
                return GLib.SOURCE_REMOVE
            fraction = kwargs.get('fraction')
            messages = []
            for value in args:
                if isinstance(value, (float, int)):
                    fraction = value
                elif isinstance(value, str):
                    translations = {
                        'Preparing chronological train / validation / test partitions': 'Separare cronologică: antrenare / validare / test',
                        'Tuning KNN on validation log loss': 'Ajustare KNN pe log loss de validare',
                        'Tuning Random Forest on validation log loss': 'Ajustare Random Forest pe log loss de validare',
                        'Optimizing Poisson training goal likelihood with L-BFGS-B': 'Optimizare Poisson cu L-BFGS-B pe setul de antrenare',
                        'Refitting selected models on train + validation; scoring held-out test once': 'Reantrenare pe antrenare + validare; evaluare unică pe test',
                        'Refitting all-history models for strictly future predictions': 'Reantrenare pe întreg istoricul pentru predicții viitoare',
                        'Training and held-out evaluation complete': 'Antrenare și evaluare finalizate',
                    }
                    messages.append(translations.get(value, value))
            if fraction is not None:
                self.progress.set_fraction(max(0, min(1, float(fraction))))
            else:
                self.progress.pulse()
            if messages:
                self.set_status(' · '.join(messages))
            return GLib.SOURCE_REMOVE
        GLib.idle_add(update)
