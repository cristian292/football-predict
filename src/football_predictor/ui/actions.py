"""Controller actions, native file choosers and service adapters."""
from datetime import timedelta
from pathlib import Path
from gi.repository import Gtk, GLib
from .validation import prediction_error


def selected_text(dropdown):
    item = dropdown.get_selected_item()
    return item.get_string() if item else None


class WorkspaceActions:
    def connect_actions(self):
        self._closed = False
        self.import_button.connect('clicked', self._choose_import)
        self.export_button.connect('clicked', self._choose_export)
        self.train_button.connect('clicked', self._train)
        self.predict_button.connect('clicked', self._predict)
        self.season_dropdown.connect('notify::selected', self._refresh_standings)
        self.history_team.connect('notify::selected', self._refresh_history)
        self.connect('close-request', self._on_close)
        self._refresh_actions()
        if self.get_application().autoload:
            GLib.idle_add(self.load_csv, self.get_application().csv_path)

    def _on_close(self, *_):
        self._closed = True
        return False

    def load_csv(self, path):
        path = Path(path)
        def load():
            from football_predictor.service import ProjectService
            return ProjectService(path)
        self.run_job('Încărcare și validare CSV…', load, lambda service: self.install_service(service, path))
        return GLib.SOURCE_REMOVE

    def install_service(self, service, path):
        matches = service.matches
        if matches.empty:
            raise ValueError('CSV-ul nu conține meciuri încheiate.')
        self.service = service
        self.trained = False
        self.latest_date = matches.date.max().date()
        self.teams = sorted(set(matches.home_team) | set(matches.away_team))
        self.seasons = sorted(matches.season.unique(), reverse=True)
        for dropdown in (self.home_team, self.away_team, self.history_team):
            dropdown.set_model(Gtk.StringList.new(self.teams))
        self.away_team.set_selected(min(1, len(self.teams) - 1))
        self.season_dropdown.set_model(Gtk.StringList.new([str(s) for s in self.seasons]))
        self.date_entry.set_text((self.latest_date + timedelta(days=1)).isoformat())
        self.dataset_label.set_text(f'{Path(path).name}  ·  {len(matches):,} meciuri  ·  {len(self.teams)} echipe  ·  {matches.date.min():%Y-%m-%d} → {self.latest_date}')
        self.warning_label.set_text(f'Ultima partidă din sursă: {self.latest_date}. Date istorice până în {self.latest_date.year}; estimările viitoare nu sunt recomandări de pariere.')
        self._refresh_standings()
        self._refresh_history()
        # Replacing the source invalidates all previous model output.
        self.metrics_table.set_rows([])
        self.probability_table.set_rows([])
        self.matrix_table.set_rows([], ['Gazde ↓ / Oaspeți →'])
        self.fixture_label.set_text('Sursă încărcată. Antrenează modelele pentru predicții.')
        self.scores_label.set_text('Distribuția Poisson apare după calcul.')
        self.xg_label.set_text('Goluri așteptate: —')
        self.features_label.set_text('Variabilele partidei vor apărea aici.')
        self.tuning_label.set_text('Fără rezultate de ajustare.')
        self.split_label.set_text('Antrenează modelele pentru separarea efectivă.')
        self.confusion_label.set_text('Rânduri = real; coloane = prezis. Ordine: 1, X, 2.')
        self.set_status('Date încărcate. Poți începe antrenarea.')
        self._refresh_actions()

    def _refresh_standings(self, *_):
        if self.service is None:
            return
        index = self.season_dropdown.get_selected()
        if index >= len(self.seasons):
            return
        from football_predictor.data import standings
        frame = standings(self.service.matches, season=self.seasons[index])
        self.standings_table.set_rows([[r.position, r.team, r.played, r.points, r.gf, r.ga, r.gd] for r in frame.itertuples()])

    def _refresh_history(self, *_):
        if self.service is None:
            return
        team = selected_text(self.history_team)
        matches = self.service.matches
        frame = matches.loc[matches.home_team.eq(team) | matches.away_team.eq(team)].sort_values('date', ascending=False).head(20)
        self.history_table.set_rows([[f'{r.date:%Y-%m-%d}', r.home_team, f'{int(r.home_goals)}–{int(r.away_goals)}', r.away_team] for r in frame.itertuples()])

    def _train(self, *_):
        if self.service is None or self.busy:
            return
        def trained(report):
            self.show_report(report)
            self.trained = True
            self.stack.set_visible_child_name('experiments')
            self.set_status('Modele antrenate. Inspectează evaluarea sau calculează o predicție.')
        self.trained = False
        self.run_job('Antrenare și evaluare…', lambda: self.service.train(progress=self.training_progress), trained)

    def _predict(self, *_):
        if not self.trained or self.busy:
            return
        home, away = selected_text(self.home_team), selected_text(self.away_team)
        when = self.date_entry.get_text().strip()
        error = prediction_error(home, away, when, self.latest_date)
        if error:
            self.set_status(error, error=True)
            return
        def predicted(result):
            self.show_prediction(result)
            self.stack.set_visible_child_name('predict')
            self.set_status('Predicție calculată. Compară modelele și incertitudinea.')
        self.run_job('Calcul predicție…', lambda: self.service.predict(home, away, when), predicted)

    def export_to(self, directory):
        if not self.trained:
            return
        directory = Path(directory)
        self.run_job('Export raport…', lambda: self.service.export_report(directory), lambda _: self.set_status(f'Raport exportat în {directory}'))

    def create_import_dialog(self):
        dialog = Gtk.FileChooserNative.new('Importă date fotbal', self, Gtk.FileChooserAction.OPEN, 'Importă', 'Anulează')
        csv_filter = Gtk.FileFilter()
        csv_filter.set_name('Date CSV (*.csv)')
        csv_filter.add_pattern('*.csv')
        csv_filter.add_pattern('*.CSV')
        dialog.add_filter(csv_filter)
        return dialog

    def create_export_dialog(self):
        return Gtk.FileChooserNative.new('Alege directorul raportului', self, Gtk.FileChooserAction.SELECT_FOLDER, 'Exportă aici', 'Anulează')

    def _choose_import(self, *_):
        self._show_chooser(self.create_import_dialog(), self.load_csv)

    def _choose_export(self, *_):
        self._show_chooser(self.create_export_dialog(), self.export_to)

    def _show_chooser(self, dialog, callback):
        self._chooser = dialog
        def response(chooser, response_id):
            if response_id == Gtk.ResponseType.ACCEPT:
                chosen = chooser.get_file()
                path = chosen.get_path() if chosen else None
                if path:
                    callback(Path(path))
                else:
                    self.set_status('Alege un fișier sau director local.', error=True)
            chooser.destroy()
            self._chooser = None
        dialog.connect('response', response)
        dialog.show()
