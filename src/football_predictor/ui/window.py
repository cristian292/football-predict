"""Native monitor/compare workspace. Scientific work is wired separately."""
import json
from pathlib import Path
from gi.repository import Gtk, Gdk
from .widgets import DataTable, label, section
from .jobs import BackgroundJobs
from .actions import WorkspaceActions


class FootballWindow(WorkspaceActions, BackgroundJobs, Gtk.ApplicationWindow):
    def __init__(self, app):
        super().__init__(application=app, title='Fotbal · Laborator predictiv', default_width=1260, default_height=860)
        self.service = None
        self.trained = False
        self.busy = False
        provider = Gtk.CssProvider()
        provider.load_from_path(str(Path(__file__).with_name('style.css')))
        Gtk.StyleContext.add_provider_for_display(Gdk.Display.get_default(), provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)
        settings = Gtk.Settings.get_default()
        settings.set_property('gtk-application-prefer-dark-theme', True)
        root = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.set_child(root)
        header = Gtk.HeaderBar()
        title_label = label('FOTBAL  /  Laborator predictiv', 'heading')
        title_label.set_wrap(False)
        header.set_title_widget(title_label)
        self.set_titlebar(header)
        self.import_button = Gtk.Button(label='Importă CSV…')
        self.train_button = Gtk.Button(label='Antrenează modele', sensitive=False, css_classes=['suggested-action'])
        self.export_button = Gtk.Button(label='Exportă raport…', sensitive=False)
        header.pack_start(self.import_button)
        header.pack_end(self.export_button)
        header.pack_end(self.train_button)
        body = Gtk.Box()
        root.append(body)
        self.stack = Gtk.Stack(hexpand=True, vexpand=True)
        sidebar = Gtk.StackSidebar(stack=self.stack)
        sidebar.set_size_request(184, -1)
        body.append(sidebar)
        body.append(self.stack)
        self.pages = {}
        for name, title, subtitle in [
            ('predict', 'Predicții', 'Compară probabilitățile 1 / X / 2 și distribuția scorurilor.'),
            ('experiments', 'Experimente', 'Evaluare pe date nevăzute · aceeași separare pentru toate modelele.'),
            ('data', 'Date', 'Inspectează sursa, clasamentul și istoricul echipelor.'),
            ('method', 'Metodologie', 'Protocol reproductibil pentru un proiect academic.')]:
            page = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=18)
            for edge in ('top', 'bottom', 'start', 'end'):
                getattr(page, f'set_margin_{edge}')(24)
            page.append(label(title, 'title-1'))
            page.append(label(subtitle, 'muted'))
            scroll = Gtk.ScrolledWindow(hscrollbar_policy=Gtk.PolicyType.AUTOMATIC, vscrollbar_policy=Gtk.PolicyType.AUTOMATIC)
            scroll.set_child(page)
            self.stack.add_titled(scroll, name, title)
            self.pages[name] = page
        self._prediction_page()
        self._experiments_page()
        self._data_page()
        self._method_page()
        self.progress = Gtk.ProgressBar(show_text=False)
        root.append(self.progress)
        status_row = Gtk.Box(spacing=10, css_classes=['status'])
        self.spinner = Gtk.Spinner()
        self.status = label('Importă un CSV pentru a începe.')
        status_row.append(self.spinner)
        status_row.append(self.status)
        root.append(status_row)
        self.connect_actions()

    def _prediction_page(self):
        page = self.pages['predict']
        self.warning_label = label('Date istorice până în 2025. Estimările pentru viitor nu constituie recomandări de pariere.', 'warning')
        page.append(self.warning_label)
        form = Gtk.Box(spacing=14)
        self.home_team = Gtk.DropDown.new_from_strings([])
        self.away_team = Gtk.DropDown.new_from_strings([])
        self.date_entry = Gtk.Entry(placeholder_text='AAAA-LL-ZZ', width_chars=12)
        self.predict_button = Gtk.Button(label='Calculează predicția', sensitive=False, css_classes=['suggested-action'])
        for title, widget in [('Gazde', self.home_team), ('Oaspeți', self.away_team), ('Data partidei', self.date_entry)]:
            field = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
            field.set_hexpand(title != 'Data partidei')
            field.append(label(title, 'muted'))
            field.append(widget)
            form.append(field)
        self.predict_button.set_valign(Gtk.Align.END)
        form.append(self.predict_button)
        page.append(section('Partidă viitoare', form))
        self.fixture_label = label('Selectează echipele, apoi antrenează modelele.', 'heading')
        page.append(self.fixture_label)
        self.probability_table = DataTable(['Model', '1 · Gazde', 'X · Egal', '2 · Oaspeți'])
        page.append(section('Probabilități rezultat', self.probability_table))
        self.scores_label = label('Distribuția Poisson apare după calcul.', 'muted')
        self.xg_label = label('Goluri așteptate: —', 'muted')
        self.matrix_table = DataTable(['Gazde ↓ / Oaspeți →'])
        scores = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        scores.append(self.xg_label)
        scores.append(self.scores_label)
        scores.append(self.matrix_table)
        page.append(section('Scoruri · model Poisson', scores))
        self.features_label = label('Variabilele partidei vor apărea aici.', 'muted')
        page.append(section('Variabile la momentul predicției', self.features_label))

    def _experiments_page(self):
        page = self.pages['experiments']
        self.metrics_table = DataTable(['Model', 'Acuratețe ↑', 'Macro F1 ↑', 'Log loss ↓', 'Brier ↓', 'N test'])
        page.append(section('Comparație pe setul de test', self.metrics_table))
        self.split_label = label('Antrenează modelele pentru a vedea separarea temporală.', 'muted')
        self.tuning_label = label('Fără rezultate de ajustare.', 'muted')
        self.confusion_label = label('Rânduri = real; coloane = prezis. Ordine: 1, X, 2.', 'numeric')
        page.append(section('Separarea datelor', self.split_label))
        page.append(section('Hiperparametri · validare', self.tuning_label))
        page.append(section('Matrici de confuzie', self.confusion_label))

    def _data_page(self):
        page = self.pages['data']
        self.dataset_label = label('Nicio sursă încărcată.', 'muted')
        page.append(self.dataset_label)
        self.season_dropdown = Gtk.DropDown.new_from_strings([])
        page.append(section('Sezon · clasament', self.season_dropdown))
        self.standings_table = DataTable(['Loc', 'Echipă', 'M', 'Pct', 'GM', 'GP', 'DG'])
        page.append(self.standings_table)
        self.history_team = Gtk.DropDown.new_from_strings([])
        page.append(section('Istoric echipă · ultimele 20 de partide', self.history_team))
        self.history_table = DataTable(['Data', 'Gazde', 'Scor', 'Oaspeți'])
        page.append(self.history_table)

    def _method_page(self):
        page = self.pages['method']
        for title, text in [
            ('01 / Obiectiv și modele', 'Estimarea rezultatului 1 / X / 2 cu KNN și Random Forest; distribuție a scorurilor cu Poisson. Probabilitățile exprimă incertitudine, nu certitudini.'),
            ('02 / Prevenirea scurgerii de informații', 'Variabilele unei partide sunt calculate numai din meciuri anterioare. Separarea antrenare / validare / test este cronologică. Hiperparametrii se selectează pe validare, nu pe test.'),
            ('03 / Interpretarea evaluării', 'Acuratețea și macro F1: mai mare este mai bine. Log loss și scorul Brier: mai mic este mai bine. Consultă matricile de confuzie pentru erorile între clase; dimensiunea testului contează.'),
            ('04 / Limite', 'Istoricul se încheie în 2025. Schimbările de lot, accidentările și evenimentele neincluse în CSV nu sunt cunoscute. Extrapolarea către sezoane viitoare poate reduce calitatea estimărilor. Proiect educațional, nu consiliere financiară.'),
            ('05 / Reproductibilitate', 'Importă sursa CSV, antrenează, inspectează Experimente și exportă raportul. Păstrează fișierul-sursă împreună cu rezultatele; raportul serviciului descrie parametrii și separarea efectivă.')]:
            page.append(section(title, label(text, 'muted')))

    def show_report(self, report):
        self.metrics_table.set_rows([[r['model'], f"{r['accuracy']:.1%}", f"{r['macro_f1']:.3f}", f"{r['log_loss']:.3f}", f"{r['brier']:.3f}", r['n_test']] for r in report['metrics']])
        self.split_label.set_text(json.dumps(report.get('split', {}), ensure_ascii=False, indent=2, default=str))
        self.tuning_label.set_text('\n'.join(f"{r['model']} · {r['params']} · log loss validare {r['validation_log_loss']:.4f}" for r in report.get('tuning', [])))
        self.confusion_label.set_text('\n\n'.join(name + ' · real ↓ / prezis →  [1, X, 2]\n' + '\n'.join('   '.join(str(v) for v in row) for row in matrix) for name, matrix in report.get('confusion', {}).items()))

    def show_prediction(self, result):
        self.fixture_label.set_text(f"{result['home']}  —  {result['away']}   /   {result['date']}")
        self.probability_table.set_rows([[model] + [f'{v:.1%}' for v in values] for model, values in result['probabilities'].items()])
        self.xg_label.set_text('Goluri așteptate · gazde {:.2f} / oaspeți {:.2f}'.format(*result['expected_goals']))
        self.scores_label.set_text('   ·   '.join(f"{r['score']}  {r['probability']:.1%}" for r in result['scores'][:6]))
        matrix = result['score_matrix']
        self.matrix_table.set_rows([[str(i)] + [f'{v:.1%}' for v in row] for i, row in enumerate(matrix)], ['G ↓ / O →'] + [str(i) for i in range(len(matrix[0]) if matrix else 0)])
        self.features_label.set_text('   ·   '.join(f'{key}: {value:.3f}' if isinstance(value, (int, float)) else f'{key}: {value}' for key, value in result.get('features', {}).items()))
