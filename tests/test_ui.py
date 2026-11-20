"""Native GTK smoke tests; run with xvfb-run and system GI on PYTHONPATH."""
import os
import pytest

pytest.importorskip('gi')
import gi
gi.require_version('Gtk', '4.0')
from gi.repository import Gtk, GLib

pytestmark = pytest.mark.skipif(not os.environ.get('DISPLAY'), reason='GTK requires a display (use xvfb-run)')


def test_application_opens_four_romanian_pages_without_service(tmp_path):
    from football_predictor.ui.app import FootballApplication
    app = FootballApplication(tmp_path / 'missing.csv', autoload=False)
    assert app.register(None)
    app.activate()
    window = app.get_active_window()
    assert isinstance(window, Gtk.ApplicationWindow)
    assert [p.get_title() for p in window.stack.get_pages()] == ['Predicții', 'Experimente', 'Date', 'Metodologie']
    assert window.stack.get_visible_child_name() == 'predict'
    assert not window.train_button.get_sensitive()
    assert not window.home_team.get_sensitive()
    assert not window.status.get_selectable()
    assert 'Importă' in window.status.get_text()
    window.close()
    app.quit()


def test_prediction_validation_rejects_same_team_and_nonfuture_dates():
    from football_predictor.ui.validation import prediction_error
    assert 'diferite' in prediction_error('Arsenal', 'Arsenal', '2026-01-01', '2025-05-25')
    assert 'ulterioară' in prediction_error('Arsenal', 'Chelsea', '2025-05-25', '2025-05-25')
    assert 'AAAA-LL-ZZ' in prediction_error('Arsenal', 'Chelsea', 'bad', '2025-05-25')
    assert prediction_error('Arsenal', 'Chelsea', '2026-01-01', '2025-05-25') is None


def test_workspace_renders_model_report_and_score_matrix(tmp_path):
    from football_predictor.ui.app import FootballApplication
    app = FootballApplication(tmp_path / 'missing.csv', autoload=False)
    app.set_application_id('ro.academic.FootballPredictor.RenderTest')
    app.register(None)
    app.activate()
    w = app.get_active_window()
    assert isinstance(w.import_button, Gtk.Button)
    assert isinstance(w.home_team, Gtk.DropDown)
    assert isinstance(w.date_entry, Gtk.Entry)
    assert isinstance(w.export_button, Gtk.Button)
    assert not w.predict_button.get_sensitive()
    w.show_report({'metrics': [{'model': 'KNN', 'accuracy': .6, 'macro_f1': .5, 'log_loss': .9, 'brier': .2, 'n_test': 50}], 'tuning': [{'model': 'KNN', 'params': {'n_neighbors': 7}, 'validation_log_loss': 1.0}], 'split': {'test': 50}, 'confusion': {'KNN': [[1, 2, 3], [4, 5, 6], [7, 8, 9]]}})
    assert w.metrics_table.rows[0] == ['KNN', '60.0%', '0.500', '0.900', '0.200', '50']
    assert 'n_neighbors' in w.tuning_label.get_text()
    assert 'KNN' in w.confusion_label.get_text()
    w.show_prediction({'home': 'Arsenal', 'away': 'Chelsea', 'date': '2026-01-01', 'probabilities': {'KNN': [.5, .3, .2]}, 'scores': [{'score': '1–0', 'probability': .2}], 'score_matrix': [[.1, .2], [.3, .4]], 'expected_goals': [1.5, 1.1], 'features': {'form': .4}})
    assert w.probability_table.rows[0] == ['KNN', '50.0%', '30.0%', '20.0%']
    assert w.matrix_table.rows[1] == ['1', '30.0%', '40.0%']
    assert 'Arsenal' in w.fixture_label.get_text()
    assert '1–0' in w.scores_label.get_text()
    w.close()
    app.quit()


def test_background_job_updates_busy_success_and_error_on_main_loop(tmp_path):
    import threading
    import time
    from football_predictor.ui.app import FootballApplication
    app = FootballApplication(tmp_path / 'missing.csv', autoload=False)
    app.set_application_id('ro.academic.FootballPredictor.WorkerTest')
    app.register(None)
    app.activate()
    w = app.get_active_window()
    main = threading.get_ident()
    w.busy = True
    w.training_progress('Tuning KNN on validation log loss', .4)
    deadline = time.monotonic() + 5
    while w.progress.get_fraction() != .4 and time.monotonic() < deadline:
        GLib.MainContext.default().iteration(False)
    assert 'KNN' in w.status.get_text()
    assert 'validare' in w.status.get_text()
    w.busy = False
    results = []
    w.run_job('Calcul…', lambda: threading.get_ident(), lambda value: results.append((value, threading.get_ident())))
    assert w.busy
    assert not w.import_button.get_sensitive()
    deadline = time.monotonic() + 5
    while w.busy and time.monotonic() < deadline:
        GLib.MainContext.default().iteration(False)
    assert results[0][0] != main
    assert results[0][1] == main
    assert not w.busy
    assert w.import_button.get_sensitive()
    def fail():
        raise ValueError('CSV invalid')
    w.run_job('Import…', fail, lambda value: None)
    deadline = time.monotonic() + 5
    while w.busy and time.monotonic() < deadline:
        GLib.MainContext.default().iteration(False)
    assert 'CSV invalid' in w.status.get_text()
    assert w.status.has_css_class('error')
    w.close()
    app.quit()


def test_data_training_prediction_export_workflow(tmp_path):
    import time
    from types import SimpleNamespace
    import pandas as pd
    from football_predictor.ui.app import FootballApplication
    app = FootballApplication(tmp_path / 'missing.csv', autoload=False)
    app.set_application_id('ro.academic.FootballPredictor.WorkflowTest')
    app.register(None)
    app.activate()
    w = app.get_active_window()
    matches = pd.DataFrame([{'date': pd.Timestamp('2025-05-25'), 'season': 2025, 'home_team': 'Arsenal', 'away_team': 'Chelsea', 'home_goals': 2, 'away_goals': 1}])
    report = {'metrics': [], 'tuning': [], 'split': {}, 'confusion': {}}
    prediction = {'home': 'Arsenal', 'away': 'Chelsea', 'date': '2025-05-26', 'probabilities': {'Poisson': [.5,.3,.2]}, 'scores': [], 'score_matrix': [[1.0]], 'expected_goals': [2,1], 'features': {}}
    exported = []
    service = SimpleNamespace(matches=matches, train=lambda progress=None: report, predict=lambda *args: prediction, export_report=lambda directory: exported.append(directory))
    w.install_service(service, tmp_path / 'matches.csv')
    assert w.train_button.get_sensitive()
    assert w.home_team.get_model().get_n_items() == 2
    assert w.standings_table.rows[0][1] == 'Arsenal'
    assert w.history_table.rows[0] == ['2025-05-25', 'Arsenal', '2–1', 'Chelsea']
    assert w.date_entry.get_text() == '2025-05-26'
    service.matches.loc[:, 'date'] = pd.Timestamp('2026-06-01')
    w.install_service(service, tmp_path / 'updated.csv')
    assert '2026-06-01' in w.warning_label.get_text()
    assert 'până în 2025' not in w.warning_label.get_text()
    service.matches.loc[:, 'date'] = pd.Timestamp('2025-05-25')
    w.install_service(service, tmp_path / 'matches.csv')
    def settle():
        deadline = time.monotonic() + 5
        while w.busy and time.monotonic() < deadline:
            GLib.MainContext.default().iteration(False)
        assert not w.busy
    w.train_button.emit('clicked')
    settle()
    assert w.trained
    assert w.predict_button.get_sensitive()
    w.home_team.set_selected(0)
    w.away_team.set_selected(0)
    w.predict_button.emit('clicked')
    assert 'diferite' in w.status.get_text()
    w.away_team.set_selected(1)
    w.predict_button.emit('clicked')
    settle()
    assert 'Arsenal' in w.fixture_label.get_text()
    w.export_to(tmp_path)
    settle()
    assert exported == [tmp_path]
    assert isinstance(w.create_import_dialog(), Gtk.FileChooserNative)
    assert w.create_export_dialog().get_action() == Gtk.FileChooserAction.SELECT_FOLDER
    w.close()
    app.quit()
