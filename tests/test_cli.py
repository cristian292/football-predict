"""Command-line entry point stays usable without a display."""
import subprocess
import sys
import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_help_without_gtk_display():
    result = subprocess.run([sys.executable, '-m', 'football_predictor', '--help'], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '--evaluate' in result.stdout
    assert '--data' in result.stdout
    assert '--benchmark' in result.stdout
    assert '--doctor' in result.stdout


def test_default_dataset_path(monkeypatch):
    monkeypatch.setattr(Path, 'is_file', lambda p: p.name == 'matches.csv')
    from football_predictor.cli import default_data_path
    assert default_data_path() == ROOT / 'matches.csv'


def test_default_uses_canonical_matches_even_if_staging_file_exists(monkeypatch):
    monkeypatch.setattr(Path, 'is_file', lambda p: p.name in {'matches.csv', 'matches_updated.csv'})
    from football_predictor.cli import default_data_path
    assert default_data_path() == ROOT / 'matches.csv'


def test_benchmark_dispatches_and_exports_diagnostics(tmp_path, monkeypatch):
    import types
    import csv
    from football_predictor.cli import main
    source = tmp_path / 'input.csv'
    source.write_text('test-only input')
    out = tmp_path / 'benchmark'
    calls = []
    def benchmark(csv_path, output_dir, progress=None):
        calls.append(('benchmark', csv_path, output_dir))
        output_dir.mkdir()
        with (output_dir / 'predictions_by_season.csv').open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['date','season','home_team','away_team','actual','model','variant','p_home','p_draw','p_away','predicted'])
            writer.writerow(['2025-01-01',2025,'A','B',0,'KNN','legacy',0.6,0.2,0.2,0])
        return {'metrics': []}
    def diagnostics(predictions, output_dir):
        calls.append(('diagnostics', len(predictions), output_dir))
        return {}
    monkeypatch.setitem(sys.modules, 'football_predictor.experiments', types.SimpleNamespace(run_benchmark=benchmark))
    monkeypatch.setitem(sys.modules, 'football_predictor.diagnostics', types.SimpleNamespace(export_diagnostics=diagnostics))
    assert main(['--benchmark', '--data', str(source), '--output', str(out)]) == 0
    assert calls == [('benchmark', source, out), ('diagnostics', 1, out / 'diagnostics')]


def test_doctor_reports_missing_data_instead_of_parser_failure(tmp_path, monkeypatch, capsys):
    import types
    from football_predictor.cli import main
    missing = tmp_path / 'missing.csv'
    calls = []
    def check(path):
        calls.append(path)
        return {'ok': False, 'checks': [], 'warnings': []}
    monkeypatch.setitem(sys.modules, 'football_predictor.doctor', types.SimpleNamespace(check_environment=check, format_report=lambda report: 'CSV lipsă — verificare terminată'))
    assert main(['--doctor', '--data', str(missing)]) == 1
    assert calls == [missing]
    assert 'CSV lipsă' in capsys.readouterr().out


def test_default_launches_web_server_without_gtk(tmp_path, monkeypatch, capsys):
    import builtins
    import types
    from football_predictor.cli import main
    source = tmp_path / 'matches.csv'
    source.write_text('input')
    calls = []
    app = object()
    def create_app(**kwargs):
        calls.append(('app', kwargs))
        return app
    def serve(application, **kwargs):
        calls.append(('serve', application, kwargs))
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        assert name != 'gi' and '.ui' not in name
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)
    monkeypatch.setitem(sys.modules, 'football_predictor.web', types.SimpleNamespace(create_app=create_app))
    monkeypatch.setitem(sys.modules, 'waitress', types.SimpleNamespace(serve=serve))
    assert main(['--data', str(source)]) == 0
    assert calls == [('app', {'data_path': source, 'work_dir': None}),
                     ('serve', app, {'host': '127.0.0.1', 'port': 8080, 'threads': 8})]
    assert 'http://127.0.0.1:8080' in capsys.readouterr().out


def test_web_launch_accepts_host_port_and_work_dir(tmp_path, monkeypatch, capsys):
    import types
    from football_predictor.cli import main
    source = tmp_path / 'matches.csv'
    source.touch()
    work = tmp_path / 'uploads'
    calls = []
    app = object()
    def create_app(**kwargs):
        calls.append(kwargs)
        return app
    def serve(application, **kwargs):
        assert application is app
        calls.append(kwargs)
    monkeypatch.setitem(sys.modules, 'football_predictor.web', types.SimpleNamespace(create_app=create_app))
    monkeypatch.setitem(sys.modules, 'waitress', types.SimpleNamespace(serve=serve))
    assert main(['--data', str(source), '--host', '0.0.0.0', '--port', '9001', '--work-dir', str(work)]) == 0
    assert calls == [{'data_path': source, 'work_dir': work}, {'host': '0.0.0.0', 'port': 9001, 'threads': 8}]
    assert 'http://0.0.0.0:9001' in capsys.readouterr().out


@pytest.mark.parametrize('failure', [None, KeyboardInterrupt, RuntimeError])
def test_web_server_closes_workspace_on_exit(tmp_path, monkeypatch, failure):
    import types
    from football_predictor.cli import main
    source = tmp_path / 'matches.csv'
    source.touch()
    closed = []
    app = types.SimpleNamespace(extensions={'workspace': types.SimpleNamespace(close=lambda: closed.append(True))})
    monkeypatch.setitem(sys.modules, 'football_predictor.web', types.SimpleNamespace(create_app=lambda **kwargs: app))
    def serve(application, **kwargs):
        if failure:
            raise failure()
    monkeypatch.setitem(sys.modules, 'waitress', types.SimpleNamespace(serve=serve))
    if failure:
        with pytest.raises(failure):
            main(['--data', str(source)])
    else:
        assert main(['--data', str(source)]) == 0
    assert closed == [True]


@pytest.mark.parametrize('port', ['0', '-1', '65536'])
def test_invalid_web_port_is_a_friendly_parser_error(port, capsys):
    from football_predictor.cli import main
    with pytest.raises(SystemExit) as error:
        main(['--port', port, '--doctor'])
    assert error.value.code == 2
    assert '1..65535' in capsys.readouterr().err

