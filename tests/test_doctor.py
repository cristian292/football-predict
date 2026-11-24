"""Offline environment and CSV-header diagnostics."""
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def available_dependencies(monkeypatch):
    def load(name):
        if name == 'gi':
            return SimpleNamespace(require_version=lambda namespace, version: None)
        if name == 'gi.repository.Gtk':
            return SimpleNamespace(get_major_version=lambda: 4, get_minor_version=lambda: 12,
                                   get_micro_version=lambda: 5)
        return SimpleNamespace(__version__='1.2.3', __file__=f'/example/{name}/__init__.py')
    monkeypatch.setattr(importlib, 'import_module', load)
    monkeypatch.setattr(importlib.metadata, 'version', lambda name: '1.2.3')


def test_report_contains_python_and_runtime_dependency_versions(available_dependencies):
    from football_predictor import doctor
    report = doctor.check_environment()
    checks = {item['name']: item for item in report['checks']}
    assert report['ok'] is True
    assert checks['Python']['ok'] is True
    assert '.'.join(map(str, sys.version_info[:3])) in checks['Python']['detail']
    for name in ('numpy', 'pandas', 'scipy', 'sklearn', 'flask', 'waitress'):
        assert checks[name]['ok'] is True
        assert '1.2.3' in checks[name]['detail']
        assert f'/example/{name}/__init__.py' in checks[name]['detail']
    assert set(report) == {'ok', 'checks', 'warnings'}
    for item in report['checks']:
        assert set(item) == {'name', 'ok', 'detail'}
        assert isinstance(item['ok'], bool)
        assert isinstance(item['detail'], str)
    assert json.loads(json.dumps(report, allow_nan=False)) == report


@pytest.mark.parametrize('error', [ModuleNotFoundError('numpy absent'), OSError('broken shared library'), RuntimeError('ABI mismatch')])
def test_missing_or_broken_runtime_dependency_is_a_failed_check(monkeypatch, available_dependencies, error):
    from football_predictor import doctor
    original = importlib.import_module

    def load(name):
        if name == 'numpy':
            raise error
        return original(name)

    monkeypatch.setattr(importlib, 'import_module', load)
    report = doctor.check_environment()
    assert report['ok'] is False
    checks = {item['name']: item for item in report['checks']}
    assert checks['numpy']['ok'] is False
    assert str(error) in checks['numpy']['detail']
    assert checks['pandas']['ok'] is True
    json.dumps(report, allow_nan=False)


def test_web_diagnostics_never_import_gtk_or_extend_pythonpath(monkeypatch, available_dependencies):
    from football_predictor import doctor
    original = importlib.import_module
    original_path = sys.path.copy()
    imported = []

    def load(name):
        imported.append(name)
        if name.startswith('gi'):
            raise ImportError('GTK deliberately unavailable')
        return original(name)

    monkeypatch.setattr(importlib, 'import_module', load)
    report = doctor.check_environment()
    assert report['ok'] is True
    assert not any(name.startswith('gi') for name in imported)
    assert not any(item['name'] == 'GTK4' for item in report['checks'])
    assert sys.path == original_path


@pytest.mark.parametrize('display', [None, 'DISPLAY', 'WAYLAND_DISPLAY'])
def test_headless_web_session_needs_no_display_warning(monkeypatch, available_dependencies, display):
    from football_predictor import doctor
    monkeypatch.delenv('DISPLAY', raising=False)
    monkeypatch.delenv('WAYLAND_DISPLAY', raising=False)
    if display:
        monkeypatch.setenv(display, ':example')
    report = doctor.check_environment()
    assert report['ok'] is True
    warnings = [value for value in report['warnings'] if 'DISPLAY' in value]
    assert warnings == []


def test_scraping_dependency_is_optional_and_not_imported(monkeypatch, available_dependencies):
    from football_predictor import doctor
    original = importlib.import_module
    imported = []

    def load(name):
        imported.append(name)
        if name == 'soccerdata':
            raise ModuleNotFoundError('not installed')
        return original(name)

    monkeypatch.setattr(importlib, 'import_module', load)
    report = doctor.check_environment()
    assert report['ok'] is True
    assert 'soccerdata' not in imported
    assert any('soccerdata' in warning and 'opțional' in warning for warning in report['warnings'])


def test_missing_data_file_is_a_failed_check(tmp_path, available_dependencies):
    from football_predictor import doctor
    path = tmp_path / 'missing.csv'
    report = doctor.check_environment(path)
    assert report['ok'] is False
    check = next(item for item in report['checks'] if item['name'] == 'Date CSV')
    assert check['ok'] is False
    assert str(path) in check['detail']


@pytest.mark.parametrize(('contents', 'expected_ok', 'detail'), [
    (b'date,season,venue,team,opponent,gf,ga\n', True, 'schem'),
    ('\ufeffdate,season,venue,team,opponent,gf,ga,sh\n'.encode(), True, 'schem'),
    (b'date,team\n', False, 'season'),
    (b'', False, 'date'),
    (b'\xff\xfe', False, 'UnicodeDecodeError'),
    (b'"unterminated', False, 'Error'),
])
def test_csv_header_schema_uses_stdlib(tmp_path, available_dependencies, contents, expected_ok, detail):
    from football_predictor import doctor
    path = tmp_path / 'date.csv'
    path.write_bytes(contents)
    report = doctor.check_environment(str(path))
    check = next(item for item in report['checks'] if item['name'] == 'Date CSV')
    assert check['ok'] is expected_ok
    assert report['ok'] is expected_ok
    assert detail in check['detail'].removeprefix(f'{path}: ')
    assert str(path) in check['detail']
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize(('ok', 'status'), [(True, 'OK'), (False, 'EROARE')])
def test_formatter_is_readable_romanian_and_does_not_print(capsys, ok, status):
    from football_predictor import doctor
    report = {'ok': ok, 'checks': [{'name': 'Python', 'ok': ok, 'detail': 'versiune test'}],
              'warnings': ['fără afișaj']}
    text = doctor.format_report(report)
    assert 'Diagnostic' in text
    assert f'[{status}] Python: versiune test' in text
    assert 'AVERTISMENT: fără afișaj' in text
    assert ('Rezultat: OK' if ok else 'Rezultat: EROARE') in text
    assert capsys.readouterr().out == ''


def test_existing_pythonpath_is_unchanged_when_gi_is_available(available_dependencies):
    from football_predictor import doctor
    original_path = sys.path.copy()
    assert doctor.check_environment()['ok'] is True
    assert sys.path == original_path


def test_python_older_than_supported_is_a_failed_check(monkeypatch, available_dependencies):
    from football_predictor import doctor
    monkeypatch.setattr(sys, 'version_info', (3, 10, 9))
    report = doctor.check_environment()
    check = next(item for item in report['checks'] if item['name'] == 'Python')
    assert report['ok'] is False
    assert check['ok'] is False
    assert '3.10.9' in check['detail'] and '>=3.11' in check['detail']


def test_no_data_path_does_not_assume_a_machine_specific_dataset(available_dependencies):
    from football_predictor import doctor
    report = doctor.check_environment()
    assert report['ok'] is True
    assert not any(item['name'] == 'Date CSV' for item in report['checks'])


def test_data_directory_is_not_a_csv(tmp_path, available_dependencies):
    from football_predictor import doctor
    report = doctor.check_environment(tmp_path)
    assert report['ok'] is False
    assert next(item for item in report['checks'] if item['name'] == 'Date CSV')['ok'] is False


def test_doctor_import_is_lazy_and_silent():
    source = """
import importlib.util, sys
assert importlib.util.find_spec('football_predictor.doctor') is not None
before = sys.path.copy()
import football_predictor.doctor
assert sys.path == before
assert not any(name in sys.modules for name in ('gi', 'numpy', 'pandas', 'scipy', 'sklearn', 'football_predictor.service', 'football_predictor.data'))
"""
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / 'src'))
    result = subprocess.run([sys.executable, '-c', source], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout == ''
    assert result.stderr == ''
