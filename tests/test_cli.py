"""Command-line entry point stays usable without a display."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_help_without_gtk_display():
    result = subprocess.run([sys.executable, '-m', 'football_predictor', '--help'], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '--evaluate' in result.stdout
    assert '--data' in result.stdout
    assert '--benchmark' in result.stdout


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

