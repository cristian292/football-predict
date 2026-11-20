"""Synthetic expanding-season experiments; never read the project matches.csv."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def write_seasons(path, seasons=(2022, 2023, 2024, 2025), altered_season=None):
    rows = []
    for season in seasons:
        for i in range(9):
            hg, ag = (i * 7 + season) % 4, (i * 5) % 3
            if season == altered_season:
                hg, ag = 9, 0
            rows.append(dict(date=f'{season}-03-{i + 1:02d}', season=season,
                             venue='Home', team='ABCD'[i % 4], opponent='ABCD'[(i + 1) % 4],
                             gf=hg, ga=ag))
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_benchmark_exports_one_refitted_season_with_selection_audit(tmp_path):
    from football_predictor.experiments import run_benchmark
    source = write_seasons(tmp_path / 'synthetic.csv')
    original = source.read_bytes()
    messages = []
    report = run_benchmark(source, tmp_path / 'output', progress=messages.append, test_seasons=[2024])
    assert source.read_bytes() == original
    assert report['source_sha256'] == hashlib.sha256(original).hexdigest()
    assert len(report['folds']) == 1
    fold = report['folds'][0]
    assert fold['season'] == 2024
    assert fold['train_seasons'] == [2022]
    assert fold['validation_season'] == 2023
    assert (fold['n_train'], fold['n_validation'], fold['n_test']) == (9, 9, 9)
    assert fold['training_end'] < fold['validation_start'] <= fold['validation_end'] < fold['test_start']
    assert fold['test_complete'] is True
    assert len(report['metrics']) == 8  # Three models + baseline, paired variants.
    assert {(m['variant'], m['model']) for m in report['metrics']} == {
        (v, m) for v in ('legacy', 'optimized') for m in ('KNN', 'Random Forest', 'Poisson', 'Baseline (majority)')}
    for metric in report['metrics']:
        assert metric['season'] == 2024 and metric['n_test'] == 9
        assert np.isfinite([metric[k] for k in ('accuracy', 'macro_f1', 'log_loss', 'brier')]).all()
        assert all(k in metric for k in ('training_end', 'validation_start', 'validation_end', 'n_train', 'n_validation', 'test_complete'))
    for variant in ('legacy', 'optimized'):
        for model in ('KNN', 'Random Forest', 'Poisson'):
            selected = fold['selections'][variant][model]
            trials = [t for t in report['trials'] if t['variant'] == variant and t['model'] == model]
            assert selected['params'] == min(trials, key=lambda t: t['validation_log_loss'])['params']
        optimization = fold['poisson_optimization'][variant]
        assert optimization['selection']['n_train'] == 9
        assert optimization['evaluation']['n_train'] == 18
    paths = report['paths']
    assert set(paths) == {'metrics_by_season.csv', 'predictions_by_season.csv', 'tuning_by_season.csv', 'benchmark.json'}
    loaded = json.loads(Path(paths['benchmark.json']).read_text())
    assert loaded == report
    assert 'retrospective' in report['protocol'].lower()
    predictions = pd.read_csv(paths['predictions_by_season.csv'])
    assert set(predictions.columns) == {'date', 'season', 'home_team', 'away_team', 'actual', 'model', 'variant', 'p_home', 'p_draw', 'p_away', 'predicted'}
    assert len(predictions) == 72
    assert predictions.date.str.fullmatch(r'\d{4}-\d{2}-\d{2}').all()
    probabilities = predictions[['p_home', 'p_draw', 'p_away']].to_numpy()
    assert np.isfinite(probabilities).all() and np.all(probabilities >= 0)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    assert np.array_equal(probabilities.argmax(axis=1), predictions.predicted)
    assert len(pd.read_csv(paths['metrics_by_season.csv'])) == 8
    tuning = pd.read_csv(paths['tuning_by_season.csv'])
    assert len(tuning) == len(report['trials'])
    assert all(isinstance(json.loads(p), dict) for p in tuning.params)
    assert messages
    json.dumps(report, allow_nan=False)


def test_default_expands_history_and_labels_latest_observed_season_partial(tmp_path):
    from football_predictor.experiments import run_benchmark
    source = write_seasons(tmp_path / 'synthetic.csv')
    report = run_benchmark(source, tmp_path / 'out')
    assert [f['season'] for f in report['folds']] == [2024, 2025]
    first, last = report['folds']
    assert first['train_seasons'] == [2022]
    assert last['train_seasons'] == [2022, 2023]
    assert last['validation_season'] == 2024
    assert last['n_train'] == 18 and last['n_validation'] == 9
    assert first['test_complete'] is True
    assert last['test_complete'] is False
    assert last['test_status'] == 'partial'
    assert all(m['test_complete'] == (m['season'] == 2024) for m in report['metrics'])
    assert 'latest observed season' in report['completeness_assumption']
    predictions = pd.read_csv(report['paths']['predictions_by_season.csv'])
    assert len(predictions) == 144
    baseline = predictions.loc[predictions.model == 'Baseline (majority)']
    for _, group in baseline.groupby('season'):
        old = group.loc[group.variant == 'legacy', ['p_home', 'p_draw', 'p_away']].to_numpy()
        new = group.loc[group.variant == 'optimized', ['p_home', 'p_draw', 'p_away']].to_numpy()
        np.testing.assert_array_equal(old, new)
    for fold in report['folds']:
        for model in ('KNN', 'Random Forest', 'Poisson'):
            old = fold['selections']['legacy'][model]['trials']
            new = fold['selections']['optimized'][model]['trials']
            assert min(t['validation_log_loss'] for t in new) <= min(t['validation_log_loss'] for t in old)


def test_no_eligible_folds_fails_clearly_before_creating_outputs(tmp_path):
    from football_predictor.experiments import run_benchmark
    source = write_seasons(tmp_path / 'synthetic.csv', seasons=(2022, 2023))
    with pytest.raises(ValueError, match='eligible|prior seasons'):
        run_benchmark(source, tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


def test_missing_immediately_previous_validation_season_is_not_bridged(tmp_path):
    from football_predictor.experiments import run_benchmark
    source = write_seasons(tmp_path / 'synthetic.csv', seasons=(2021, 2022, 2025))
    with pytest.raises(ValueError, match='eligible|previous'):
        run_benchmark(source, tmp_path / 'out', test_seasons=[2025])


def test_season_labels_cannot_override_whole_date_chronology(tmp_path):
    from football_predictor.experiments import run_benchmark
    source = write_seasons(tmp_path / 'synthetic.csv')
    raw = pd.read_csv(source)
    raw.loc[raw.season == 2024, 'date'] = raw.loc[raw.season == 2024, 'date'].str.replace('2024-', '2023-')
    raw.to_csv(source, index=False)
    with pytest.raises(ValueError, match='chronolog|date'):
        run_benchmark(source, tmp_path / 'out', test_seasons=[2024])
    assert not (tmp_path / 'out').exists()


def test_output_path_cannot_overwrite_the_input_csv(tmp_path):
    from football_predictor.experiments import run_benchmark
    source = write_seasons(tmp_path / 'metrics_by_season.csv')
    original = source.read_bytes()
    with pytest.raises(ValueError, match='source|input|overwrite'):
        run_benchmark(source, tmp_path, test_seasons=[2024])
    assert source.read_bytes() == original
