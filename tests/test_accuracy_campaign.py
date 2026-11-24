"""Campaign guardrails must be independent of searched model implementations."""
import numpy as np
import pandas as pd
import pytest


def test_inner_folds_expand_without_splitting_dates():
    from football_predictor.accuracy_campaign import development_folds
    frame = pd.DataFrame({'date': np.repeat(pd.date_range('2020-01-01', periods=24), 2)},
                         index=np.arange(48) + 100)
    folds = development_folds(frame)
    assert len(folds) == 3
    previous_train = set()
    validation_labels = []
    for train, validation in folds:
        assert previous_train <= set(train)
        assert set(train).isdisjoint(validation)
        assert frame.loc[train, 'date'].max() < frame.loc[validation, 'date'].min()
        for date in frame.loc[validation, 'date'].unique():
            assert set(frame.index[frame.date.eq(date)]) <= set(validation)
        previous_train = set(train)
        validation_labels.extend(validation)
    assert len(set(validation_labels)) == len(validation_labels)
    assert set(validation_labels) == set(frame.index[frame.date >= frame.date.unique()[12]])


def test_paired_interval_is_zero_when_predictions_are_identical():
    from football_predictor.accuracy_campaign import paired_accuracy_interval
    dates = pd.date_range('2020-01-01', periods=20).repeat(2)
    actual = np.tile([0, 1], 20)
    predictions = np.tile([0, 2], 20)
    result = paired_accuracy_interval(actual, predictions, predictions, dates,
                                      n_bootstrap=100, seed=42)
    assert result['delta'] == 0
    assert result['low'] == result['high'] == 0
    assert result['n_matches'] == len(actual)
    assert result['n_dates'] == 20


def test_paired_interval_reports_real_paired_gain_reproducibly():
    from football_predictor.accuracy_campaign import paired_accuracy_interval
    actual = np.tile([0, 1, 2], 20)
    before = np.zeros(len(actual), dtype=int)
    dates = pd.date_range('2020-01-01', periods=20).repeat(3)
    first = paired_accuracy_interval(actual, before, actual, dates, n_bootstrap=100)
    second = paired_accuracy_interval(actual, before, actual, dates, n_bootstrap=100)
    assert first == second
    assert first['delta'] == float(np.mean((actual == actual).astype(float) - (before == actual)))
    assert first['low'] > 0


def test_protocol_freeze_refuses_to_overwrite_a_different_search(tmp_path):
    from football_predictor.accuracy_campaign import freeze_campaign_protocol
    protocol = {'source_sha256': 'fixture-hash', 'objective': 'accuracy', 'seed': 42,
                'candidates': [{'id': 'candidate-1'}]}
    digest = freeze_campaign_protocol(tmp_path, protocol)
    assert freeze_campaign_protocol(tmp_path, protocol) == digest
    original = (tmp_path / 'protocol.json').read_bytes()
    with pytest.raises(ValueError, match='frozen'):
        freeze_campaign_protocol(tmp_path, {**protocol, 'seed': 43})
    assert (tmp_path / 'protocol.json').read_bytes() == original


def test_probability_adjustment_is_normalized_and_identity_preserving():
    from football_predictor.accuracy_campaign import adjust_probabilities
    p = np.array([[0.4, 0.3, 0.3], [0.1, 0.2, 0.7]])
    np.testing.assert_allclose(adjust_probabilities(p, [1, 1, 1]), p)
    adjusted = adjust_probabilities(p, [1, 0.8, 1.2])
    np.testing.assert_allclose(adjusted.sum(axis=1), 1)
    assert np.all(adjusted >= 0)
    np.testing.assert_allclose(p, [[0.4, 0.3, 0.3], [0.1, 0.2, 0.7]])
    with pytest.raises(ValueError):
        adjust_probabilities(p, [1, -1, 1])


def test_class_bias_is_selected_only_from_supplied_validation_predictions():
    from football_predictor.accuracy_campaign import select_probability_adjustment
    actual = np.array([0, 2, 2, 1])
    p = np.array([[.6, .2, .2], [.41, .1, .49], [.45, .05, .5], [.1, .8, .1]])
    result = select_probability_adjustment(actual, p)
    assert result['selected']['multipliers'] == [1.0, 1.0, 1.0]
    assert len(result['trials']) == 9
    assert result['selected']['metrics']['accuracy'] == 1
    assert result['selected']['metrics']['log_loss'] <= result['raw_metrics']['log_loss'] + .05


def test_class_bias_can_improve_validation_without_touching_input():
    from football_predictor.accuracy_campaign import select_probability_adjustment
    p = np.array([[.7, .2, .1], [.1, .7, .2], [.45, .11, .44]])
    original = p.copy()
    result = select_probability_adjustment([0, 1, 2], p)
    assert result['selected']['metrics']['accuracy'] > result['raw_metrics']['accuracy']
    assert result['selected']['multipliers'] != [1.0, 1.0, 1.0]
    np.testing.assert_array_equal(p, original)


def test_small_campaign_uses_real_models_and_exports_paired_predictions(tmp_path):
    import hashlib
    import json
    from football_predictor.accuracy_campaign import run_accuracy_campaign
    rng = np.random.default_rng(12)
    teams = ['A', 'B', 'C', 'D']
    rows = []
    for i, day in enumerate(pd.date_range('2020-01-01', periods=75, freq='3D')):
        rows.append(dict(date=day, season=day.year + int(day.month >= 8), venue='Home',
                         team=teams[i % 4], opponent=teams[(i + 1 + (i // 4) % 3) % 4],
                         gf=int(rng.poisson(1.6)), ga=int(rng.poisson(1.1))))
    source = tmp_path / 'matches.csv'
    pd.DataFrame(rows).to_csv(source, index=False)
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    candidates = {
        'KNN': [dict(id='small-knn', feature_set='strength', params={'n_neighbors': 7}, train_window_days=None)],
        'Random Forest': [dict(id='small-rf', feature_set='strength', params={'n_estimators': 10, 'max_depth': 3}, train_window_days=None)],
        'Poisson': [dict(id='small-poisson', kind='rolling_poisson', params={'window': 10, 'smoothing': 5.0})],
    }
    out = tmp_path / 'campaign'
    report = run_accuracy_campaign(source, out, candidate_sets=candidates, workers=1)
    assert report['status'] == 'complete'
    assert report['source_sha256'] == before
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    assert report['test']['start'] > report['development']['end']
    assert set(report['selections']) == {'KNN', 'Random Forest', 'Poisson'}
    predictions = pd.read_csv(out / 'predictions.csv')
    assert set(predictions.variant) == {'legacy', 'advanced_raw', 'advanced_selected'}
    keys = ['date', 'home_team', 'away_team']
    expected = None
    for _, part in predictions.groupby(['variant', 'model']):
        assert not part.duplicated(keys).any()
        fixtures = set(map(tuple, part[keys].to_numpy()))
        expected = fixtures if expected is None else expected
        assert fixtures == expected
        np.testing.assert_allclose(part[['p_home','p_draw','p_away']].sum(axis=1), 1)
    assert len(report['confidence_intervals']) == 3
    assert json.loads((out / 'protocol.json').read_text())['sha256'] == report['protocol_sha256']
    assert (out / 'selected_configs.json').is_file()
    assert (out / 'per_class.csv').is_file()
    assert (out / 'metrics_by_season.csv').is_file()


def test_campaign_cli_exposes_reproducible_arguments():
    import os
    import subprocess
    import sys
    result = subprocess.run([sys.executable, '-m', 'football_predictor.accuracy_campaign', '--help'],
                            capture_output=True, text=True, env={**os.environ, 'PYTHONNOUSERSITE': '1'})
    assert result.returncode == 0
    assert '--data' in result.stdout and '--output' in result.stdout and '--workers' in result.stdout

