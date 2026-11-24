"""Opt-in adapter contracts, exercised with real small estimators."""
import copy
import hashlib
import json

import numpy as np
import pandas as pd
import pytest


def inputs(tmp_path):
    path = tmp_path / 'matches.csv'
    rows = []
    teams = ['A', 'B', 'C', 'D']
    for day in range(30):
        for offset in (0, 2):
            home, away = teams[(day + offset) % 4], teams[(day + offset + 1) % 4]
            gf, ga = [(2, 0), (1, 1), (0, 2)][(day + offset) % 3]
            rows.append(dict(date=str((pd.Timestamp('2024-01-01') + pd.Timedelta(days=day)).date()),
                             season=2024, venue='Home', team=home, opponent=away, gf=gf, ga=ga))
    pd.DataFrame(rows).to_csv(path, index=False)
    metrics = dict(accuracy=.4, macro_f1=.3, log_loss=1.2, brier=.7, n_test=12)
    config = dict(schema_version=1, source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  protocol_sha256='a' * 64, selection_cutoff='2024-01-24', models={
        'KNN': dict(candidate=dict(id='small-knn', feature_set='form', params=dict(n_neighbors=3)),
                    multipliers=[1, 1.2, .8], validation_metrics=metrics),
        'Random Forest': dict(candidate=dict(id='small-rf', feature_set='strength',
                                             params=dict(n_estimators=5, max_depth=3)),
                              multipliers=[1, 1, 1], validation_metrics=metrics),
        'Poisson': dict(candidate=dict(id='small-poisson', kind='rolling_poisson',
                                      params=dict(window=5, smoothing=3.)),
                        multipliers=[1, 1.4, .7], validation_metrics=metrics)})
    return path, copy.deepcopy(config)


def service_type():
    from football_predictor.optimized_service import OptimizedProjectService
    return OptimizedProjectService


def test_constructor_requires_exact_source_and_pretest_selection(tmp_path):
    path, config = inputs(tmp_path)
    cls = service_type()
    service = cls(path, config)
    assert len(service.matches) == 60
    assert service.report is None
    assert 'home_form_20_gf' in service.features
    config['source_sha256'] = 'b' * 64
    with pytest.raises(ValueError, match='source|hash'):
        cls(path, config)
    _, config = inputs(tmp_path)
    config['selection_cutoff'] = '2024-01-25'
    with pytest.raises(ValueError, match='cutoff|test'):
        cls(path, config)


@pytest.mark.parametrize('change', [
    lambda c: c['models'].pop('Poisson'),
    lambda c: c.update(protocol_sha256='invalid'),
    lambda c: c.update(schema_version=True),
    lambda c: c.update(selection_cutoff='2024-01-24T12:00:00'),
    lambda c: c['models']['KNN'].update(multipliers=[1, 0, 1]),
    lambda c: c['models']['KNN']['candidate'].update(feature_set='result'),
    lambda c: c['models']['KNN']['candidate']['params'].update(unknown=3),
    lambda c: c['models']['Poisson']['candidate'].update(kind='unknown'),
    lambda c: c['models']['Poisson']['validation_metrics'].update(n_test=0),
    lambda c: c['models']['KNN'].update(multipliers=[True, 1, 1]),
    lambda c: c['models']['KNN'].update(multipliers=['1', 1, 1]),
    lambda c: c.update(models=['KNN', 'Random Forest', 'Poisson']),
])
def test_invalid_config_rejected_before_training(tmp_path, change):
    path, config = inputs(tmp_path)
    change(config)
    with pytest.raises(ValueError):
        service_type()(path, config)


def test_train_exports_web_report_with_fixed_development_choices(tmp_path, monkeypatch):
    from football_predictor import advanced_selection as classifiers
    from football_predictor import advanced_poisson_selection as poisson
    from football_predictor.evaluation import evaluate_probabilities
    path, config = inputs(tmp_path)
    original_bytes = path.read_bytes()
    service = service_type()(path, config)
    # Keep real estimators, recording only the temporal boundaries of their fits.
    fits = []
    original_classifier, original_poisson = classifiers.fit_classifier_candidate, poisson.fit_poisson_candidate

    def fit_classifier(name, candidate, frame, indices, feature_sets):
        selected = frame.loc[indices]
        assert list(indices) == list(selected.sort_values(['date', 'home_team', 'away_team']).index)
        fits.append((name, len(indices), selected.date.max()))
        return original_classifier(name, candidate, frame, indices, feature_sets)

    def fit_poisson(candidate, frame):
        fits.append(('Poisson', len(frame), frame.date.max()))
        return original_poisson(candidate, frame)

    monkeypatch.setattr(classifiers, 'fit_classifier_candidate', fit_classifier)
    monkeypatch.setattr(poisson, 'fit_poisson_candidate', fit_poisson)
    progress = []
    report = service.train(progress.append)
    assert service.train() is report
    assert len(fits) == 6
    for name in config['models']:
        assert [(n, str(date.date())) for model, n, date in fits if model == name] == [
            (48, '2024-01-24'), (60, '2024-01-30')]
    assert len(report['metrics']) == 4
    assert report['class_order'] == ['home', 'draw', 'away']
    assert report['future_training_end'] == '2024-01-30'
    assert report['source_sha256'] == config['source_sha256']
    assert report['protocol_sha256'] == config['protocol_sha256']
    assert 'previously selected' in report['split']['protocol']
    assert report['split']['test']['n'] == 12
    assert report['split']['train']['end'] < report['split']['validation']['start']
    assert report['split']['validation']['end'] < report['split']['test']['start']
    assert len(report['tuning']) == 3
    for row in report['tuning']:
        assert row['validation_log_loss'] == 1.2
        assert row['params']['candidate'] == config['models'][row['model']]['candidate']
    assert set(report['poisson_optimization']) == {'selection', 'evaluation', 'future'}
    assert report['poisson_optimization']['evaluation']['trace']
    assert report['poisson_optimization']['future']['n_train'] == 60
    assert report['poisson_optimization']['selection']['status'] == 'previously_selected'
    paths = service.export_report(tmp_path / 'exports')
    assert set(paths) == {'metrics.csv', 'tuning.csv', 'predictions_test.csv', 'report.json'}
    from pathlib import Path
    saved = json.loads(Path(paths['report.json']).read_text())
    assert saved == report
    predictions = pd.read_csv(paths['predictions_test.csv'])
    assert len(predictions) == 48
    assert not predictions.duplicated(['date', 'home_team', 'away_team', 'model']).any()
    for row in report['metrics']:
        subset = predictions[predictions.model == row['model']]
        actual, confusion = evaluate_probabilities(row['model'], subset.actual,
                                                  subset[['p_home', 'p_draw', 'p_away']])
        for key in ('accuracy', 'macro_f1', 'log_loss', 'brier'):
            assert actual[key] == pytest.approx(row[key])
        assert confusion == report['confusion'][row['model']]
    assert progress[0] == 'Preparing chronological train / validation / test partitions'
    assert path.read_bytes() == original_bytes


@pytest.mark.parametrize('kind', ['rolling_poisson', 'dixon_coles'])
def test_future_prediction_uses_rich_features_and_one_weighted_score_distribution(tmp_path, kind):
    from football_predictor.accuracy_campaign import adjust_probabilities
    from football_predictor.advanced_features import FEATURE_SETS, fixture_advanced_features
    from football_predictor.advanced_selection import classifier_candidate_probabilities
    from football_predictor.models.poisson import historical_rates, score_distribution
    path, config = inputs(tmp_path)
    if kind == 'dixon_coles':
        config['models']['Poisson']['candidate'] = dict(id='small-dc', kind=kind,
            params=dict(half_life_days=None, ridge=.1, use_correction=True))
    original_bytes = path.read_bytes()
    service = service_type()(path, config)
    with pytest.raises(RuntimeError):
        service.predict('A', 'B', '2024-02-01')
    service.train()
    snapshot = service.matches.copy(deep=True)
    result = service.predict('A', 'B', '2024-02-01')
    assert result['home'] == 'A' and result['away'] == 'B'
    assert result['date'] == '2024-02-01'
    assert set(result['probabilities']) == set(config['models'])
    fixture = fixture_advanced_features(service.matches, 'A', 'B', '2024-02-01')
    for name in ('KNN', 'Random Forest'):
        setting = config['models'][name]
        raw = classifier_candidate_probabilities(service.models_[name], setting['candidate'], fixture, FEATURE_SETS)
        expected = adjust_probabilities(raw, setting['multipliers'])[0]
        assert result['probabilities'][name] == pytest.approx(expected)
    assert result['features']['home_form_20_gf'] == fixture.iloc[0].home_form_20_gf
    matrix = np.asarray(result['score_matrix'])
    assert matrix.sum() == pytest.approx(1)
    assert np.isfinite(matrix).all() and (matrix >= 0).all()
    outcomes = [np.tril(matrix, -1).sum(), np.trace(matrix), np.triu(matrix, 1).sum()]
    assert result['probabilities']['Poisson'] == pytest.approx(outcomes)
    home_grid, away_grid = np.indices(matrix.shape)
    assert result['expected_goals'] == pytest.approx([(home_grid * matrix).sum(), (away_grid * matrix).sum()])
    future = pd.DataFrame([dict(date=pd.Timestamp('2024-02-01'), season=2024,
                               home_team='A', away_team='B', home_goals=np.nan, away_goals=np.nan)])
    if kind == 'dixon_coles':
        rates = service.poisson_.expected_goals(future)[0]
        original, raw_outcomes, tail = service.poisson_.score_distribution(*rates)
    else:
        history = pd.concat([service.matches, future], ignore_index=True)
        rates = service.poisson_.expected_goals(historical_rates(history, window=5, smoothing=3.).iloc[[-1]])[0]
        original, raw_outcomes, tail = score_distribution(*rates)
    weighted = original * np.where(home_grid > away_grid, 1, np.where(home_grid == away_grid, 1.4, .7))
    weighted /= weighted.sum()
    assert matrix == pytest.approx(weighted)
    assert not np.allclose(matrix, original)
    assert result['probabilities']['Poisson'] == pytest.approx(adjust_probabilities([raw_outcomes], [1, 1.4, .7])[0])
    assert result['intensity_rates'] == pytest.approx(rates)
    assert result['score_tail'] == pytest.approx(tail)
    assert len(result['scores']) == 10
    order = np.argsort(-matrix.ravel(), kind='stable')[:10]
    for row, index in zip(result['scores'], order):
        assert row['score'] == f'{index // matrix.shape[1]}–{index % matrix.shape[1]}'
        assert row['probability'] == pytest.approx(matrix.ravel()[index])
    json.dumps(result, allow_nan=False)
    pd.testing.assert_frame_equal(service.matches, snapshot)
    assert path.read_bytes() == original_bytes
    for home, away, date in [('A', 'A', '2024-02-01'), ('A', 'Unknown', '2024-02-01'),
                             ('A', 'B', '2024-01-30'), ('A', 'B', '2024-01-10'),
                             ('A', 'B', 'NaT'), ('A', 'B', '2024-02-01T12:00:00Z')]:
        with pytest.raises(ValueError):
            service.predict(home, away, date)


@pytest.mark.parametrize('operation', ['train', 'cached_train', 'predict', 'export'])
def test_changed_csv_is_rejected_not_silently_reused(tmp_path, operation):
    path, config = inputs(tmp_path)
    service = service_type()(path, config)
    if operation != 'train':
        service.train()
    path.write_bytes(path.read_bytes() + b'\n')
    with pytest.raises(ValueError, match='changed|hash'):
        if operation in ('train', 'cached_train'):
            service.train()
        elif operation == 'predict':
            service.predict('A', 'B', '2024-02-01')
        else:
            service.export_report(tmp_path / 'export')


def test_config_is_detached_and_export_cannot_overwrite_source(tmp_path):
    path, config = inputs(tmp_path)
    target = path.with_name('metrics.csv')
    path.rename(target)
    service = service_type()(target, config)
    config['models']['Poisson']['multipliers'][1] = 999
    config['models']['KNN']['candidate']['params']['n_neighbors'] = 999
    report = service.train()
    assert report['selected_params']['KNN']['candidate']['params']['n_neighbors'] == 3
    assert report['selected_params']['Poisson']['multipliers'] == [1, 1.4, .7]
    original = target.read_bytes()
    with pytest.raises(ValueError, match='overwrite|source'):
        service.export_report(tmp_path)
    assert target.read_bytes() == original


def test_source_change_during_training_cannot_publish_report(tmp_path):
    path, config = inputs(tmp_path)
    service = service_type()(path, config)

    def change_source(message):
        if message == 'Refitting all-history models for strictly future predictions':
            path.write_bytes(path.read_bytes() + b'\n')

    with pytest.raises(ValueError, match='changed|hash'):
        service.train(change_source)
    assert service.report is None


def test_explicit_adapter_factory_works_through_real_web_jobs(tmp_path):
    import re
    import time
    from io import BytesIO
    from zipfile import ZipFile
    from football_predictor.web import create_app
    path, config = inputs(tmp_path)
    application = create_app(path, tmp_path / 'web', lambda csv: service_type()(csv, config))
    application.config['TESTING'] = True
    client = application.test_client()
    token = re.search(r'name="csrf-token" content="([^"]+)"', client.get('/').text).group(1)

    def job(endpoint, payload):
        response = client.post(endpoint, json=payload, headers={'X-CSRF-Token': token})
        assert response.status_code == 202, response.json
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            state = client.get('/api/jobs/' + response.json['job_id']).json
            if state['status'] in ('complete', 'error'):
                assert state['status'] == 'complete', state
                return state['result']
            time.sleep(.01)
        pytest.fail('Real web job timed out')

    try:
        assert client.get('/api/state').json['trained'] is False
        report = job('/api/train', {})
        assert client.get('/api/report').json == report
        assert client.get('/api/state').json['trained'] is True
        prediction = job('/api/predict', dict(home='A', away='B', date='2024-02-01'))
        assert set(prediction['probabilities']) == {'KNN', 'Random Forest', 'Poisson'}
        export = job('/api/export', {})
        download = next(row['url'] for row in export['files'] if row['name'] == 'report.zip')
        bundle = client.get(download)
        assert bundle.status_code == 200
        with ZipFile(BytesIO(bundle.data)) as archive:
            assert json.loads(archive.read('report.json')) == report
            assert len(pd.read_csv(BytesIO(archive.read('predictions_test.csv')))) == 48
    finally:
        application.extensions['workspace'].close()
