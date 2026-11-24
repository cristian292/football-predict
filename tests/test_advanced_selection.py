"""Chronological classifier search contracts (small real-estimator fixtures)."""
import json

import numpy as np
import pandas as pd
import pytest


def tiny_frame():
    return pd.DataFrame({
        'date': pd.to_datetime(['2018-01-01', '2020-01-01', '2021-01-01', '2022-01-01', '2023-01-01']),
        'x': [999., 1., np.nan, 3., 100000.],
        'result': [2, 0, 0, 1, 2],
    }, index=['old', 'a', 'b', 'c', 'future'])


def candidate(identifier='one', **extra):
    return dict(id=identifier, feature_set='form', params={'n_neighbors': 151, 'weights': 'distance'},
                train_window_days=None, **extra)


def test_fit_predict_uses_only_trailing_training_labels_and_aligns_classes():
    from football_predictor.advanced_selection import (
        fit_classifier_candidate, classifier_candidate_probabilities,
    )

    frame = tiny_frame()
    config = candidate(probability_smoothing=0.06)
    config['train_window_days'] = 730
    model = fit_classifier_candidate('KNN', config, frame, ['old', 'a', 'b', 'c'], {'form': ['x']})
    assert model.named_steps['classifier'].n_neighbors == 2
    assert model.named_steps['imputer'].statistics_[0] == 3.
    assert model.training_metadata_['n_train'] == 2
    assert model.training_metadata_['train_end'] == '2022-01-01'
    p = classifier_candidate_probabilities(model, config, frame.loc[['future']], {'form': ['x']})
    assert p.shape == (1, 3)
    assert p[0, 2] == pytest.approx(0.02)
    assert p.sum() == pytest.approx(1.)
    assert config['params']['n_neighbors'] == 151  # no caller mutation
    altered = frame.copy()
    altered.loc['future', ['x', 'result', 'date']] = [np.nan, 1, pd.Timestamp('2099-01-01')]
    again = fit_classifier_candidate('KNN', config, altered, ['old', 'a', 'b', 'c'], {'form': ['x']})
    assert np.array_equal(model.named_steps['imputer'].statistics_, again.named_steps['imputer'].statistics_)


def test_forest_decay_matches_explicit_training_end_weights():
    from football_predictor.advanced_selection import fit_classifier_candidate, make_advanced_classifier

    frame = tiny_frame().loc[['old', 'a', 'b', 'c']].copy()
    frame['x'] = [0., 0., 0., 0.]
    config = candidate(sample_half_life_days=365)
    config['params'] = {'n_estimators': 9, 'max_depth': 1, 'bootstrap': False}
    model = fit_classifier_candidate('Random Forest', config, frame, frame.index, {'form': ['x']})
    weights = np.exp2(-(frame.date.max() - frame.date).dt.days.to_numpy() / 365.)
    explicit = make_advanced_classifier('Random Forest', config['params']).fit(
        frame[['x']], frame.result, classifier__sample_weight=weights)
    assert np.array_equal(model.predict_proba(frame[['x']]), explicit.predict_proba(frame[['x']]))
    unweighted = make_advanced_classifier('Random Forest', config['params']).fit(frame[['x']], frame.result)
    assert not np.array_equal(model.predict_proba(frame[['x']]), unweighted.predict_proba(frame[['x']]))


@pytest.mark.parametrize('name', ['KNN', 'Random Forest'])
def test_pipeline_imputes_and_has_reproducible_single_cpu_estimator(name):
    from football_predictor.advanced_selection import make_advanced_classifier
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    params = {'n_neighbors': 1} if name == 'KNN' else {'n_estimators': 7, 'max_depth': 2}
    model = make_advanced_classifier(name, params)
    assert isinstance(model, Pipeline)
    assert isinstance(model.named_steps['imputer'], SimpleImputer)
    assert model.named_steps['imputer'].strategy == 'median'
    if name == 'KNN':
        assert isinstance(model.named_steps['scaler'], StandardScaler)
    else:
        assert 'scaler' not in model.named_steps
        assert model.named_steps['classifier'].n_estimators == 7
        assert model.named_steps['classifier'].random_state == 42
    assert model.named_steps['classifier'].n_jobs == 1
    x = pd.DataFrame({'x': [0., 1., np.nan, 3.], 'missing': [np.nan] * 4})
    y = [0, 1, 2, 0]
    p = model.fit(x, y).predict_proba(x)
    assert np.isfinite(p).all()
    assert np.array_equal(p, make_advanced_classifier(name, params).fit(x, y).predict_proba(x))


def selection_frame():
    return pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=16),
        'x': [0., .1, .2, 10., 10.1, 10.2, 20., 20.1, 20.2, 0., 10., 20., 0., 10., 20., -999.],
        'result': [0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 1, 2, 0, 1, 0, 2],
    }, index=np.arange(16) * 7 + 100)


def selection_candidates():
    hard = candidate('z-hard')
    hard['params'] = {'n_neighbors': 1}
    smooth = candidate('b-smooth', probability_smoothing=0.00003)
    smooth['params'] = {'n_neighbors': 1}
    majority = candidate('a-majority')
    majority['params'] = {'n_neighbors': 151}
    return [majority, hard, smooth]


def test_selector_pools_validation_accuracy_before_f1_loss_and_is_test_independent():
    from football_predictor.advanced_selection import select_advanced_classifier
    from sklearn.metrics import accuracy_score, f1_score, log_loss

    frame = selection_frame()
    folds = [(frame.index[:9], frame.index[9:11]), (frame.index[:9], frame.index[11:15])]
    messages = []
    result = select_advanced_classifier('KNN', frame, {'form': ['x']}, folds,
                                        candidates=selection_candidates(), progress=messages.append)
    assert result['selected']['id'] == 'b-smooth'
    trials = {trial['id']: trial for trial in result['trials']}
    assert trials['a-majority']['validation_log_loss'] < trials['b-smooth']['validation_log_loss']
    assert trials['a-majority']['validation_accuracy'] < trials['b-smooth']['validation_accuracy']
    records = result['oof_records']
    assert len(records) == 6
    p = np.array([[r['p_home'], r['p_draw'], r['p_away']] for r in records])
    truth = [r['actual'] for r in records]
    chosen = trials['b-smooth']
    assert chosen['validation_accuracy'] == pytest.approx(accuracy_score(truth, p.argmax(axis=1)))
    assert chosen['validation_accuracy'] == pytest.approx(5 / 6)
    assert chosen['validation_macro_f1'] == pytest.approx(f1_score(truth, p.argmax(axis=1), labels=[0, 1, 2], average='macro'))
    assert chosen['validation_log_loss'] == pytest.approx(log_loss(truth, p, labels=[0, 1, 2]))
    assert len(chosen['fold_metrics']) == 2
    assert messages
    json.dumps(result, allow_nan=False)
    frame.loc[frame.index[-1], ['date', 'x', 'result']] = [pd.Timestamp('2099-01-01'), np.nan, -999]
    assert result == select_advanced_classifier('KNN', frame, {'form': ['x']}, folds,
                                                candidates=list(reversed(selection_candidates())))


@pytest.mark.parametrize('case', ['same_date', 'reverse', 'overlap', 'unknown', 'duplicate_labels',
                                       'duplicate_frame_index', 'empty_train', 'empty_validation',
                                       'empty_folds', 'repeated_validation', 'invalid_date', 'invalid_result'])
def test_selector_rejects_ambiguous_or_nonchronological_folds(case):
    from football_predictor.advanced_selection import select_advanced_classifier

    frame = selection_frame()
    train, validation = list(frame.index[:9]), list(frame.index[9:15])
    folds = [(train, validation)]
    if case == 'same_date':
        frame.loc[validation[0], 'date'] = frame.loc[train[-1], 'date'] + pd.Timedelta(hours=12)
    elif case == 'reverse':
        folds = [(validation, train)]
    elif case == 'overlap':
        validation.append(train[-1])
    elif case == 'unknown':
        train.append(-777)
    elif case == 'duplicate_labels':
        train.append(train[0])
    elif case == 'duplicate_frame_index':
        frame.index = [0] * len(frame)
    elif case == 'empty_train':
        train.clear()
    elif case == 'empty_validation':
        validation.clear()
    elif case == 'empty_folds':
        folds = []
    elif case == 'repeated_validation':
        folds = folds * 2
    elif case == 'invalid_date':
        frame.loc[train[0], 'date'] = pd.NaT
    else:
        frame.loc[validation[0], 'result'] = 4
    with pytest.raises(ValueError):
        select_advanced_classifier('KNN', frame, {'form': ['x']}, folds, candidates=selection_candidates()[:1])


@pytest.mark.parametrize('fault', ['unknown_feature', 'missing_column', 'invalid_params', 'negative_window',
                                  'negative_smoothing', 'unit_smoothing', 'nan_smoothing', 'knn_decay'])
def test_candidate_failures_are_auditable_and_never_win(fault):
    from football_predictor.advanced_selection import select_advanced_classifier

    frame = selection_frame()
    bad = candidate('bad')
    features = {'form': ['x'], 'missing': ['absent']}
    if fault == 'unknown_feature':
        bad['feature_set'] = 'absent'
    elif fault == 'missing_column':
        bad['feature_set'] = 'missing'
    elif fault == 'invalid_params':
        bad['params'] = {'n_neighbors': 0}
    elif fault == 'negative_window':
        bad['train_window_days'] = -1
    elif fault == 'negative_smoothing':
        bad['probability_smoothing'] = -0.1
    elif fault == 'unit_smoothing':
        bad['probability_smoothing'] = 1.
    elif fault == 'nan_smoothing':
        bad['probability_smoothing'] = 'nan'
    else:
        bad['sample_half_life_days'] = 365
    folds = [(frame.index[:9], frame.index[9:15])]
    result = select_advanced_classifier('KNN', frame, features, folds, candidates=[bad, candidate('good')])
    assert result['selected']['id'] == 'good'
    failed = next(t for t in result['trials'] if t['id'] == 'bad')
    assert failed['status'] == 'failed'
    assert failed['error']
    assert 'validation_accuracy' not in failed
    assert failed['failed_fold'] == 0
    json.dumps(result, allow_nan=False)
    failed_only = select_advanced_classifier('KNN', frame, features, folds, candidates=[bad])
    assert failed_only['selected'] is None
    assert failed_only['oof_records'] == []
    assert len(failed_only['trials']) == 1


@pytest.mark.parametrize('column', ['result', 'home_goals', 'away_goals'])
def test_candidate_cannot_use_outcomes_as_features(column):
    from football_predictor.advanced_selection import select_advanced_classifier

    frame = selection_frame()
    frame['home_goals'] = 1
    frame['away_goals'] = 0
    result = select_advanced_classifier('KNN', frame, {'form': [column]},
                                      [(frame.index[:9], frame.index[9:15])], candidates=[candidate()])
    assert result['selected'] is None
    assert 'outcome' in result['trials'][0]['error'].lower()


@pytest.mark.parametrize('case', ['duplicate_ids', 'empty_candidates', 'unknown_name', 'split_validation_date',
                                 'boolean_indices'])
def test_search_rejects_ambiguous_search_definitions(case):
    from football_predictor.advanced_selection import select_advanced_classifier

    frame = selection_frame()
    configs = [candidate()]
    name = 'KNN'
    folds = [(list(frame.index[:9]), list(frame.index[9:15]))]
    if case == 'duplicate_ids':
        configs = configs * 2
    elif case == 'empty_candidates':
        configs = []
    elif case == 'unknown_name':
        name = 'other'
    elif case == 'split_validation_date':
        frame.loc[frame.index[11], 'date'] = frame.loc[frame.index[10], 'date']
        folds = [(frame.index[:9], frame.index[9:11]), (frame.index[:9], frame.index[11:15])]
    else:
        frame.index = np.arange(len(frame))
        folds = [([True] * 9 + [False] * 7, list(frame.index[9:15]))]
    with pytest.raises(ValueError):
        select_advanced_classifier(name, frame, {'form': ['x']}, folds, candidates=configs)


def test_forest_selector_runs_real_recency_candidates_reproducibly():
    from football_predictor.advanced_selection import select_advanced_classifier

    frame = selection_frame()
    configs = [candidate('plain'), candidate('decay', sample_half_life_days=365)]
    for config in configs:
        config['params'] = {'n_estimators': 5, 'min_samples_leaf': 2, 'max_depth': 3}
    folds = [(frame.index[:9], frame.index[9:11]), (frame.index[:11], frame.index[11:15])]
    result = select_advanced_classifier('Random Forest', frame, {'form': ['x']}, folds, candidates=configs)
    assert all(trial['status'] == 'ok' for trial in result['trials'])
    assert len(result['oof_records']) == 6
    ranked = min(result['trials'], key=lambda t: (-t['validation_accuracy'], -t['validation_macro_f1'],
                                                 t['validation_log_loss'], t['id']))
    assert result['selected']['id'] == ranked['id']
    assert result == select_advanced_classifier('Random Forest', frame, {'form': ['x']}, folds, candidates=configs)
    json.dumps(result, allow_nan=False)


def test_accuracy_tie_uses_macro_f1_before_log_loss():
    from football_predictor.advanced_selection import select_advanced_classifier

    frame = selection_frame()
    frame['signal'] = frame['x']
    frame.loc[frame.index[9:15], 'signal'] = [0., 0., 10., 10., 10., 20.]
    frame.loc[frame.index[9:15], 'x'] = 0.
    frame.loc[frame.index[9:15], 'result'] = [0, 0, 0, 0, 1, 2]
    majority = candidate('a-majority', probability_smoothing=0.3)
    majority['params'] = {'n_neighbors': 1}
    diverse = candidate('z-diverse', probability_smoothing=0.00003)
    diverse['params'] = {'n_neighbors': 1}
    diverse['feature_set'] = 'strength'
    result = select_advanced_classifier('KNN', frame, {'form': ['x'], 'strength': ['signal']},
                                      [(frame.index[:9], frame.index[9:15])], candidates=[majority, diverse])
    first, second = result['trials']
    assert first['validation_accuracy'] == second['validation_accuracy']
    assert first['validation_macro_f1'] < second['validation_macro_f1']
    assert first['validation_log_loss'] < second['validation_log_loss']
    assert result['selected']['id'] == 'z-diverse'


def test_exact_metric_tie_uses_candidate_id_not_caller_order():
    from football_predictor.advanced_selection import select_advanced_classifier

    frame = selection_frame()
    result = select_advanced_classifier('KNN', frame, {'form': ['x']},
                                      [(frame.index[:9], frame.index[9:15])],
                                      candidates=[candidate('z'), candidate('a')])
    assert result['selected']['id'] == 'a'
    assert result['trials'][0]['validation_accuracy'] == result['trials'][1]['validation_accuracy']
    assert result['trials'][0]['validation_macro_f1'] == result['trials'][1]['validation_macro_f1']
    assert result['trials'][0]['validation_log_loss'] == result['trials'][1]['validation_log_loss']


def test_candidate_failing_a_later_fold_cannot_compete_on_partial_metrics():
    from football_predictor.advanced_selection import select_advanced_classifier

    frame = selection_frame()
    frame['broken'] = frame.x
    frame.loc[frame.index[11], 'broken'] = np.inf
    broken = candidate('a-broken')
    broken['feature_set'] = 'broken'
    result = select_advanced_classifier('KNN', frame, {'form': ['x'], 'broken': ['broken']},
                                      [(frame.index[:9], frame.index[9:11]),
                                       (frame.index[:9], frame.index[11:15])],
                                      candidates=[broken, candidate('good')])
    failed = result['trials'][0]
    assert failed['status'] == 'failed'
    assert failed['failed_fold'] == 1
    assert len(failed['fold_metrics']) == 1
    assert 'validation_accuracy' not in failed
    assert result['selected']['id'] == 'good'
    assert len(result['oof_records']) == 6


def test_datetime_index_labels_are_returned_as_json_safe_identifiers():
    from football_predictor.advanced_selection import select_advanced_classifier

    frame = selection_frame()
    frame.index = pd.date_range('2010-01-01', periods=len(frame))
    result = select_advanced_classifier('KNN', frame, {'form': ['x']},
                                      [(frame.index[:9], frame.index[9:15])], candidates=[candidate()])
    json.dumps(result, allow_nan=False)
    assert result['oof_records'][0]['index'] == frame.index[9].isoformat()


def test_knn_retains_entire_existing_benchmark_grid_on_legacy_schema():
    from football_predictor.advanced_selection import classifier_candidates

    configs = classifier_candidates('KNN')
    for k in (3, 7, 15, 31, 61, 101):
        for weights in ('uniform', 'distance'):
            for p in (1, 2):
                assert any(c['feature_set'] == 'legacy_basic' and c['train_window_days'] is None
                           and c['params']['n_neighbors'] == k and c['params']['weights'] == weights
                           and c['params'].get('p', 2) == p for c in configs)


def test_forest_retains_entire_existing_benchmark_grid_on_legacy_schema():
    from football_predictor.advanced_selection import classifier_candidates

    configs = classifier_candidates('Random Forest')
    for depth in (5, 10, None):
        for leaf in (2, 5, 10):
            for max_features in ('sqrt', 1.0):
                assert any(c['feature_set'] == 'legacy_complex' and c['train_window_days'] is None
                           and c['params']['max_depth'] == depth and c['params']['min_samples_leaf'] == leaf
                           and c['params']['max_features'] == max_features and c['params']['n_estimators'] == 100
                           for c in configs)


def test_fixed_candidate_grids_cover_features_legacy_and_recency():
    from football_predictor.advanced_selection import classifier_candidates

    families = {'legacy_basic', 'legacy_complex', 'form', 'strength', 'extended'}
    for name, limit in [('KNN', 200), ('Random Forest', 80)]:
        candidates = classifier_candidates(name)
        assert 10 < len(candidates) <= limit
        assert candidates == classifier_candidates(name)
        assert len({c['id'] for c in candidates}) == len(candidates)
        assert {c['feature_set'] for c in candidates} == families
        assert {c['train_window_days'] for c in candidates} == {None, 730, 1095}
        json.dumps(candidates, allow_nan=False)
    knn = classifier_candidates('KNN')
    legacy = [c['params'] for c in knn if c['feature_set'] == 'legacy_basic' and c['train_window_days'] is None]
    for k in (3, 7, 15, 31):
        for weights in ('uniform', 'distance'):
            assert any(p['n_neighbors'] == k and p['weights'] == weights and p.get('p', 2) == 2 for p in legacy)
    assert {7, 15, 31, 61, 101, 151} <= {c['params']['n_neighbors'] for c in knn}
    rf = classifier_candidates('Random Forest')
    for depth in (5, None):
        for leaf in (2, 5):
            assert any(c['feature_set'] == 'legacy_complex' and c['train_window_days'] is None
                       and c['params']['max_depth'] == depth and c['params']['min_samples_leaf'] == leaf
                       and c['params']['n_estimators'] == 100 for c in rf)
    assert {365, 730} <= {c.get('sample_half_life_days') for c in rf}
    with pytest.raises(ValueError, match='Unknown'):
        classifier_candidates('other')
