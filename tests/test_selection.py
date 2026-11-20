"""Validation-only selection contracts, exercised with real estimators."""
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import log_loss

from football_predictor.features import BASIC_FEATURES, COMPLEX_FEATURES, build_features
from football_predictor.models import aligned_probabilities, make_classifier


def classifier_parts():
    rng = np.random.default_rng(11)
    frame = pd.DataFrame(rng.normal(size=(42, len(COMPLEX_FEATURES))), columns=COMPLEX_FEATURES)
    frame['result'] = np.arange(42) % 3
    frame['date'] = pd.date_range('2020-01-01', periods=42)
    return frame.iloc[:30], frame.iloc[30:]


def test_knn_selects_minimum_validation_loss_with_legacy_candidates():
    from football_predictor.selection import select_classifier
    train, validation = classifier_parts()
    messages = []
    selected = select_classifier('KNN', train, validation, progress=messages.append)
    assert selected['columns'] == BASIC_FEATURES
    assert selected['params'] == min(selected['trials'], key=lambda x: x['validation_log_loss'])['params']
    expected = set((min(k, len(train)), w, 2) for k in (3, 7, 15, 31) for w in ('uniform', 'distance'))
    actual = {(t['params']['n_neighbors'], t['params']['weights'], t['params'].get('p', 2)) for t in selected['trials']}
    assert expected <= actual
    assert any(t['params'].get('p') == 1 for t in selected['trials'])
    assert all(1 <= t['params']['n_neighbors'] <= len(train) for t in selected['trials'])
    model = make_classifier('KNN', selected['params']).fit(train[BASIC_FEATURES], train.result)
    score = log_loss(validation.result, aligned_probabilities(model, validation[BASIC_FEATURES]), labels=[0, 1, 2])
    assert score == pytest.approx(min(t['validation_log_loss'] for t in selected['trials']))
    assert messages
    json.dumps(selected, allow_nan=False)


def test_forest_grid_is_bounded_reproducible_and_contains_entire_legacy_grid():
    from football_predictor.selection import select_classifier
    train, validation = classifier_parts()
    selected = select_classifier('Random Forest', train, validation)
    assert selected['columns'] == COMPLEX_FEATURES
    trials = selected['trials']
    assert 4 < len(trials) <= 24
    for depth, leaf in [(5, 2), (5, 5), (None, 2), (None, 5)]:
        assert any(t['params']['max_depth'] == depth and t['params']['min_samples_leaf'] == leaf
                   and t['params'].get('max_features', 'sqrt') == 'sqrt' for t in trials)
    assert all(t['model'] == 'Random Forest' and 'n_estimators' not in t['params'] for t in trials)
    legacy = [t['validation_log_loss'] for t in trials
              if t['params']['max_depth'] in (5, None) and t['params']['min_samples_leaf'] in (2, 5)
              and t['params'].get('max_features', 'sqrt') == 'sqrt']
    assert min(t['validation_log_loss'] for t in trials) <= min(legacy)
    assert selected == select_classifier('Random Forest', train, validation)


def goal_features():
    rows = [dict(date=pd.Timestamp('2020-01-01') + pd.Timedelta(days=i), season=2020,
                 home_team='ABCD'[i % 4], away_team='ABCD'[(i + 1) % 4],
                 home_goals=(i * 7) % 4, away_goals=(i * 5) % 3) for i in range(24)]
    frame = build_features(pd.DataFrame(rows))
    frame.index = np.arange(len(frame)) * 3 + 100  # Labels, not iloc offsets.
    return frame


def test_poisson_selects_validation_minimum_and_preserves_training_only_fit():
    from football_predictor.selection import select_poisson
    from football_predictor.models.poisson import PoissonModel, historical_rates
    frame = goal_features()
    train, validation = frame.index[:12], frame.index[12:18]
    messages = []
    selected = select_poisson(frame, train, validation, progress=messages.append)
    assert 6 <= len(selected['trials']) <= 9
    assert {'window': 10, 'smoothing': 5.0} in [t['params'] for t in selected['trials']]
    best = min(selected['trials'], key=lambda t: t['validation_log_loss'])
    assert selected['params'] == best['params']
    rates = historical_rates(frame, **selected['params'])
    model = PoissonModel().fit(rates.loc[train], frame.loc[train, ['home_goals', 'away_goals']])
    assert best['validation_log_loss'] == pytest.approx(log_loss(
        frame.loc[validation, 'result'], model.predict_proba(rates.loc[validation]), labels=[0, 1, 2]))
    assert selected['optimization']['n_train'] == len(train)
    altered = frame.copy()
    altered.loc[frame.index[18:], ['home_goals', 'away_goals', 'result']] = [10, 0, 0]
    assert selected == select_poisson(altered, train, validation)
    assert messages
    json.dumps(selected, allow_nan=False)


@pytest.mark.parametrize('case', ['empty', 'overlap', 'same_date', 'reverse'])
def test_classifier_rejects_invalid_temporal_partitions(case):
    from football_predictor.selection import select_classifier
    train, validation = (part.copy() for part in classifier_parts())
    if case == 'empty':
        train = train.iloc[:0]
    elif case == 'overlap':
        validation = train.iloc[-3:]
    elif case == 'same_date':
        validation.loc[:, 'date'] = train.date.max()
    else:
        train, validation = validation, train
    with pytest.raises(ValueError, match='train|validation|chronolog|date'):
        select_classifier('KNN', train, validation)


@pytest.mark.parametrize('case', ['duplicate', 'noninteger', 'overlap', 'unknown', 'empty', 'reverse'])
def test_poisson_rejects_ambiguous_or_non_temporal_index_labels(case):
    from football_predictor.selection import select_poisson
    frame = goal_features()
    train, validation = list(frame.index[:12]), list(frame.index[12:18])
    if case == 'duplicate':
        frame.index = [100] * len(frame)
    elif case == 'noninteger':
        frame.index = frame.index.astype(str)
        train, validation = list(frame.index[:12]), list(frame.index[12:18])
    elif case == 'overlap':
        validation = train[-3:]
    elif case == 'unknown':
        train = [-999]
    elif case == 'empty':
        validation = []
    else:
        train, validation = validation, train
    with pytest.raises(ValueError, match='index|indices|train|validation|chronolog|date'):
        select_poisson(frame, train, validation)
