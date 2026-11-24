"""Bounded, deterministic classifier candidates; never tune against a test set.

The fixed grid contains 155 KNN and 79 forest configurations. Feature generation
must itself be pre-match; this module only controls fitting/selection chronology.
"""
import json
from itertools import product
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_advanced_classifier(name, params):
    """Build train-only median preprocessing and a single-CPU estimator."""
    params = dict(params)
    steps = [('imputer', SimpleImputer(strategy='median', keep_empty_features=True))]
    if name == 'KNN':
        params['n_jobs'] = 1
        steps.extend([('scaler', StandardScaler()), ('classifier', KNeighborsClassifier(**params))])
    elif name == 'Random Forest':
        options = dict(n_estimators=100, random_state=42, n_jobs=1)
        options.update(params)
        options.update(random_state=42, n_jobs=1)
        steps.append(('classifier', RandomForestClassifier(**options)))
    else:
        raise ValueError(f'Unknown classifier: {name}')
    return Pipeline(steps)


def _partition(frame, indices):
    """Resolve unique labels and validate only the requested development rows."""
    indices = list(indices)
    if not frame.index.is_unique:
        raise ValueError('Frame index must be unique.')
    if not indices or not pd.Index(indices).is_unique or not pd.Index(indices).isin(frame.index).all():
        raise ValueError('Partition indices must be nonempty unique labels present in frame index.')
    part = frame.loc[indices]
    if 'date' not in part or 'result' not in part:
        raise ValueError('Partitions require date and result columns.')
    try:
        dates = pd.to_datetime(part['date'])
        if dates.isna().any() or dates.dt.tz is not None:
            raise ValueError('Dates must be valid and timezone-naive.')
        dates = dates.dt.normalize()
    except (TypeError, AttributeError) as error:
        raise ValueError('Dates must be valid and timezone-naive.') from error
    if not part['result'].isin([0, 1, 2]).all():
        raise ValueError('Result labels must be 0, 1 or 2.')
    return part, dates


def _smoothing(candidate):
    alpha = candidate.get('probability_smoothing', 0.)
    if isinstance(alpha, bool) or not isinstance(alpha, Real) or not np.isfinite(alpha) or not 0 <= alpha < 1:
        raise ValueError('probability_smoothing must be a finite number in [0, 1).')
    return float(alpha)


def _positive_days(value, field):
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or value <= 0:
        raise ValueError(f'{field} must be a positive finite number of days.')


def _feature_columns(candidate, feature_sets):
    columns = list(feature_sets[candidate['feature_set']])
    if {'result', 'home_goals', 'away_goals'}.intersection(columns):
        raise ValueError('Post-match outcome columns cannot be classifier features.')
    return columns


def fit_classifier_candidate(name, candidate, frame, train_indices, feature_sets):
    """Fit preprocessing and classifier on index *labels*, optionally trailing days.

    The inclusive window cutoff is training-end minus ``train_window_days``;
    neither validation dates nor their labels determine the training window.
    """
    _smoothing(candidate)
    train, dates = _partition(frame, train_indices)
    end = dates.max()
    window = candidate.get('train_window_days')
    if window is not None:
        _positive_days(window, 'train_window_days')
        train = train.loc[dates >= end - pd.Timedelta(days=window)]
    params = dict(candidate['params'])
    if name == 'KNN':
        params['n_neighbors'] = min(params.get('n_neighbors', 5), len(train))
    model = make_advanced_classifier(name, params)
    fit_options = {}
    half_life = candidate.get('sample_half_life_days')
    if half_life is not None:
        _positive_days(half_life, 'sample_half_life_days')
        if name != 'Random Forest':
            raise ValueError('sample_half_life_days is supported only for Random Forest.')
        ages = (end - pd.to_datetime(train['date']).dt.normalize()).dt.days.to_numpy()
        fit_options['classifier__sample_weight'] = np.exp2(-ages / half_life)
    model.fit(train[_feature_columns(candidate, feature_sets)], train['result'], **fit_options)
    model.training_metadata_ = dict(n_train=len(train), train_end=end.strftime('%Y-%m-%d'))
    return model


def classifier_candidate_probabilities(model, candidate, xframe, feature_sets):
    """Return Nx3 probabilities in home/draw/away order (class 0/1/2).

    ``probability_smoothing`` is a convex mixture with the uniform prior,
    constrained to [0,1); it cannot improve argmax accuracy by itself.
    """
    raw = model.predict_proba(xframe[_feature_columns(candidate, feature_sets)])
    probabilities = np.zeros((len(xframe), 3), dtype=float)
    probabilities[:, np.asarray(model.classes_, dtype=int)] = raw
    alpha = _smoothing(candidate)
    return (1. - alpha) * probabilities + alpha / 3.


def _validation_metrics(truth, probabilities):
    predictions = probabilities.argmax(axis=1)
    return dict(validation_accuracy=float(accuracy_score(truth, predictions)),
                validation_macro_f1=float(f1_score(truth, predictions, labels=[0, 1, 2],
                                                    average='macro', zero_division=0)),
                validation_log_loss=float(log_loss(truth, probabilities, labels=[0, 1, 2])),
                n_validation=len(truth))


def select_advanced_classifier(name, frame, feature_sets, folds, candidates=None, progress=None):
    """Select by pooled validation accuracy, macro-F1, log loss, then candidate ID.

    ``folds`` are (training-index-labels, validation-index-labels) pairs. No test
    set is accepted. Return ``selected`` (a candidate), all ``trials``, and only
    the selected candidate's ``oof_records``; no fitted estimator is serialized.
    Validation folds must be ordered with disjoint whole dates. Invalid split
    definitions raise ValueError before fitting; individual candidate failures
    retain errors and completed fold metrics. If every candidate fails, selected
    is None and oof_records is empty. Candidate definitions must be JSON-safe.
    """
    if name not in ('KNN', 'Random Forest'):
        raise ValueError(f'Unknown classifier: {name}')
    candidates = classifier_candidates(name) if candidates is None else candidates
    candidates = json.loads(json.dumps(list(candidates), allow_nan=False))
    ids = [c.get('id') for c in candidates]
    if not ids or any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError('Candidates require nonempty unique string IDs.')
    candidates.sort(key=lambda c: c['id'])
    folds = [(list(train), list(validation)) for train, validation in folds]
    if not folds:
        raise ValueError('At least one nonempty chronological fold required.')
    seen_validation = set()
    previous_validation_end = None
    for train, validation in folds:
        _, train_dates = _partition(frame, train)
        _, validation_dates = _partition(frame, validation)
        if set(train) & set(validation) or train_dates.max() >= validation_dates.min():
            raise ValueError('Disjoint train dates must strictly precede validation dates (whole-date chronology).')
        if seen_validation.intersection(validation):
            raise ValueError('Validation indices cannot repeat across folds.')
        if previous_validation_end is not None and validation_dates.min() <= previous_validation_end:
            raise ValueError('Validation folds must be ordered and disjoint on whole dates.')
        seen_validation.update(validation)
        previous_validation_end = validation_dates.max()
    trials, winner_records, best_key, selected = [], [], None, None
    for config in candidates:
        if progress is not None:
            progress(f'Tuning {name}: {config["id"]}')
        records, truths, probabilities, fold_metrics = [], [], [], []
        for fold_number, (train, validation) in enumerate(folds):
            try:
                model = fit_classifier_candidate(name, config, frame, train, feature_sets)
                part = frame.loc[validation]
                p = classifier_candidate_probabilities(model, config, part, feature_sets)
                truth = part['result'].to_numpy(dtype=int)
                metrics = _validation_metrics(truth, p)
                truths.append(truth)
                probabilities.append(p)
                fold_metrics.append(dict(fold=fold_number, **model.training_metadata_, **metrics))
                for index, date, actual, probability in zip(part.index, part['date'], truth, p):
                    if isinstance(index, pd.Timestamp):
                        index = index.isoformat()
                    elif isinstance(index, np.generic):
                        index = index.item()
                    records.append(dict(fold=fold_number, index=index,
                                        date=pd.Timestamp(date).strftime('%Y-%m-%d'), actual=int(actual),
                                        predicted=int(probability.argmax()), p_home=float(probability[0]),
                                        p_draw=float(probability[1]), p_away=float(probability[2])))
            except Exception as error:
                # Failed candidates cannot compete on a successful subset of folds.
                trials.append(dict(id=config['id'], candidate=config, model=name, status='failed',
                                   failed_fold=fold_number, fold_metrics=fold_metrics,
                                   error=f'{type(error).__name__}: {error}'))
                break
        else:
            metrics = _validation_metrics(np.concatenate(truths), np.concatenate(probabilities))
            trials.append(dict(id=config['id'], candidate=config, model=name, status='ok',
                               fold_metrics=fold_metrics, **metrics))
            key = (-metrics['validation_accuracy'], -metrics['validation_macro_f1'],
                   metrics['validation_log_loss'], config['id'])
            if best_key is None or key < best_key:
                best_key, selected, winner_records = key, config, records
    return dict(selected=selected, trials=trials, oof_records=winner_records)


_FEATURE_FAMILIES = ('legacy_basic', 'legacy_complex', 'form', 'strength', 'extended')


def classifier_candidates(name):
    """Return a fresh JSON-safe fixed budget, including all web legacy settings.

    KNN: 120 full-history Minkowski, 4 legacy k=3, 20 recency,
    6 compact-family cosine, 5 uniform probability smoothing configurations.
    Forest: 45 depth/leaf, 9 legacy all-feature, 5 balanced, 10 recency, 10 decay.
    Uniform smoothing only changes probabilities, not classification argmax.
    """
    if name not in ('KNN', 'Random Forest'):
        raise ValueError(f'Unknown classifier: {name}')
    candidates = []

    def add(family, params, window=None, **extra):
        candidates.append(dict(id=f'{"knn" if name == "KNN" else "rf"}-{len(candidates):03d}',
                               feature_set=family, params=params,
                               train_window_days=window, **extra))

    if name == 'KNN':
        for family, k, weights, p in product(_FEATURE_FAMILIES, (7, 15, 31, 61, 101, 151),
                                              ('uniform', 'distance'), (1, 2)):
            add(family, dict(n_neighbors=k, weights=weights, p=p))
        for weights, p in product(('uniform', 'distance'), (1, 2)):
            add('legacy_basic', dict(n_neighbors=3, weights=weights, p=p))
        for family, window, k in product(_FEATURE_FAMILIES, (730, 1095), (31, 101)):
            add(family, dict(n_neighbors=k, weights='distance', p=2), window)
        for family, k in product(('legacy_basic', 'form', 'strength'), (31, 101)):
            add(family, dict(n_neighbors=k, weights='distance', metric='cosine', algorithm='brute'))
        for family in _FEATURE_FAMILIES:
            add(family, dict(n_neighbors=31, weights='distance', p=2), probability_smoothing=0.03)
    else:
        def forest(depth: int | None = 10, leaf: int = 5, **extra) -> dict[str, Any]:
            return dict(n_estimators=100, max_depth=depth, min_samples_leaf=leaf,
                        max_features='sqrt', **extra)
        for family, depth, leaf in product(_FEATURE_FAMILIES, (5, 10, None), (2, 5, 10)):
            add(family, forest(depth, leaf))
        for depth, leaf in product((5, 10, None), (2, 5, 10)):
            params = forest(depth, leaf)
            params['max_features'] = 1.0
            add('legacy_complex', params)
        for family in _FEATURE_FAMILIES:
            add(family, forest(None, class_weight='balanced'))
        for family, window in product(_FEATURE_FAMILIES, (730, 1095)):
            add(family, forest(), window)
        for family, half_life in product(_FEATURE_FAMILIES, (365, 730)):
            add(family, forest(), sample_half_life_days=half_life)
    return candidates
