"""Deterministic hyperparameter selection using validation labels only."""
from itertools import product
from numbers import Integral
from typing import Any

import pandas as pd
from sklearn.metrics import log_loss

from .features import BASIC_FEATURES, COMPLEX_FEATURES
from .models import aligned_probabilities, make_classifier


def _validate_partitions(train, validation):
    if train.empty or validation.empty:
        raise ValueError('Nonempty train and validation partitions required.')
    for label, part in [('train', train), ('validation', validation)]:
        if not part.index.is_unique:
            raise ValueError(f'{label} index must be unique.')
        if 'date' not in part or part.date.isna().any():
            raise ValueError(f'{label} requires valid fixture dates.')
    if pd.to_datetime(train.date).dt.normalize().max() >= pd.to_datetime(validation.date).dt.normalize().min():
        raise ValueError('Train dates must strictly precede validation dates (whole-date chronology).')


def select_classifier(name, train, validation, progress=None) -> dict[str, Any]:
    """Select the first minimum validation log loss; never accepts a test set."""
    _validate_partitions(train, validation)
    if name == 'KNN':
        columns = list(BASIC_FEATURES)
        grid = [dict(n_neighbors=k, weights=w, p=p)
                for k, w, p in product(sorted({min(k, len(train)) for k in (3, 7, 15, 31, 61, 101)}),
                                       ('uniform', 'distance'), (1, 2))]
    elif name == 'Random Forest':
        columns = list(COMPLEX_FEATURES)
        grid = [dict(max_depth=d, min_samples_leaf=leaf, max_features=m)
                for d, leaf, m in product((5, 10, None), (2, 5, 10), ('sqrt', 1.0))]
    else:
        raise ValueError(f'Unknown classifier: {name}')
    trials = []
    for params in grid:
        if progress is not None:
            progress(f'Tuning {name}: {params}')
        model = make_classifier(name, params).fit(train[columns], train.result)
        probabilities = aligned_probabilities(model, validation[columns])
        trials.append(dict(model=name, params=dict(params), validation_log_loss=float(
            log_loss(validation.result, probabilities, labels=[0, 1, 2]))))
    return dict(params=dict(min(trials, key=lambda row: row['validation_log_loss'])['params']),
                columns=columns, trials=trials)


def select_poisson(features, train_indices, validation_indices, progress=None) -> dict[str, Any]:
    """Tune historical-rate settings; optimize goal likelihood on training only.

    Indices are integer *labels* into features, not positional offsets. Historical
    rates may walk forward through validation, but fitted coefficients never do.
    """
    from .models.poisson import PoissonModel, historical_rates

    if not features.index.is_unique or any(not isinstance(i, Integral) or isinstance(i, bool) for i in features.index):
        raise ValueError('Feature index must contain unique integer labels.')
    train_indices, validation_indices = list(train_indices), list(validation_indices)
    for indices in (train_indices, validation_indices):
        if (len(set(indices)) != len(indices)
                or any(not isinstance(i, Integral) or isinstance(i, bool) for i in indices)
                or not pd.Index(indices).isin(features.index).all()):
            raise ValueError('Partition indices must be unique integer labels present in feature index.')
    _validate_partitions(features.loc[train_indices], features.loc[validation_indices])
    trials, optimizations = [], []
    for window, smoothing in product((5, 10, 20), (3.0, 5.0, 10.0)):
        params = dict(window=window, smoothing=smoothing)
        if progress is not None:
            progress(f'Tuning Poisson: {params}')
        rates = historical_rates(features, window=window, smoothing=smoothing)
        model = PoissonModel().fit(rates.loc[train_indices],
                                   features.loc[train_indices, ['home_goals', 'away_goals']])
        score = log_loss(features.loc[validation_indices, 'result'],
                         model.predict_proba(rates.loc[validation_indices]), labels=[0, 1, 2])
        trials.append(dict(model='Poisson', params=params, validation_log_loss=float(score)))
        optimizations.append(model.optimization_)
    winner = min(range(len(trials)), key=lambda i: trials[i]['validation_log_loss'])
    return dict(params=dict(trials[winner]['params']), trials=trials,
                optimization=optimizations[winner])
