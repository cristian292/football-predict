"""Development-only Poisson selection; no test-set argument or fitted artifacts."""
import json
import warnings
from itertools import product
from numbers import Integral

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss

from .models.poisson import PoissonModel, historical_rates


def poisson_candidates():
    """Return the frozen 20 rolling + 24 Dixon-Coles JSON-safe configurations."""
    candidates = []
    for window, smoothing in product((5, 10, 20, 40, 60), (3., 5., 10., 20.)):
        candidates.append(dict(id=f'rolling-{len(candidates):03d}', kind='rolling_poisson',
                               params=dict(window=window, smoothing=smoothing)))
    for half_life, ridge, correction in product((180., 365., 730., None), (.01, .1, 1.), (False, True)):
        candidates.append(dict(id=f'dixon-coles-{len(candidates) - 20:03d}', kind='dixon_coles',
                               params=dict(half_life_days=half_life, ridge=ridge, use_correction=correction)))
    return candidates


def fit_poisson_candidate(candidate, train_frame):
    """Fit goal likelihood using training scores only."""
    candidate = _candidates([candidate])[0]
    train_frame = _dated_frame(train_frame)
    _indices(train_frame, train_frame.index)
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter('always', RuntimeWarning)
        if candidate['kind'] == 'dixon_coles':
            from .models.dixon_coles import DixonColesModel
            model = DixonColesModel(**candidate['params']).fit(train_frame)
        else:
            rates = historical_rates(train_frame, **candidate['params'])
            model = PoissonModel().fit(rates, train_frame[['home_goals', 'away_goals']])
    diagnostics = dict(model.optimization_)
    if emitted:
        diagnostics['warnings'] = [str(w.message) for w in emitted]
    reason = None
    if not np.isfinite(model.params_).all() or not np.isfinite(diagnostics['final_nll']):
        reason = 'Fit produced nonfinite parameters or likelihood.'
    elif candidate['kind'] == 'dixon_coles' and not diagnostics.get('success', False):
        reason = f'Dixon-Coles optimization did not converge: {diagnostics.get("message", "")}'
    if reason:
        error = RuntimeError(reason)
        error.optimization = _json_safe(diagnostics)
        raise error
    return model


def poisson_candidate_probabilities(model, candidate, history_frame, indices):
    """Return home/draw/away probabilities using only earlier-date history."""
    candidate = _candidates([candidate])[0]
    history_frame = _dated_frame(history_frame)
    indices = _indices(history_frame, indices)
    if candidate['kind'] == 'dixon_coles':
        return model.predict_proba(history_frame.loc[indices])
    rates = _prediction_rates(candidate, history_frame, indices)
    return model.predict_proba(rates.loc[indices])


def _prediction_rates(candidate, history_frame, indices):
    end = history_frame.loc[indices, 'date'].max()
    history = history_frame.loc[history_frame.date <= end].copy()
    # No prediction needs the final day's scores; never consume them or later scores.
    history.loc[history.date == end, ['home_goals', 'away_goals']] = np.nan
    return historical_rates(history, **candidate['params'])


def _json_safe(value):
    """Retain failed-fit diagnostics without emitting invalid JSON numbers."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _probabilities(raw, count):
    probabilities = np.asarray(raw, dtype=float)
    if probabilities.shape != (count, 3) or not np.isfinite(probabilities).all():
        raise ValueError('Expected finite Nx3 home/draw/away probabilities.')
    if ((probabilities < 0).any() or (probabilities > 1).any()
            or not np.allclose(probabilities.sum(axis=1), 1., atol=1e-8, rtol=0.)):
        raise ValueError('Probabilities must be nonnegative and sum to one.')
    return probabilities


def _metrics(truth, probabilities):
    predicted = probabilities.argmax(axis=1)
    return dict(validation_accuracy=float(accuracy_score(truth, predicted)),
                validation_macro_f1=float(f1_score(truth, predicted, labels=[0, 1, 2],
                                                  average='macro', zero_division=0)),
                validation_log_loss=float(log_loss(truth, probabilities, labels=[0, 1, 2])),
                n_validation=len(truth))


def _dated_frame(frame):
    if not frame.index.is_unique or any(not isinstance(i, Integral) or isinstance(i, bool) for i in frame.index):
        raise ValueError('Frame index must contain unique integer labels.')
    if 'date' not in frame:
        raise ValueError('Valid timezone-naive fixture dates required.')
    try:
        dates = pd.to_datetime(frame['date'])
        if dates.isna().any() or dates.dt.tz is not None:
            raise ValueError('Valid timezone-naive fixture dates required.')
        dates = dates.dt.normalize()
    except (TypeError, AttributeError) as error:
        raise ValueError('Valid timezone-naive fixture dates required.') from error
    return frame.assign(date=dates)


def _indices(frame, indices):
    indices = list(indices)
    if (not indices or any(not isinstance(i, Integral) or isinstance(i, bool) for i in indices)
            or not pd.Index(indices).is_unique or not pd.Index(indices).isin(frame.index).all()):
        raise ValueError('Partition indices must be nonempty unique integer labels present in frame index.')
    return indices


def _candidates(candidates):
    try:
        candidates = json.loads(json.dumps(list(candidates), allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError('Candidates must be finite JSON configurations.') from error
    if not candidates:
        raise ValueError('At least one candidate required.')
    seen = set()
    for candidate in candidates:
        if (not isinstance(candidate, dict) or not isinstance(candidate.get('id'), str)
                or not candidate['id'] or candidate['id'] in seen
                or candidate.get('kind') not in ('rolling_poisson', 'dixon_coles')
                or not isinstance(candidate.get('params'), dict)):
            raise ValueError('Candidates require unique string IDs, supported kind and parameter dictionaries.')
        seen.add(candidate['id'])
    return sorted(candidates, key=lambda c: c['id'])


def select_advanced_poisson(frame, folds, candidates=None, progress=None):
    """Rank pooled validation accuracy, macro-F1, log loss, then candidate ID."""
    frame = _dated_frame(frame)
    folds = [(_indices(frame, train), _indices(frame, valid)) for train, valid in folds]
    if not folds:
        raise ValueError('At least one nonempty chronological fold required.')
    seen_validation = set()
    for train, valid in folds:
        if set(train).intersection(valid) or frame.loc[train, 'date'].max() >= frame.loc[valid, 'date'].min():
            raise ValueError('Disjoint train dates must strictly precede validation dates (whole-date chronology).')
        if seen_validation.intersection(valid):
            raise ValueError('Validation indices cannot repeat across folds.')
        if 'result' not in frame or not frame.loc[valid, 'result'].isin([0, 1, 2]).all():
            raise ValueError('Result labels must be 0, 1 or 2 on validation rows.')
        seen_validation.update(valid)
    candidates = _candidates(poisson_candidates() if candidates is None else candidates)
    trials, selected, winner_records, best_key = [], None, [], None
    for candidate in candidates:
        if progress is not None:
            progress(f'Tuning Poisson: {candidate["id"]}')
        truths, predictions, records, fold_metrics = [], [], [], []
        number, model = None, None
        try:
            rates = (_prediction_rates(candidate, frame, list(seen_validation))
                     if candidate['kind'] == 'rolling_poisson' else None)
            for number, (train, valid) in enumerate(folds):
                model = None
                model = fit_poisson_candidate(candidate, frame.loc[train])
                p = _probabilities(model.predict_proba(rates.loc[valid] if rates is not None else frame.loc[valid]), len(valid))
                truth = frame.loc[valid, 'result'].to_numpy(dtype=int)
                truths.append(truth)
                predictions.append(p)
                fold_metrics.append(dict(fold=number, status='ok', optimization=_json_safe(model.optimization_), **_metrics(truth, p)))
                records.extend(dict(index=int(index), actual=int(actual), p_home=float(row[0]),
                                    p_draw=float(row[1]), p_away=float(row[2]))
                               for index, actual, row in zip(valid, truth, p))
        except Exception as error:
            diagnostic = dict(fold=number, status='error', error=f'{type(error).__name__}: {error}')
            optimization = getattr(error, 'optimization', getattr(model, 'optimization_', None))
            if optimization is not None:
                diagnostic['optimization'] = _json_safe(optimization)
            fold_metrics.append(diagnostic)
            trials.append(dict(id=candidate['id'], candidate=candidate, status='error',
                               error=diagnostic['error'], fold_metrics=fold_metrics))
            continue
        metrics = _metrics(np.concatenate(truths), np.concatenate(predictions))
        trials.append(dict(id=candidate['id'], candidate=candidate, status='ok',
                           fold_metrics=fold_metrics, **metrics))
        key = (-metrics['validation_accuracy'], -metrics['validation_macro_f1'],
               metrics['validation_log_loss'], candidate['id'])
        if best_key is None or key < best_key:
            best_key, selected, winner_records = key, candidate, records
    return dict(selected=selected, trials=trials, oof_records=winner_records,
                selection_metric='pooled_validation_accuracy')
