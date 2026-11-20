"""Reusable diagnostics for labeled three-outcome football predictions.

Class order is fixed: 0 Home, 1 Draw, 2 Away. No training, prediction, or network
access happens here. Results describe the supplied evaluation rows only; they do
not correct source/season shifts or establish future betting performance.
"""


import json
import re
from numbers import Real
from pathlib import Path

import numpy as np
import pandas as pd


PREDICTION_COLUMNS = ['date', 'season', 'home_team', 'away_team', 'actual', 'model',
                      'variant', 'p_home', 'p_draw', 'p_away', 'predicted']
PROBABILITY_COLUMNS = ['p_home', 'p_draw', 'p_away']


def _validate(predictions):
    if not isinstance(predictions, pd.DataFrame):
        raise ValueError('predictions must be a pandas DataFrame with labeled actual outcomes.')
    missing = [name for name in PREDICTION_COLUMNS if name not in predictions]
    if missing:
        raise ValueError('Missing required prediction columns: ' + ', '.join(missing))
    if not predictions.columns.is_unique:
        raise ValueError('Prediction columns must be unique.')
    for name in ('actual', 'predicted', 'season'):
        values = predictions[name]
        valid = values.map(lambda x: isinstance(x, Real) and not isinstance(x, (bool, np.bool_))
                           and np.isfinite(x) and x == int(x))
        if not valid.all() or (name != 'season' and not values.isin([0, 1, 2]).all()):
            raise ValueError(f'{name} must contain non-null integers' +
                             (' in 0..2.' if name != 'season' else '.'))
    for name in ('date', 'home_team', 'away_team', 'variant', 'model'):
        if not predictions[name].map(lambda x: isinstance(x, str) and bool(x.strip())).all():
            raise ValueError(f'{name} must contain non-null, non-empty strings.')
    if not predictions['date'].map(lambda x: bool(re.fullmatch(r'\d{4}-\d{2}-\d{2}', x))).all():
        raise ValueError('date must use YYYY-MM-DD.')
    try:
        pd.to_datetime(predictions['date'], format='%Y-%m-%d', errors='raise')
    except (ValueError, TypeError) as exc:
        raise ValueError('date must contain valid YYYY-MM-DD dates.') from exc
    try:
        p = predictions[PROBABILITY_COLUMNS].to_numpy(dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError('probabilities must be numeric.') from exc
    if (not np.isfinite(p).all() or (p < 0).any() or (p > 1).any()
            or not np.allclose(p.sum(axis=1), 1., rtol=0., atol=1e-8)):
        raise ValueError('probabilities must be finite normalized three-class rows in [0, 1].')
    if len(predictions) and not np.array_equal(predictions['predicted'].to_numpy(), p.argmax(axis=1)):
        raise ValueError('predicted must match probability argmax (first class wins ties).')
    if predictions.duplicated(['variant', 'model', 'date', 'home_team', 'away_team']).any():
        raise ValueError('Duplicate fixture within a variant/model group.')


def build_diagnostics(predictions: pd.DataFrame) -> dict:
    """Build JSON-safe diagnostic tables without changing the input frame.

    Required columns are PREDICTION_COLUMNS, including observed ``actual`` labels;
    inference-only predictions are rejected. Extra metadata is retained on error
    rows, and missing optional metadata stays null rather than becoming a source
    label. Groups are independent (variant, model) pairs; model names are open.

    Calibration is *top-label* reliability, not classwise calibration: confidence
    is max(p_home, p_draw, p_away). Ten bins are [0, .1), ..., [.9, 1]; empty-bin
    means, accuracy and gaps are null. Gap = confidence - accuracy (positive is
    overconfidence). ECE is sum(count / n * abs(gap)), not evidence that all three
    probabilities are accurate. Precision/recall/F1 use zero for zero denominators.

    Team rows attribute the fixture outcome to both participating teams, including
    away appearances; they are not team-win rates. Log loss clips the probability
    of the observed outcome to [float64 epsilon, 1] to remain finite for zeroes.
    Every misclassified fixture is retained, globally sorted by descending maximum
    probability with stable ties. Expected-schema empty frames yield empty tables.
    """
    _validate(predictions)
    result = {key: [] for key in ('per_class', 'calibration', 'confident_errors',
                                 'by_team', 'by_season', 'summary')}
    for (variant, model), group in predictions.groupby(['variant', 'model'], sort=True):
        y, predicted = group['actual'].to_numpy(), group['predicted'].to_numpy()
        for class_id, label in enumerate(('Home', 'Draw', 'Away')):
            support = int((y == class_id).sum())
            selected = int((predicted == class_id).sum())
            tp = int(((y == class_id) & (predicted == class_id)).sum())
            precision = tp / selected if selected else 0.
            recall = tp / support if support else 0.
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.
            result['per_class'].append(dict(variant=variant, model=model, class_id=class_id,
                                            class_label=label, support=support,
                                            precision=precision, recall=recall, f1=f1))
        p = group[PROBABILITY_COLUMNS].to_numpy(dtype=float)
        confidence = p.max(axis=1)
        correct = y == predicted
        # Search decimal edges directly: .7 belongs to [.7, .8), not [.6, .7).
        edges = np.arange(11) / 10
        assignments = np.minimum(np.searchsorted(edges, confidence, side='right') - 1, 9)
        ece = 0.
        for index in range(10):
            mask = assignments == index
            count = int(mask.sum())
            mean_confidence = float(confidence[mask].mean()) if count else None
            accuracy = float(correct[mask].mean()) if count else None
            gap = mean_confidence - accuracy if count else None
            if count:
                ece += count / len(group) * abs(gap)
            result['calibration'].append(dict(variant=variant, model=model,
                bin_lower=float(edges[index]), bin_upper=float(edges[index + 1]),
                count=count, mean_confidence=mean_confidence, accuracy=accuracy, gap=gap))
        for record, conf in zip(group.to_dict('records'), confidence):
            if record['actual'] != record['predicted']:
                record.update(confidence=float(conf),
                    actual_label=('Home', 'Draw', 'Away')[int(record['actual'])],
                    predicted_label=('Home', 'Draw', 'Away')[int(record['predicted'])])
                result['confident_errors'].append(record)
        losses = -np.log(np.clip(p[np.arange(len(group)), y.astype(int)], np.finfo(float).eps, 1.))
        for team in sorted(set(group['home_team']) | set(group['away_team'])):
            mask = ((group['home_team'] == team) | (group['away_team'] == team)).to_numpy()
            result['by_team'].append(dict(variant=variant, model=model, team=team,
                n_matches=int(mask.sum()), accuracy=float(correct[mask].mean()),
                mean_log_loss=float(losses[mask].mean())))
        for season in sorted(group['season'].unique()):
            mask = (group['season'] == season).to_numpy()
            result['by_season'].append(dict(variant=variant, model=model, season=int(season),
                n_matches=int(mask.sum()), accuracy=float(correct[mask].mean()),
                log_loss=float(losses[mask].mean())))
        draws = y == 1
        result['summary'].append(dict(variant=variant, model=model, n_matches=len(group),
            n_errors=int((~correct).sum()), draw_recall=float(correct[draws].mean()) if draws.any() else 0.,
            accuracy=float(correct.mean()), log_loss=float(losses.mean()), ece=float(ece)))
    result['confident_errors'].sort(key=lambda row: -row['confidence'])
    return _json_safe(result)


def _json_safe(value):
    """Retain missing optional source metadata as JSON null, never invented labels."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def export_diagnostics(predictions: pd.DataFrame, output_dir) -> dict:
    """Write six header-preserving CSV tables and strict JSON; return keyed paths.

    Validate before creating output directories. Existing diagnostic files are
    replaced; unrelated files in output_dir are left untouched.
    """
    report = build_diagnostics(predictions)
    columns = {
        'per_class': ['variant', 'model', 'class_id', 'class_label', 'support', 'precision', 'recall', 'f1'],
        'calibration': ['variant', 'model', 'bin_lower', 'bin_upper', 'count', 'mean_confidence', 'accuracy', 'gap'],
        'confident_errors': list(dict.fromkeys([*predictions.columns, 'confidence', 'actual_label', 'predicted_label'])),
        'by_team': ['variant', 'model', 'team', 'n_matches', 'accuracy', 'mean_log_loss'],
        'by_season': ['variant', 'model', 'season', 'n_matches', 'accuracy', 'log_loss'],
        'summary': ['variant', 'model', 'n_matches', 'n_errors', 'draw_recall', 'accuracy', 'log_loss', 'ece'],
    }
    # Encode first: serialization errors must not leave partially written exports.
    encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths = {}
    for key, names in columns.items():
        filename = {'by_team': 'errors_by_team', 'by_season': 'errors_by_season'}.get(key, key)
        path = destination / f'{filename}.csv'
        pd.DataFrame(report[key], columns=names).to_csv(path, index=False)
        paths[key] = str(path)
    path = destination / 'diagnostics.json'
    path.write_text(encoded, encoding='utf-8')
    paths['diagnostics'] = str(path)
    return paths
