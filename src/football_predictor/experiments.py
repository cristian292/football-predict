"""Retrospective, validation-selected, expanding-season model comparisons."""
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import load_matches
from .features import BASIC_FEATURES, COMPLEX_FEATURES, build_features
from .evaluation import evaluate_probabilities
from .models import aligned_probabilities, make_classifier
from .models.poisson import PoissonModel, historical_rates
from .selection import select_classifier, select_poisson


_MODELS = ('KNN', 'Random Forest', 'Poisson')
_VARIANTS = ('legacy', 'optimized')
_GOALS = ['home_goals', 'away_goals']
_PROTOCOL = (
    'Retrospective expanding-season benchmark, not a pristine new holdout: prior old test labels '
    'have already been exposed during project development. Test seasons default to ending years '
    '2023..2027. For each test season, the immediately previous ending year is validation; '
    'all earlier seasons train. At least two prior seasons and that validation season are required. '
    'Both variants use the same folds and feature columns. Fixed deterministic candidate grids are selected only by '
    'validation three-class log loss; selected models refit on train+validation before scoring '
    'the season once. Features and Poisson rates walk forward using strictly earlier dates, '
    'including earlier test results; this is not season-ahead forecasting. No candidates adapt '
    'to test scores. Legacy is the original 8-KNN/4-forest grid (deduplicated for small training '
    'sets) and Poisson window=10/smoothing=5; optimized includes every legacy candidate. '
    'Baseline is Laplace-smoothed development class frequencies, paired across variants. '
    'No test performance improvement is guaranteed.'
)


def _legacy_selection(name, optimized, n_train) -> dict[str, Any]:
    """Reuse identical fitted validation trials instead of fitting legacy twice."""
    if name == 'KNN':
        neighbors = {min(k, n_train) for k in (3, 7, 15, 31)}
        trials = [t for t in optimized['trials']
                  if t['params']['n_neighbors'] in neighbors and t['params']['p'] == 2]
    elif name == 'Random Forest':
        trials = [t for t in optimized['trials']
                  if t['params']['max_depth'] in (5, None)
                  and t['params']['min_samples_leaf'] in (2, 5)
                  and t['params']['max_features'] == 'sqrt']
    else:
        trials = [t for t in optimized['trials'] if t['params'] == {'window': 10, 'smoothing': 5.0}]
    best = min(trials, key=lambda t: t['validation_log_loss'])
    result = dict(params=dict(best['params']), trials=trials)
    if 'columns' in optimized:
        result['columns'] = list(optimized['columns'])
    return result


def run_benchmark(csv_path, output_dir, progress=None, *, test_seasons=None) -> dict[str, Any]:
    """Export serializable metrics, predictions and validation traces.

    ``test_seasons`` optionally restricts the default ending years 2023..2027,
    primarily for synthetic tests. Source CSV is never modified.
    """
    source = Path(csv_path).resolve()
    directory = Path(output_dir).resolve()
    paths = {name: str(directory / name) for name in ('metrics_by_season.csv', 'predictions_by_season.csv',
                                                    'tuning_by_season.csv', 'benchmark.json')}
    if any(Path(path).resolve() == source for path in paths.values()):
        raise ValueError('Benchmark outputs must not overwrite the input source CSV.')
    source_hash = sha256(source.read_bytes()).hexdigest()
    frame = build_features(load_matches(source))
    seasons = sorted(int(s) for s in frame.season.unique())
    requested = list(range(2023, 2028)) if test_seasons is None else sorted(set(test_seasons))
    emit = progress if progress is not None else lambda message: None
    metrics, predictions, trials, folds = [], [], [], []
    for season in requested:
        previous = [s for s in seasons if s < season]
        if season not in seasons or len(previous) < 2 or season - 1 not in previous:
            continue
        validation_season = season - 1
        train = frame.loc[frame.season < validation_season]
        validation = frame.loc[frame.season == validation_season]
        test = frame.loc[frame.season == season]
        if train.date.max() >= validation.date.min() or validation.date.max() >= test.date.min():
            raise ValueError('Season partitions must follow strict whole-date chronology: train < validation < test.')
        development = pd.concat([train, validation]).sort_values('date')
        emit(f'Season {season}: validation-only selection')
        metadata = dict(season=int(season), n_train=len(train), n_validation=len(validation), n_test=len(test),
                        training_end=train.date.max().strftime('%Y-%m-%d'),
                        validation_start=validation.date.min().strftime('%Y-%m-%d'),
                        validation_end=validation.date.max().strftime('%Y-%m-%d'),
                        test_complete=bool(season != seasons[-1]),
                        test_status='partial' if season == seasons[-1] else 'complete')
        fold: dict[str, Any] = dict(metadata, train_seasons=previous[:-1], validation_season=validation_season,
                    test_start=test.date.min().strftime('%Y-%m-%d'),
                    test_end=test.date.max().strftime('%Y-%m-%d'),
                    development_end=development.date.max().strftime('%Y-%m-%d'),
                    selections={}, poisson_optimization={}, confusion={})
        optimized = {name: select_classifier(name, train, validation, progress=progress)
                     for name in _MODELS[:2]}
        optimized['Poisson'] = select_poisson(development, train.index, validation.index, progress=progress)
        selections = dict(optimized=optimized,
                          legacy={name: _legacy_selection(name, optimized[name], len(train)) for name in _MODELS})
        frequencies = (np.bincount(development.result, minlength=3) + 1) / (len(development) + 3)
        rate_cache = {}
        for variant in _VARIANTS:
            fold['selections'][variant] = selections[variant]
            fold['confusion'][variant] = {}
            probabilities = {}
            for name in _MODELS:
                selected = selections[variant][name]
                trials.extend(dict(t, season=int(season), variant=variant) for t in selected['trials'])
                emit(f'Season {season}: refit {variant} {name} on train + validation')
                if name == 'Poisson':
                    params = selected['params']
                    key = (params['window'], params['smoothing'])
                    if key not in rate_cache:
                        rate_cache[key] = historical_rates(frame, **params)
                    rates = rate_cache[key]
                    selection_optimization = selected.get('optimization')
                    if selection_optimization is None:
                        selection_optimization = PoissonModel().fit(rates.loc[train.index], train[_GOALS]).optimization_
                    model = PoissonModel().fit(rates.loc[development.index], development[_GOALS])
                    probabilities[name] = model.predict_proba(rates.loc[test.index])
                    fold['poisson_optimization'][variant] = dict(selection=selection_optimization,
                                                                evaluation=model.optimization_)
                else:
                    columns = selected['columns']
                    model = make_classifier(name, selected['params']).fit(development[columns], development.result)
                    probabilities[name] = aligned_probabilities(model, test[columns])
            probabilities['Baseline (majority)'] = np.tile(frequencies, (len(test), 1))
            for name, p in probabilities.items():
                metric, confusion = evaluate_probabilities(name, test.result, p)
                metrics.append(dict(metric, **metadata, variant=variant))
                fold['confusion'][variant][name] = confusion
                for row, probability in zip(test.itertuples(), p):
                    predictions.append(dict(date=row.date.strftime('%Y-%m-%d'), season=int(row.season),
                                             home_team=row.home_team, away_team=row.away_team,
                                             actual=int(row.result), model=name, variant=variant,
                                             p_home=float(probability[0]), p_draw=float(probability[1]),
                                             p_away=float(probability[2]), predicted=int(probability.argmax())))
        folds.append(fold)
    if not folds:
        raise ValueError('No eligible test seasons: at least two prior seasons are required.')
    directory.mkdir(parents=True, exist_ok=True)
    report = dict(metrics=metrics, folds=folds, trials=trials, paths=paths, protocol=_PROTOCOL,
                  source=str(source), source_sha256=source_hash, n_matches=len(frame),
                  class_order=['home', 'draw', 'away'], random_state=42,
                  completeness_assumption=('The latest observed season is conservatively marked partial; earlier '
                                           'observed seasons are assumed complete. Fixture-count completeness is '
                                           'not certified by this benchmark.'),
                  feature_columns={'KNN': list(BASIC_FEATURES), 'Random Forest': list(COMPLEX_FEATURES)})
    serialized = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
    pd.DataFrame(metrics).to_csv(paths['metrics_by_season.csv'], index=False)
    pd.DataFrame(predictions).to_csv(paths['predictions_by_season.csv'], index=False)
    tuning = pd.DataFrame(trials)
    tuning['params'] = tuning['params'].map(lambda params: json.dumps(params, sort_keys=True, allow_nan=False))
    tuning.to_csv(paths['tuning_by_season.csv'], index=False)
    Path(paths['benchmark.json']).write_text(serialized, encoding='utf-8')
    emit('Expanding-season benchmark complete')
    return report
