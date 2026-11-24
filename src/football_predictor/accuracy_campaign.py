"""Pre-registered, development-only accuracy experiments; production stays separate."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def freeze_campaign_protocol(directory, protocol):
    """Persist exact search/data criteria once; refuse silent post-result edits."""
    encoded = json.dumps(protocol, sort_keys=True, allow_nan=False, ensure_ascii=False)
    digest = hashlib.sha256(encoded.encode()).hexdigest()
    directory = Path(directory)
    path = directory / 'protocol.json'
    if path.exists():
        existing = json.loads(path.read_text())
        if existing.get('sha256') != digest or existing.get('protocol') != protocol:
            raise ValueError('The campaign protocol is frozen; use a new campaign directory.')
        return digest
    directory.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'sha256': digest, 'protocol': protocol}, indent=2,
                               ensure_ascii=False, allow_nan=False))
    return digest


def development_folds(frame, n_splits=3, initial_fraction=0.5):
    """Expanding whole-date folds within development data, never a test argument."""
    if frame.empty or not frame.index.is_unique or frame.date.isna().any():
        raise ValueError('Unique indexed, dated, nonempty development rows required.')
    if not isinstance(n_splits, int) or isinstance(n_splits, bool) or n_splits < 1:
        raise ValueError('n_splits must be a positive integer.')
    if not 0 < initial_fraction < 1:
        raise ValueError('initial_fraction must be between zero and one.')
    dates = np.sort(pd.to_datetime(frame.date).unique())
    start = max(1, int(len(dates) * initial_fraction))
    if len(dates) - start < n_splits:
        raise ValueError('Insufficient distinct development dates for expanding folds.')
    folds = []
    for block in np.array_split(dates[start:], n_splits):
        train = frame.index[frame.date < block[0]].tolist()
        validation = frame.index[frame.date.isin(block)].tolist()
        folds.append((train, validation))
    return folds


def adjust_probabilities(probabilities, multipliers):
    """Validation-fitted class bias, not a guarantee of probability calibration."""
    p = np.asarray(probabilities, dtype=float)
    w = np.asarray(multipliers, dtype=float)
    if (p.ndim != 2 or p.shape[1] != 3 or w.shape != (3,)
            or not np.isfinite(p).all() or not np.isfinite(w).all()
            or (p < 0).any() or (w <= 0).any()
            or not np.allclose(p.sum(axis=1), 1)):
        raise ValueError('Normalized three-class probabilities and positive multipliers required.')
    adjusted = p * w
    return adjusted / adjusted.sum(axis=1, keepdims=True)


def select_probability_adjustment(actual, probabilities) -> dict[str, Any]:
    """Small frozen validation-only bias search, enabled only for accuracy gain."""
    from itertools import product
    from .evaluation import evaluate_probabilities
    raw, _ = evaluate_probabilities('validation', actual, probabilities)
    identity: dict[str, Any] = dict(multipliers=[1.0, 1.0, 1.0], metrics=raw)
    trials: list[dict[str, Any]] = []
    for draw, away in product((.8, 1.0, 1.2), repeat=2):
        multipliers = [1.0, draw, away]
        adjusted = adjust_probabilities(probabilities, multipliers)
        metric, _ = evaluate_probabilities('validation', actual, adjusted)
        trials.append(dict(multipliers=multipliers, metrics=metric))
    eligible = [trial for trial in trials if trial['metrics']['accuracy'] > float(raw['accuracy'])
                and trial['metrics']['log_loss'] <= float(raw['log_loss']) + .05]
    selected = (min(eligible, key=lambda trial: (-trial['metrics']['accuracy'],
                -trial['metrics']['macro_f1'], trial['metrics']['log_loss'], trial['multipliers']))
                if eligible else identity)
    return dict(selected=selected, raw_metrics=raw, trials=trials,
                criterion='strict validation accuracy gain, log loss degradation <=0.05; otherwise identity')


def paired_accuracy_interval(actual, before, after, dates, *, n_bootstrap=2000,
                             block_dates=7, seed=42) -> dict[str, Any]:
    """Paired circular moving-date-block bootstrap; retrospective, approximate CI."""
    actual, before, after = [np.asarray(values) for values in (actual, before, after)]
    dates = pd.to_datetime(dates)
    if (not len(actual) or any(values.shape != actual.shape for values in (before, after))
            or actual.ndim != 1 or len(dates) != len(actual) or dates.isna().any()):
        raise ValueError('Nonempty aligned one-dimensional outcomes and dates required.')
    if any(not np.isin(values, [0, 1, 2]).all() for values in (actual, before, after)):
        raise ValueError('Outcome labels must be home=0, draw=1, away=2.')
    if n_bootstrap < 1 or block_dates < 1:
        raise ValueError('Positive bootstrap count and date block length required.')
    frame = pd.DataFrame({'date': dates, 'difference': (after == actual).astype(float)
                          - (before == actual).astype(float)})
    grouped = frame.groupby('date', sort=True).difference.agg(['sum', 'count'])
    n_dates = len(grouped)
    block_dates = min(block_dates, n_dates)
    rng = np.random.default_rng(seed)
    n_blocks = (n_dates + block_dates - 1) // block_dates
    starts = rng.integers(0, n_dates, size=(n_bootstrap, n_blocks))
    indices = ((starts[:, :, None] + np.arange(block_dates)) % n_dates).reshape(n_bootstrap, -1)[:, :n_dates]
    samples = grouped['sum'].to_numpy()[indices].sum(axis=1) / grouped['count'].to_numpy()[indices].sum(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return dict(delta=float(frame.difference.mean()), low=float(low), high=float(high),
                n_matches=len(actual), n_dates=n_dates, n_bootstrap=n_bootstrap,
                block_dates=block_dates, seed=seed,
                method='paired circular moving-date-block percentile bootstrap; approximate 95% interval')


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    temporary.replace(path)


def run_accuracy_campaign(csv_path, output_dir, *, candidate_sets=None, workers=3, progress=None) -> dict[str, Any]:
    """Freeze grids, select on development, then evaluate frozen choices once.

    Cached selections are reusable only under an identical protocol fingerprint.
    This is retrospective evaluation of previously inspected historical labels.
    No application default, source CSV or container image is modified.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from datetime import datetime
    from importlib.metadata import version
    from sklearn.metrics import classification_report
    from .advanced_features import build_advanced_features, FEATURE_SETS
    from .advanced_selection import (classifier_candidates, select_advanced_classifier,
        fit_classifier_candidate, classifier_candidate_probabilities)
    from .advanced_poisson_selection import (poisson_candidates, select_advanced_poisson,
        fit_poisson_candidate, poisson_candidate_probabilities)
    from .data import load_matches
    from .evaluation import chronological_split, evaluate_probabilities
    from .service import ProjectService

    if workers not in (1, 2, 3):
        raise ValueError('workers must be 1, 2 or 3.')
    source, directory = Path(csv_path).resolve(), Path(output_dir).resolve()
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    frame = build_advanced_features(load_matches(source))
    train, validation, test = chronological_split(frame)
    development = pd.concat([train, validation]).sort_values(['date', 'home_team', 'away_team'])
    folds = development_folds(development)
    models = ('KNN', 'Random Forest', 'Poisson')
    grids = candidate_sets if candidate_sets is not None else {
        'KNN': classifier_candidates('KNN'), 'Random Forest': classifier_candidates('Random Forest'),
        'Poisson': poisson_candidates()}
    if set(grids) != set(models) or any(not grids[name] for name in models):
        raise ValueError('Nonempty candidate grids required for all three models.')
    source_files = ('accuracy_campaign.py', 'advanced_features.py', 'advanced_selection.py',
                    'advanced_poisson_selection.py', 'models/dixon_coles.py', 'models/poisson.py',
                    'models/__init__.py', 'features.py', 'data.py', 'service.py', 'evaluation.py')
    package = Path(__file__).parent
    code_hashes = {name: hashlib.sha256((package / name).read_bytes()).hexdigest() for name in source_files}
    protocol = dict(source_sha256=source_hash, n_matches=len(frame),
        development_indices=list(map(int, development.index)), test_indices=list(map(int, test.index)),
        folds=[dict(train=list(map(int, a)), validation=list(map(int, b))) for a, b in folds],
        feature_sets=FEATURE_SETS, candidates=grids, code_sha256=code_hashes,
        versions={name: version(name) for name in ('numpy', 'pandas', 'scipy', 'scikit-learn')},
        primary_metric='pooled development OOF accuracy',
        tie_breakers=['macro_f1 descending', 'log_loss ascending', 'candidate ID ascending'],
        bias_grid=[[1., d, a] for d in (.8, 1., 1.2) for a in (.8, 1., 1.2)],
        bias_policy='strict validation accuracy gain; log-loss degradation <=0.05; otherwise identity',
        bootstrap=dict(n_bootstrap=2000, block_dates=7, seed=42),
        epistemic_status='retrospective previously inspected labels; not prospective evidence',
        test_protocol='fixed fitted models; pre-match features/rates walk forward through earlier observed results',
        legacy_protocol='unchanged actual ProjectService with original validation-log-loss selection')
    digest = freeze_campaign_protocol(directory, protocol)
    emit = progress if progress is not None else lambda message: None
    history = []

    def phase(name):
        history.append(dict(phase=name, at=datetime.now().astimezone().isoformat()))
        _write_json(directory / 'status.json', dict(phase=name, protocol_sha256=digest, history=history))
        emit(name)

    def select_one(name):
        slug = name.lower().replace(' ', '_')
        path = directory / f'selection_{slug}.json'
        if path.exists():
            cached = json.loads(path.read_text())
            if cached.get('protocol_sha256') != digest:
                raise ValueError('Selection cache belongs to a different frozen protocol.')
            return cached['selection']
        notify = lambda message: emit(f'{name}: {message}')
        result = (select_advanced_poisson(development, folds, candidates=grids[name], progress=notify)
                  if name == 'Poisson' else select_advanced_classifier(
                      name, development, FEATURE_SETS, folds, candidates=grids[name], progress=notify))
        _write_json(path, dict(protocol_sha256=digest, selection=result))
        return result

    phase('protocol_frozen')
    phase('development_selection')
    selections = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(select_one, name): name for name in models}
        for future in as_completed(futures):
            name = futures[future]
            selection = future.result()
            if selection['selected'] is None:
                phase(f'failed_all_candidates_{name}')
                raise RuntimeError(f'All {name} candidates failed; inspect retained selection diagnostics.')
            records = pd.DataFrame(selection['oof_records'])
            probabilities = records[['p_home', 'p_draw', 'p_away']].to_numpy()
            adjustment = select_probability_adjustment(records.actual.to_numpy(), probabilities)
            selections[name] = dict(candidate=selection['selected'], adjustment=adjustment,
                trials=selection['trials'], n_validation=len(records))
            emit(f'{name}: selection complete, candidate={selection["selected"]["id"]}')
    _write_json(directory / 'selected_configs.json', dict(protocol_sha256=digest,
        source_sha256=source_hash, selection_cutoff=str(development.date.max().date()),
        selections=selections))
    phase('selection_frozen')
    # From this point the target labels are scored, never used to change choices.
    phase('legacy_evaluation')
    baseline = ProjectService(source)
    baseline_report = baseline.train(progress=emit)
    legacy_rows = baseline.predictions_test
    keys = ['date', 'home_team', 'away_team']
    test_keys = test[keys].copy()
    test_keys['date'] = test_keys.date.dt.strftime('%Y-%m-%d')
    probabilities = {}
    for name, part in legacy_rows.groupby('model', sort=True):
        paired = test_keys.merge(part, on=keys, how='left', validate='one_to_one')
        if paired[['p_home', 'p_draw', 'p_away']].isna().any().any():
            raise RuntimeError('Legacy/test fixture alignment failed.')
        probabilities[('legacy', name)] = paired[['p_home','p_draw','p_away']].to_numpy()
    phase('advanced_evaluation')
    fit_diagnostics = {}
    for name in models:
        candidate = selections[name]['candidate']
        if name == 'Poisson':
            model = fit_poisson_candidate(candidate, development)
            p = poisson_candidate_probabilities(model, candidate, frame, test.index)
            fit_diagnostics[name] = model.optimization_
        else:
            model = fit_classifier_candidate(name, candidate, development, development.index, FEATURE_SETS)
            p = classifier_candidate_probabilities(model, candidate, test, FEATURE_SETS)
            fit_diagnostics[name] = model.training_metadata_
        probabilities[('advanced_raw', name)] = p
        probabilities[('advanced_selected', name)] = adjust_probabilities(
            p, selections[name]['adjustment']['selected']['multipliers'])
    metrics, predictions, per_class, by_season = [], [], [], []
    confusion, intervals = {}, {}
    for (variant, name), p in probabilities.items():
        metric, matrix = evaluate_probabilities(name, test.result, p)
        metrics.append(dict(variant=variant, **metric))
        confusion[f'{variant}/{name}'] = matrix
        classes = classification_report(test.result, p.argmax(axis=1), labels=[0,1,2],
                                         output_dict=True, zero_division=0)
        per_class.extend(dict(variant=variant, model=name, label=label, **classes[str(label)])
                         for label in (0,1,2))
        for row, probability in zip(test.itertuples(), p):
            predictions.append(dict(variant=variant, model=name, date=str(row.date.date()),
                season=int(row.season), home_team=row.home_team, away_team=row.away_team,
                actual=int(row.result), predicted=int(probability.argmax()),
                p_home=float(probability[0]), p_draw=float(probability[1]), p_away=float(probability[2])))
        for season in sorted(test.season.unique()):
            mask = test.season.eq(season).to_numpy()
            metric, _ = evaluate_probabilities(name, test.loc[mask].result, p[mask])
            by_season.append(dict(variant=variant, season=int(season), **metric))
    for name in models:
        intervals[name] = paired_accuracy_interval(test.result,
            probabilities[('legacy', name)].argmax(axis=1),
            probabilities[('advanced_selected', name)].argmax(axis=1), test.date)
    if hashlib.sha256(source.read_bytes()).hexdigest() != source_hash:
        raise RuntimeError('Source CSV changed during the campaign; results cannot be finalized.')
    report = dict(status='complete', source=str(source), source_sha256=source_hash,
        protocol_sha256=digest, protocol=protocol,
        development=dict(n=len(development), start=str(development.date.min().date()), end=str(development.date.max().date())),
        test=dict(n=len(test), start=str(test.date.min().date()), end=str(test.date.max().date())),
        selections=selections, metrics=metrics, confidence_intervals=intervals,
        confusion=confusion, final_fit_diagnostics=fit_diagnostics, legacy_report=baseline_report,
        caveat='Retrospective experiment, previously inspected labels; no guaranteed future accuracy or profit.')
    pd.DataFrame(metrics).to_csv(directory / 'metrics.csv', index=False)
    pd.DataFrame(predictions).to_csv(directory / 'predictions.csv', index=False)
    pd.DataFrame(per_class).to_csv(directory / 'per_class.csv', index=False)
    pd.DataFrame(by_season).to_csv(directory / 'metrics_by_season.csv', index=False)
    tuning = []
    for name, selection in selections.items():
        for trial in selection['trials']:
            row = {key: value for key, value in trial.items() if key not in ('candidate', 'fold_metrics')}
            row.update(model=name, candidate=json.dumps(trial['candidate'], sort_keys=True),
                       fold_metrics=json.dumps(trial.get('fold_metrics', []), sort_keys=True))
            tuning.append(row)
    pd.DataFrame(tuning).to_csv(directory / 'tuning.csv', index=False)
    _write_json(directory / 'confidence_intervals.json', intervals)
    _write_json(directory / 'report.json', report)
    phase('complete')
    return report


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description='Retrospective validation-only accuracy campaign')
    parser.add_argument('--data', default='matches.csv', help='Team-perspective input CSV')
    parser.add_argument('--output', default='reports/accuracy_campaign', help='Frozen campaign directory')
    parser.add_argument('--workers', type=int, choices=(1, 2, 3), default=3)
    args = parser.parse_args(argv)
    report = run_accuracy_campaign(args.data, args.output, workers=args.workers,
                                    progress=lambda message: print(message, flush=True))
    print(pd.DataFrame(report['metrics']).to_string(index=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


