"""Explicitly configured adapter; no selection, global config, or import-time I/O.

The caller supplies development-selected candidates and their provenance. A
protocol digest is retained, not authenticated: its protocol is not an input.
Nothing in this adapter establishes an accuracy improvement.
"""
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

from .accuracy_campaign import adjust_probabilities
from .advanced_features import FEATURE_SETS, build_advanced_features, fixture_advanced_features
from .advanced_selection import make_advanced_classifier
from .data import load_matches
from .evaluation import chronological_split
from .service import ProjectService


_NAMES = ('KNN', 'Random Forest', 'Poisson')


def _validate_config(config):
    """Detach finite JSON input and reject incomplete or unsupported settings."""
    try:
        if not isinstance(config, dict):
            raise ValueError('config must be a dictionary.')
        config = json.loads(json.dumps(config, allow_nan=False))
        required = {'schema_version', 'source_sha256', 'protocol_sha256', 'selection_cutoff', 'models'}
        if set(config) != required or type(config['schema_version']) is not int or config['schema_version'] != 1:
            raise ValueError('Expected complete schema_version 1 config.')
        for field in ('source_sha256', 'protocol_sha256'):
            if not isinstance(config[field], str) or not re.fullmatch(r'[0-9a-f]{64}', config[field]):
                raise ValueError(f'{field} must be an exact lowercase SHA-256 digest.')
        cutoff = config['selection_cutoff']
        if not isinstance(cutoff, str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2}', cutoff):
            raise ValueError('selection_cutoff must be YYYY-MM-DD.')
        pd.Timestamp(cutoff)
        if not isinstance(config['models'], dict) or set(config['models']) != set(_NAMES):
            raise ValueError('Config requires exactly KNN, Random Forest and Poisson.')
        for name, entry in config['models'].items():
            if set(entry) != {'candidate', 'multipliers', 'validation_metrics'}:
                raise ValueError('Each model requires candidate, multipliers and validation_metrics.')
            weights = entry['multipliers']
            if not isinstance(weights, list) or len(weights) != 3 or any(type(w) not in (int, float) for w in weights):
                raise ValueError('multipliers require three positive JSON numbers.')
            adjust_probabilities([[1/3, 1/3, 1/3]], weights)
            metrics = entry['validation_metrics']
            if set(metrics) != {'accuracy', 'macro_f1', 'log_loss', 'brier', 'n_test'}:
                raise ValueError('Complete validation_metrics required.')
            for key, maximum in [('accuracy', 1), ('macro_f1', 1), ('log_loss', np.inf), ('brier', 2)]:
                value = metrics[key]
                if type(value) not in (int, float) or not 0 <= value <= maximum:
                    raise ValueError(f'Invalid validation metric {key}.')
            if type(metrics['n_test']) is not int or metrics['n_test'] <= 0:
                raise ValueError('Positive validation n_test required.')
            candidate = entry['candidate']
            if not isinstance(candidate['id'], str) or not candidate['id'] or not isinstance(candidate['params'], dict):
                raise ValueError('Candidate requires string ID and params dictionary.')
            if name == 'Poisson':
                kind = candidate['kind']
                if set(candidate) != {'id', 'kind', 'params'}:
                    raise ValueError('Unsupported Poisson candidate fields.')
                params = candidate['params']
                if kind == 'rolling_poisson':
                    if set(params) != {'window', 'smoothing'} or type(params['window']) is not int or params['window'] < 1:
                        raise ValueError('Positive integer rolling window required.')
                    if type(params['smoothing']) not in (int, float) or params['smoothing'] <= 0:
                        raise ValueError('Positive rolling smoothing required.')
                elif kind == 'dixon_coles':
                    from .models.dixon_coles import DixonColesModel
                    DixonColesModel(**params)
                else:
                    raise ValueError('Unsupported Poisson kind.')
            else:
                allowed = {'id', 'feature_set', 'params', 'train_window_days', 'sample_half_life_days', 'probability_smoothing'}
                if set(candidate) - allowed or candidate['feature_set'] not in FEATURE_SETS:
                    raise ValueError('Unsupported classifier feature_set or candidate fields.')
                model = make_advanced_classifier(name, candidate['params'])
                model.named_steps['classifier']._validate_params()
                for field in ('train_window_days', 'sample_half_life_days'):
                    value = candidate.get(field)
                    if value is not None and (type(value) not in (int, float) or value <= 0):
                        raise ValueError(f'Positive {field} required.')
                if name != 'Random Forest' and candidate.get('sample_half_life_days') is not None:
                    raise ValueError('Sample decay requires Random Forest.')
                smoothing = candidate.get('probability_smoothing', 0.)
                if type(smoothing) not in (int, float) or not 0 <= smoothing < 1:
                    raise ValueError('probability_smoothing must be in [0, 1).')
        return config
    except (KeyError, TypeError, OverflowError) as error:
        raise ValueError('Invalid optimized service configuration.') from error


class OptimizedProjectService(ProjectService):
    """ProjectService-compatible, opt-in use of previously selected settings."""

    def __init__(self, csv_path, config):
        self._config = _validate_config(config)
        self.csv_path = Path(csv_path)
        data = self.csv_path.read_bytes()
        self.source_sha256 = hashlib.sha256(data).hexdigest()
        if self.source_sha256 != self._config['source_sha256']:
            raise ValueError('CSV source hash does not match optimized config.')
        # pandas accepts file-like objects; use precisely the bytes just hashed.
        self.matches = load_matches(BytesIO(data))  # type: ignore[arg-type]
        if self.matches.empty:
            raise ValueError('No completed fixtures in CSV.')
        self.features = build_advanced_features(self.matches)
        _, _, test = chronological_split(self.features)
        if pd.Timestamp(self._config['selection_cutoff']) >= test.date.min().normalize():
            raise ValueError('Selection cutoff must strictly precede held-out test start; refusing leakage.')
        self.report = None

    def _check_source(self):
        """Refuse stale provenance if the caller changes/removes the CSV."""
        try:
            digest = hashlib.sha256(self.csv_path.read_bytes()).hexdigest()
        except OSError as error:
            raise ValueError('CSV source changed or is no longer readable.') from error
        if digest != self.source_sha256:
            raise ValueError('CSV source changed; hash no longer matches optimized config.')

    def train(self, progress=None) -> dict[str, Any]:
        """Fit frozen choices on development, score test once, refit for future."""
        from . import advanced_selection as classifiers
        from . import advanced_poisson_selection as poisson
        from .evaluation import evaluate_probabilities
        self._check_source()
        if self.report is not None:
            return self.report
        emit = progress if progress is not None else lambda message: None
        emit('Preparing chronological train / validation / test partitions')
        train, validation, test = chronological_split(self.features)
        development = pd.concat([train, validation]).sort_values(['date', 'home_team', 'away_team'])
        settings = self._config['models']
        columns = {name: list(FEATURE_SETS[settings[name]['candidate']['feature_set']]) for name in _NAMES[:2]}
        emit('Refitting selected models on train + validation; scoring held-out test once')
        probabilities = {}
        for name in _NAMES[:2]:
            candidate = settings[name]['candidate']
            model = classifiers.fit_classifier_candidate(name, candidate, self.features, development.index, FEATURE_SETS)
            raw = classifiers.classifier_candidate_probabilities(model, candidate, test, FEATURE_SETS)
            probabilities[name] = adjust_probabilities(raw, settings[name]['multipliers'])
        emit('Optimizing Poisson training goal likelihood with L-BFGS-B')
        candidate = settings['Poisson']['candidate']
        evaluation_poisson = poisson.fit_poisson_candidate(candidate, development)
        raw = poisson.poisson_candidate_probabilities(evaluation_poisson, candidate, self.features, test.index)
        probabilities['Poisson'] = adjust_probabilities(raw, settings['Poisson']['multipliers'])
        frequencies = (np.bincount(development.result, minlength=3) + 1) / (len(development) + 3)
        probabilities['Baseline (majority)'] = np.tile(frequencies, (len(test), 1))
        metrics, confusion, records = [], {}, []
        for name, p in probabilities.items():
            metric, matrix = evaluate_probabilities(name, test.result, p)
            metrics.append(metric)
            confusion[name] = matrix
            for row, probability in zip(test.itertuples(), p):
                records.append(dict(date=row.date.strftime('%Y-%m-%d'), home_team=row.home_team,
                                    away_team=row.away_team, actual=int(row.result), model=name,
                                    p_home=float(probability[0]), p_draw=float(probability[1]),
                                    p_away=float(probability[2]), predicted=int(probability.argmax())))
        self.predictions_test = pd.DataFrame(records)
        emit('Refitting all-history models for strictly future predictions')
        self.models_ = {name: classifiers.fit_classifier_candidate(
            name, settings[name]['candidate'], self.features, self.features.index, FEATURE_SETS) for name in _NAMES[:2]}
        self.poisson_ = poisson.fit_poisson_candidate(candidate, self.features)
        self.model_columns_ = columns
        self.last_training_date_ = pd.Timestamp(self.features.date.max()).normalize()
        split: dict[str, Any] = {name: dict(n=len(part), n_dates=int(part.date.nunique()),
                           start=part.date.min().strftime('%Y-%m-%d'), end=part.date.max().strftime('%Y-%m-%d'))
                 for name, part in [('train', train), ('validation', validation), ('test', test)]}
        split['protocol'] = ('Whole-date chronological 60/20/20 split. Fixed candidates and class multipliers '
                             'previously selected on earlier development OOF, never searched on this test. '
                             'Train+validation refit; one held-out test evaluation with pre-date walk-forward '
                             'features including earlier observed test results, not season-ahead. '
                             'All-history refit for strictly future predictions only.')
        trials = [dict(model=name, params=dict(candidate=settings[name]['candidate'],
                                               multipliers=settings[name]['multipliers']),
                       validation_log_loss=settings[name]['validation_metrics']['log_loss'],
                       validation_accuracy=settings[name]['validation_metrics']['accuracy'],
                       validation_macro_f1=settings[name]['validation_metrics']['macro_f1'],
                       validation_brier=settings[name]['validation_metrics']['brier'],
                       n_validation=settings[name]['validation_metrics']['n_test'],
                       provenance='Caller-supplied development OOF metrics, not recomputed on test') for name in _NAMES]
        self._check_source()
        report: dict[str, Any] = poisson._json_safe(dict(
            metrics=metrics, tuning=trials, split=split, confusion=confusion,
            selected_params={name: row['params'] for name, row in zip(_NAMES, trials)},
            feature_columns=columns, class_order=['home', 'draw', 'away'],
            poisson_optimization=dict(selection=dict(status='previously_selected',
                candidate=candidate, validation_metrics=settings['Poisson']['validation_metrics'],
                trace=[], method='Previously selected on development OOF; not refitted for selection'),
                evaluation=evaluation_poisson.optimization_, future=self.poisson_.optimization_),
            baseline='Majority argmax of Laplace-smoothed train+validation class frequencies',
            brier_definition='Mean sum of squared errors over all three classes (range 0 to 2)',
            random_state=42, n_matches=len(self.matches), source=str(self.csv_path.resolve()),
            source_sha256=self.source_sha256, protocol_sha256=self._config['protocol_sha256'],
            selection_cutoff=self._config['selection_cutoff'], schema_version=1,
            future_training_end=self.last_training_date_.strftime('%Y-%m-%d')))
        emit('Training and held-out evaluation complete')
        self._check_source()
        self.report = report
        return report

    def predict(self, home, away, date) -> dict[str, Any]:
        """Predict from all-history fits; reweight the entire goal distribution."""
        from .advanced_selection import classifier_candidate_probabilities
        from .models.poisson import historical_rates, score_distribution
        if self.report is None:
            raise RuntimeError('Train models before making predictions.')
        self._check_source()
        if home == away:
            raise ValueError('Home and away teams must be different.')
        teams = set(self.matches.home_team) | set(self.matches.away_team)
        if home not in teams or away not in teams:
            raise ValueError('Unknown team; select a team present in the loaded data.')
        date = pd.Timestamp(date)
        if pd.isna(date) or date.tzinfo is not None:
            raise ValueError('Use a valid timezone-naive fixture date.')
        date = date.normalize()
        if date <= self.last_training_date_:
            raise ValueError(f'Prediction date must be after {self.last_training_date_:%Y-%m-%d}; retrospective predictions would leak training data.')
        fixture = fixture_advanced_features(self.matches, home, away, date)
        settings = self._config['models']
        probabilities = {name: adjust_probabilities(classifier_candidate_probabilities(
            model, settings[name]['candidate'], fixture, FEATURE_SETS), settings[name]['multipliers'])[0].tolist()
            for name, model in self.models_.items()}
        query = pd.DataFrame([dict(date=date, season=date.year + int(date.month >= 8),
                                  home_team=home, away_team=away, home_goals=np.nan, away_goals=np.nan)])
        candidate = settings['Poisson']['candidate']
        if candidate['kind'] == 'dixon_coles':
            rates = self.poisson_.expected_goals(query)[0]
            matrix, _, tail = self.poisson_.score_distribution(*rates)
        else:
            history = pd.concat([self.matches, query], ignore_index=True)
            rates = self.poisson_.expected_goals(historical_rates(history, **candidate['params']).iloc[[-1]])[0]
            matrix, _, tail = score_distribution(*rates)
        home_grid, away_grid = np.indices(matrix.shape)
        weights = np.asarray(settings['Poisson']['multipliers'], dtype=float)
        matrix *= weights[np.where(home_grid > away_grid, 0, np.where(home_grid == away_grid, 1, 2))]
        matrix /= matrix.sum()
        probabilities['Poisson'] = [float(np.tril(matrix, -1).sum()), float(np.trace(matrix)),
                                    float(np.triu(matrix, 1).sum())]
        expected = [float((home_grid * matrix).sum()), float((away_grid * matrix).sum())]
        order = np.argsort(-matrix.ravel(), kind='stable')[:10]
        scores = [dict(score=f'{index // matrix.shape[1]}–{index % matrix.shape[1]}',
                       probability=float(matrix.ravel()[index])) for index in order]
        names = list(dict.fromkeys(column for group in self.model_columns_.values() for column in group))
        values = {column: float(fixture.iloc[0][column]) if pd.notna(fixture.iloc[0][column]) else None
                  for column in names}
        return dict(home=home, away=away, date=date.strftime('%Y-%m-%d'), probabilities=probabilities,
                    scores=scores, score_matrix=matrix.tolist(), expected_goals=expected,
                    intensity_rates=rates.tolist(), features=values, score_tail=float(tail),
                    score_tail_definition='Unweighted omitted base-distribution mass before truncation and outcome-region reweighting')

    def export_report(self, directory):
        """Reuse the legacy four-file schema without overwriting the CSV."""
        self._check_source()
        source = self.csv_path.resolve()
        for name in ('metrics.csv', 'tuning.csv', 'predictions_test.csv', 'report.json'):
            target = Path(directory) / name
            if target.resolve() == source or (target.exists() and target.samefile(source)):
                raise ValueError('Report export cannot overwrite the CSV source.')
        return super().export_report(directory)
