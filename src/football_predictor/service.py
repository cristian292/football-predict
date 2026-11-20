"""UI-facing orchestration with disjoint temporal model selection and evaluation."""
from pathlib import Path
import json
from typing import Any
import numpy as np
import pandas as pd
from .data import load_matches
from . import features as feature_module
from .evaluation import chronological_split, evaluate_probabilities
from .models import make_classifier, aligned_probabilities
from .models.poisson import PoissonModel, historical_rates, score_distribution
from sklearn.metrics import log_loss


class ProjectService:
    def __init__(self, csv_path):
        self.csv_path = Path(csv_path)
        self.matches = load_matches(csv_path)
        if self.matches.empty:
            raise ValueError('No completed fixtures in CSV.')
        self.features = feature_module.build_features(self.matches)
        self.report: dict[str, Any] | None = None

    def train(self, progress=None) -> dict[str, Any]:
        """Tune on validation, score test once, then refit for future fixtures only."""
        if self.report is not None:
            return self.report
        emit = progress if progress is not None else lambda message: None
        emit('Preparing chronological train / validation / test partitions')
        train, validation, test = chronological_split(self.features)
        rates = historical_rates(self.features)
        goals = ['home_goals', 'away_goals']
        columns = {
            'KNN': list(feature_module.BASIC_FEATURES),
            'Random Forest': list(feature_module.COMPLEX_FEATURES),
        }
        grids = {
            'KNN': [dict(n_neighbors=k, weights=w)
                    for k in sorted({min(k, len(train)) for k in (3, 7, 15, 31)})
                    for w in ('uniform', 'distance')],
            'Random Forest': [dict(max_depth=depth, min_samples_leaf=leaf)
                              for depth in (5, None) for leaf in (2, 5)],
        }
        trials, selected = [], {}
        for name, grid in grids.items():
            emit(f'Tuning {name} on validation log loss')
            candidates = []
            for params in grid:
                model = make_classifier(name, params).fit(train[columns[name]], train.result)
                p = aligned_probabilities(model, validation[columns[name]])
                score = float(log_loss(validation.result, p, labels=[0, 1, 2]))
                record = dict(model=name, params=dict(params), validation_log_loss=score)
                trials.append(record)
                candidates.append(record)
            selected[name] = min(candidates, key=lambda row: row['validation_log_loss'])['params']
        emit('Optimizing Poisson training goal likelihood with L-BFGS-B')
        poisson_selection = PoissonModel().fit(rates.loc[train.index], train[goals])
        poisson_validation = poisson_selection.predict_proba(rates.loc[validation.index])
        trials.append(dict(model='Poisson', params={'window': 10, 'smoothing': 5.,
                           'log_scale': float(poisson_selection.params_[0]),
                           'home_intercept': float(poisson_selection.params_[1])},
                           validation_log_loss=float(log_loss(validation.result, poisson_validation, labels=[0, 1, 2]))))
        development = pd.concat([train, validation]).sort_values('date')
        emit('Refitting selected models on train + validation; scoring held-out test once')
        evaluation_models = {name: make_classifier(name, params).fit(development[columns[name]], development.result)
                             for name, params in selected.items()}
        poisson_evaluation = PoissonModel().fit(rates.loc[development.index], development[goals])
        probabilities = {name: aligned_probabilities(model, test[columns[name]])
                         for name, model in evaluation_models.items()}
        probabilities['Poisson'] = poisson_evaluation.predict_proba(rates.loc[test.index])
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
        self.models_ = {name: make_classifier(name, params).fit(self.features[columns[name]], self.features.result)
                        for name, params in selected.items()}
        self.poisson_ = PoissonModel().fit(rates, self.features[goals])
        self.model_columns_ = columns
        self.last_training_date_ = pd.Timestamp(self.features.date.max()).normalize()
        split: dict[str, Any] = {name: dict(n=len(part), n_dates=int(part.date.nunique()),
                            start=part.date.min().strftime('%Y-%m-%d'), end=part.date.max().strftime('%Y-%m-%d'))
                 for name, part in [('train', train), ('validation', validation), ('test', test)]}
        split['protocol'] = ('Whole-date chronological 60/20/20 split. Hyperparameters selected by validation log loss; '
                             'train+validation refit before one held-out test evaluation. Pre-match features update '
                             'sequentially using observed earlier results, including earlier test dates (walk-forward, '
                             'not season-ahead). Future prediction models refitted on all data; retrospective dates rejected.')
        self.report = dict(metrics=metrics, tuning=trials, split=split, confusion=confusion,
                           selected_params=selected, feature_columns=columns, class_order=['home', 'draw', 'away'],
                           poisson_optimization=dict(selection=poisson_selection.optimization_,
                                                     evaluation=poisson_evaluation.optimization_, future=self.poisson_.optimization_),
                           baseline='Majority argmax of Laplace-smoothed train+validation class frequencies',
                           brier_definition='Mean sum of squared errors over all three classes (range 0 to 2)',
                           random_state=42, n_matches=len(self.matches), source=str(self.csv_path.resolve()),
                           future_training_end=self.last_training_date_.strftime('%Y-%m-%d'))
        emit('Training and held-out evaluation complete')
        return self.report

    def predict(self, home, away, date) -> dict[str, Any]:
        if self.report is None:
            raise RuntimeError('Train models before making predictions.')
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
        row = dict(date=date, season=self.matches.sort_values('date').iloc[-1].season,
                   home_team=home, away_team=away, home_goals=np.nan, away_goals=np.nan)
        future = pd.DataFrame([row])
        history = pd.concat([self.matches, future], ignore_index=True)
        fixture = feature_module.fixture_features(self.matches, home, away, date)
        probabilities = {name: aligned_probabilities(model, fixture[self.model_columns_[name]])[0].tolist()
                         for name, model in self.models_.items()}
        rates = historical_rates(history).iloc[[-1]]
        expected = self.poisson_.expected_goals(rates)[0]
        matrix, outcomes, tail = score_distribution(*expected)
        probabilities['Poisson'] = outcomes.tolist()
        order = np.argsort(-matrix.ravel(), kind='stable')[:10]
        scores = [dict(score=f'{index // matrix.shape[1]}–{index % matrix.shape[1]}',
                       probability=float(matrix.ravel()[index])) for index in order]
        names = list(dict.fromkeys(column for group in self.model_columns_.values() for column in group))
        values = {column: (float(fixture.iloc[0][column]) if pd.notna(fixture.iloc[0][column]) else None)
                  for column in names}
        return dict(home=home, away=away, date=date.strftime('%Y-%m-%d'), probabilities=probabilities,
                    scores=scores, score_matrix=matrix.tolist(), expected_goals=expected.tolist(),
                    features=values, score_tail=tail)

    def export_report(self, directory):
        """Export inspectable metadata and predictions, never executable model pickles."""
        if self.report is None:
            raise RuntimeError('Train models before exporting a report.')
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        paths = {name: directory / name for name in ('metrics.csv', 'tuning.csv', 'predictions_test.csv', 'report.json')}
        pd.DataFrame(self.report['metrics']).to_csv(paths['metrics.csv'], index=False)
        tuning = pd.DataFrame(self.report['tuning'])
        tuning['params'] = tuning['params'].map(lambda params: json.dumps(params, sort_keys=True, allow_nan=False))
        tuning.to_csv(paths['tuning.csv'], index=False)
        self.predictions_test.to_csv(paths['predictions_test.csv'], index=False)
        paths['report.json'].write_text(json.dumps(self.report, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
        return {name: str(path) for name, path in paths.items()}
