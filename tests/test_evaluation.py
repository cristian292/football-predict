import numpy as np
import pandas as pd
import pytest

from football_predictor.evaluation import chronological_split


@pytest.fixture
def csv_path(tmp_path):
    rows = []
    rng = np.random.default_rng(42)
    for day, date in enumerate(pd.date_range('2020-01-01', periods=40)):
        teams = ['A', 'B', 'C', 'D']
        for offset in (0, 2):
            home, away = teams[(day + offset) % 4], teams[(day + offset + 1) % 4]
            hg, ag = rng.poisson([1.6, 1.1])
            for venue, team, opponent, gf, ga in [('Home', home, away, hg, ag), ('Away', away, home, ag, hg)]:
                rows.append(dict(date=date, season=2020, venue=venue, team=team, opponent=opponent,
                                 gf=gf, ga=ga, sh=10, sot=4, poss=50))
    path = tmp_path / 'matches.csv'
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_service_exposes_canonical_and_prematch_data(csv_path):
    from football_predictor.service import ProjectService
    from football_predictor.features import BASIC_FEATURES
    service = ProjectService(csv_path)
    assert len(service.matches) == len(service.features) == 80
    assert {'date', 'home_team', 'away_team', 'home_goals', 'away_goals'} <= set(service.matches)
    assert set(BASIC_FEATURES) <= set(service.features)
    assert service.report is None


def test_service_trains_tunes_and_evaluates_disjoint_dates(csv_path):
    from football_predictor.service import ProjectService
    service = ProjectService(csv_path)
    messages = []
    report = service.train(progress=messages.append)
    assert report is service.report
    assert len(report['metrics']) == 4
    assert {row['model'] for row in report['metrics']} == {'KNN', 'Random Forest', 'Poisson', 'Baseline (majority)'}
    assert all(row['n_test'] == 16 for row in report['metrics'])
    assert all(np.isfinite(row['log_loss']) for row in report['metrics'])
    assert report['split']['train']['n'] == 48
    assert report['split']['validation']['n'] == report['split']['test']['n'] == 16
    assert report['split']['train']['end'] < report['split']['validation']['start']
    assert report['split']['validation']['end'] < report['split']['test']['start']
    for name in ['KNN', 'Random Forest']:
        trials = [row for row in report['tuning'] if row['model'] == name]
        assert len(trials) >= 4
        best = min(trials, key=lambda row: row['validation_log_loss'])
        assert report['selected_params'][name] == best['params']
    assert len(service.predictions_test) == 64
    assert all(np.array(matrix).sum() == 16 for matrix in report['confusion'].values())
    assert report['poisson_optimization']['selection']['n_train'] == 48
    assert report['poisson_optimization']['evaluation']['n_train'] == 64
    assert report['poisson_optimization']['future']['n_train'] == 80
    assert messages
    assert service.train() is report  # Held-out test is not repeatedly rescored.


def test_future_prediction_returns_normalized_class_and_score_probabilities(csv_path):
    from football_predictor.service import ProjectService
    service = ProjectService(csv_path)
    service.train()
    prediction = service.predict('A', 'B', '2020-02-10')
    assert prediction['home'] == 'A' and prediction['away'] == 'B'
    assert prediction['date'] == '2020-02-10'
    assert set(prediction['probabilities']) == {'KNN', 'Random Forest', 'Poisson'}
    assert all(np.sum(p) == pytest.approx(1) for p in prediction['probabilities'].values())
    matrix = np.array(prediction['score_matrix'])
    assert matrix.sum() == pytest.approx(1)
    assert 0 <= prediction['score_tail'] < 1e-8
    assert len(prediction['scores']) == 10
    assert prediction['scores'][0]['probability'] == pytest.approx(matrix.max())
    assert all('–' in row['score'] for row in prediction['scores'])
    assert len(prediction['expected_goals']) == 2
    assert prediction['features']
    with pytest.raises(ValueError, match='after'):
        service.predict('A', 'B', '2020-02-09')
    with pytest.raises(ValueError, match='different'):
        service.predict('A', 'A', '2020-02-10')
    with pytest.raises(ValueError, match='Unknown'):
        service.predict('A', 'Unknown', '2020-02-10')
    with pytest.raises(RuntimeError, match='Train'):
        ProjectService(csv_path).predict('A', 'B', '2020-02-10')


def test_future_season_prediction_uses_fixture_feature_contract(csv_path):
    from football_predictor.service import ProjectService
    from football_predictor.features import fixture_features
    service = ProjectService(csv_path)
    service.train()
    expected = fixture_features(service.matches, 'A', 'B', '2020-08-01').iloc[0]
    prediction = service.predict('A', 'B', '2020-08-01')
    assert prediction['features'] == {key: float(expected[key]) for key in prediction['features']}


def test_report_exports_verified_csv_and_json_without_serialized_models(csv_path, tmp_path):
    import json
    from football_predictor.service import ProjectService
    service = ProjectService(csv_path)
    destination = tmp_path / 'report'
    with pytest.raises(RuntimeError, match='Train'):
        service.export_report(destination)
    service.train()
    service.export_report(destination)
    assert {path.name for path in destination.iterdir()} == {'metrics.csv', 'tuning.csv', 'predictions_test.csv', 'report.json'}
    report = json.loads((destination / 'report.json').read_text())
    assert report['metrics'] == service.report['metrics']
    assert len(pd.read_csv(destination / 'predictions_test.csv')) == 64
    assert len(pd.read_csv(destination / 'metrics.csv')) == 4
    tuning = pd.read_csv(destination / 'tuning.csv')
    assert isinstance(json.loads(tuning.iloc[0].params), dict)


def test_metrics_use_fixed_three_class_order():
    from football_predictor.evaluation import evaluate_probabilities
    probabilities = np.array([[.8, .1, .1], [.1, .8, .1]])
    metrics, confusion = evaluate_probabilities('example', [0, 1], probabilities)
    assert metrics['accuracy'] == 1
    assert metrics['macro_f1'] == pytest.approx(2 / 3)
    assert metrics['brier'] == pytest.approx(.06)
    assert metrics['log_loss'] == pytest.approx(-np.log(.8))
    assert np.asarray(confusion).shape == (3, 3)
    with pytest.raises(ValueError):
        evaluate_probabilities('bad', [0, 1], probabilities * 2)


def test_split_keeps_whole_dates_and_strict_order():
    frame = pd.DataFrame({'date': pd.to_datetime(np.repeat(pd.date_range('2020-01-01', periods=10), 2))})
    train, validation, test = chronological_split(frame)
    assert (len(train), len(validation), len(test)) == (12, 4, 4)
    assert train.date.max() < validation.date.min() < test.date.min()
    with pytest.raises(ValueError):
        chronological_split(frame.iloc[:2])
