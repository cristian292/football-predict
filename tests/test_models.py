import numpy as np
import pandas as pd
import pytest


def history_frame():
    return pd.DataFrame({'date': pd.to_datetime(['2020-01-01', '2020-01-01', '2020-01-02', '2020-01-03']),
                         'home_team': ['A', 'C', 'A', 'B'], 'away_team': ['B', 'D', 'B', 'A'],
                         'home_goals': [3, 0, 2, 1], 'away_goals': [0, 5, 1, 2]})


def test_poisson_rolling_rates_exclude_same_day_and_future_results():
    from football_predictor.models.poisson import historical_rates
    frame = history_frame()
    rates = historical_rates(frame)
    altered = frame.copy()
    altered.loc[1:, ['home_goals', 'away_goals']] = 100
    changed = historical_rates(altered)
    np.testing.assert_allclose(rates.iloc[:2], changed.iloc[:2])
    assert (rates.to_numpy() > 0).all()
    assert not np.allclose(rates.iloc[2:], changed.iloc[2:])
    np.testing.assert_allclose(rates, historical_rates(frame.sample(frac=1, random_state=2)).loc[frame.index])


def test_poisson_optimizer_fits_only_supplied_training_rows():
    from football_predictor.models.poisson import PoissonModel
    rates = pd.DataFrame({'poisson_home': np.ones(60), 'poisson_away': np.ones(60)})
    model = PoissonModel().fit(rates, np.tile([3, 1], (60, 1)))
    expected = model.expected_goals(rates.iloc[:1])
    np.testing.assert_allclose(expected[0], [3, 1], atol=.001)
    assert model.optimization_['success']
    assert model.optimization_['final_nll'] < model.optimization_['initial_nll']
    assert model.optimization_['trace']
    assert np.allclose(model.predict_proba(rates.iloc[:2]).sum(axis=1), 1)
    assert model.optimization_['n_train'] == 60


def test_poisson_score_grid_is_normalized_and_class_ordered():
    from football_predictor.models.poisson import score_distribution
    matrix, outcomes, tail = score_distribution(2.5, .7)
    assert matrix.sum() == pytest.approx(1)
    assert 0 <= tail < 1e-8
    assert outcomes.sum() == pytest.approx(1)
    assert outcomes[0] == pytest.approx(np.tril(matrix, -1).sum())
    assert outcomes[1] == pytest.approx(np.trace(matrix))
    assert outcomes[0] > outcomes[2]
    with pytest.raises(ValueError):
        score_distribution(-1, 2)


def test_classifier_pipeline_imputes_scales_and_preserves_missing_class_order():
    from football_predictor.models import make_classifier, aligned_probabilities
    x = pd.DataFrame({'a': [1., 2., np.nan, 4.], 'b': [5., 6., 7., 8.]})
    for name, params in [('KNN', {'n_neighbors': 2, 'weights': 'distance'}),
                         ('Random Forest', {'max_depth': 3, 'min_samples_leaf': 1})]:
        model = make_classifier(name, params)
        model.fit(x, [0, 2, 0, 2])
        p = aligned_probabilities(model, x)
        assert p.shape == (4, 3)
        assert np.allclose(p.sum(axis=1), 1)
        assert np.all(p[:, 1] == 0)
        assert model.named_steps['scale'].mean_[0] == pytest.approx(2.25)  # Median-imputed [1, 2, 2, 4].
