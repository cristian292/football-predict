"""Small deterministic, synthetic checks; no downloaded fixtures or benchmarks."""
import json

import numpy as np
import pandas as pd
import pytest

from football_predictor.models.dixon_coles import DixonColesModel


def training_frame():
    return pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=24, freq='14D'),
        'home_team': ['A', 'B', 'A', 'C', 'B', 'C'] * 4,
        'away_team': ['B', 'A', 'C', 'A', 'C', 'B'] * 4,
        'home_goals': [3, 0, 4, 0, 2, 1] * 4,
        'away_goals': [0, 2, 0, 3, 1, 1] * 4,
    })


def test_fit_learns_team_strengths_with_finite_serializable_diagnostics():
    model = DixonColesModel(use_correction=False).fit(training_frame())
    assert isinstance(model, DixonColesModel)
    assert model.optimization_['success']
    assert model.optimization_['iterations'] > 0
    assert model.optimization_['final_nll'] < model.optimization_['initial_nll']
    assert model.optimization_['rho'] == 0.0
    assert model.optimization_['training_cutoff'].startswith('2023-11-19')
    json.dumps(model.optimization_, allow_nan=False)
    fixtures = pd.DataFrame({'home_team': ['A', 'B'], 'away_team': ['B', 'A']})
    rates = model.expected_goals(fixtures)
    assert rates.shape == (2, 2)
    assert np.isfinite(rates).all() and (rates > 0).all()
    assert rates[0, 0] > rates[1, 0]
    assert rates[1, 1] > rates[0, 1]


def test_correction_is_fitted_from_low_score_dependence():
    frame = training_frame()
    frame['home_goals'] = [0, 1, 0, 1, 2, 2] * 4
    frame['away_goals'] = [0, 1, 0, 1, 1, 2] * 4
    model = DixonColesModel().fit(frame)
    assert model.optimization_['success'], model.optimization_
    assert model.rho_ < -0.05
    assert model.optimization_['rho'] == model.rho_
    assert model.optimization_['final_nll'] < model.optimization_['initial_nll']
    rates = model.expected_goals(frame)
    assert (1 - rates.prod(axis=1) * model.rho_ > 0).all()
    assert (1 + rates * model.rho_ > 0).all()
    assert 1 - model.rho_ > 0


def test_zero_rho_exactly_matches_existing_independent_score_distribution():
    from football_predictor.models.poisson import score_distribution

    model = DixonColesModel(use_correction=False).fit(training_frame())
    for rates in [(1.8, 0.6), (0.0, 0.0), (0.001, 20.0)]:
        actual = model.score_distribution(*rates)
        expected = score_distribution(*rates)
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])
        assert actual[2] == expected[2]


def test_low_score_tau_correction_uses_home_rows_and_away_columns():
    from football_predictor.models.poisson import score_distribution

    model = DixonColesModel()
    model.rho_ = -0.15
    lam, mu = 1.8, 0.7
    independent, independent_outcomes, tail = score_distribution(lam, mu)
    expected = independent.copy()
    expected[:2, :2] *= [[1 - lam * mu * model.rho_, 1 + lam * model.rho_],
                         [1 + mu * model.rho_, 1 - model.rho_]]
    expected /= expected.sum()
    matrix, outcomes, actual_tail = model.score_distribution(lam, mu)
    np.testing.assert_allclose(matrix, expected, rtol=1e-14, atol=1e-16)
    np.testing.assert_allclose(outcomes, [np.tril(matrix, -1).sum(),
                                       np.trace(matrix), np.triu(matrix, 1).sum()])
    assert outcomes[1] > independent_outcomes[1]
    assert outcomes[0] > outcomes[2]
    assert actual_tail == tail


def test_predictions_ignore_observed_goals_dates_and_handle_neutral_unknown_teams():
    model = DixonColesModel().fit(training_frame())
    fixtures = pd.DataFrame({'home_team': ['A', 'B', 'unknown', 'other'],
                             'away_team': ['B', 'A', 'other', 'unknown']})
    probabilities = model.predict_proba(fixtures)
    assert probabilities.shape == (4, 3)
    assert np.isfinite(probabilities).all() and (probabilities > 0).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-14)
    assert probabilities[0, 0] > probabilities[0, 2]
    assert probabilities[1, 2] > probabilities[1, 0]
    np.testing.assert_array_equal(probabilities[2], probabilities[3])
    neutral = model.expected_goals(fixtures.iloc[2:])
    np.testing.assert_allclose(neutral, np.tile(np.exp([model.params_[0] + model.params_[1],
                                                       model.params_[0]]), (2, 1)))
    poisoned = fixtures.assign(home_goals=[np.nan, 999, -100, 'bad'],
                               away_goals=[-2, 'bad', np.inf, 1000], date='2099-01-01')
    np.testing.assert_array_equal(probabilities, model.predict_proba(poisoned))
    np.testing.assert_array_equal(model.expected_goals(fixtures), model.expected_goals(poisoned))


@pytest.mark.parametrize('rho,rates', [(-0.9, (30.0, 0.2)), (0.9, (20.0, 30.0)),
                                     (-0.8, (500.0, 1e-12)), (0.8, (0.0, 0.0)),
                                     (-0.9, (np.float64(1e-320), np.float64(1e-320)))])
def test_extreme_rates_clip_effective_rho_not_individual_cells(rho, rates):
    from football_predictor.models.poisson import score_distribution

    model = DixonColesModel()
    model.rho_ = rho
    matrix, outcomes, tail = model.score_distribution(*rates)
    assert np.isfinite(matrix).all() and (matrix >= 0).all()
    assert np.isfinite(outcomes).all() and (outcomes >= 0).all()
    np.testing.assert_allclose(matrix.sum(), 1.0, atol=1e-14)
    np.testing.assert_allclose(outcomes.sum(), 1.0, atol=1e-14)
    assert 0 <= tail < 1e-9
    effective = model.last_effective_rho_
    lam, mu = rates
    tau = np.array([[1 - lam * mu * effective, 1 + lam * effective],
                    [1 + mu * effective, 1 - effective]])
    assert (tau > 0).all()
    independent = score_distribution(*rates)[0]
    independent[:2, :2] *= tau
    independent /= independent.sum()
    np.testing.assert_allclose(matrix, independent, atol=1e-16)
    assert model.rho_ == rho


@pytest.mark.parametrize('kwargs', [
    {'half_life_days': 0}, {'half_life_days': -1}, {'half_life_days': np.nan},
    {'half_life_days': np.inf}, {'half_life_days': True}, {'half_life_days': '365'},
    {'ridge': -1}, {'ridge': np.nan}, {'ridge': np.inf}, {'ridge': True},
    {'maxiter': 0}, {'maxiter': -2}, {'maxiter': 2.5}, {'maxiter': True},
    {'use_correction': 'yes'}, {'use_correction': 1},
])
def test_invalid_hyperparameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        DixonColesModel(**kwargs)


@pytest.mark.parametrize('column,value', [
    ('home_goals', -1), ('away_goals', 0.5), ('home_goals', np.inf),
    ('away_goals', np.nan), ('home_goals', 'invalid'),
    ('date', None), ('date', 'not-a-date'), ('home_team', None),
    ('away_team', ''), ('home_team', 123),
])
def test_invalid_training_rows_are_rejected_with_value_error(column, value):
    frame = training_frame().astype({'home_goals': object, 'away_goals': object,
                                    'date': object})
    frame.loc[0, column] = value
    with pytest.raises(ValueError):
        DixonColesModel().fit(frame)


@pytest.mark.parametrize('kind', ['empty', 'missing_goals', 'duplicate_columns', 'same_team'])
def test_malformed_training_frame_is_rejected(kind):
    frame = training_frame()
    if kind == 'empty':
        frame = frame.iloc[:0]
    elif kind == 'missing_goals':
        frame = frame.drop(columns='away_goals')
    elif kind == 'duplicate_columns':
        frame = pd.concat([frame, frame[['home_team']]], axis=1)
    else:
        frame.loc[0, 'away_team'] = frame.loc[0, 'home_team']
    with pytest.raises(ValueError):
        DixonColesModel().fit(frame)


def test_decay_uses_training_end_only_including_weighted_initial_likelihood():
    from scipy.special import gammaln

    frame = training_frame().iloc[:6].copy()
    model = DixonColesModel(half_life_days=14, use_correction=False).fit(frame)
    weights = np.exp2(-np.arange(5, -1, -1))
    np.testing.assert_allclose(model.sample_weight_, weights)
    goals = frame[['home_goals', 'away_goals']].to_numpy()
    mean = np.average(goals, weights=weights, axis=0)
    expected_nll = np.average(np.sum(mean - goals * np.log(mean) + gammaln(goals + 1), axis=1),
                              weights=weights)
    assert model.optimization_['initial_nll'] == pytest.approx(expected_nll)
    shifted = frame.assign(date=frame['date'] + pd.Timedelta(days=700))
    other = DixonColesModel(half_life_days=14, use_correction=False).fit(shifted)
    np.testing.assert_allclose(model.params_, other.params_, atol=1e-12)
    uniform = DixonColesModel(half_life_days=None, use_correction=False).fit(frame)
    np.testing.assert_array_equal(uniform.sample_weight_, np.ones(len(frame)))
    assert not np.allclose(model.expected_goals(frame), uniform.expected_goals(frame))


def test_unpenalized_strengths_have_an_explicit_zero_mean_gauge():
    model = DixonColesModel(ridge=0.0, use_correction=False).fit(training_frame())
    n = len(model.teams_)
    assert np.mean(model.params_[2:2 + n]) == pytest.approx(0.0, abs=1e-12)
    assert np.mean(model.params_[2 + n:2 + 2 * n]) == pytest.approx(0.0, abs=1e-12)
    assert np.isfinite(model.expected_goals(training_frame())).all()


@pytest.mark.parametrize('correction', [False, True])
def test_iteration_limit_is_distinguished_from_convergence(correction):
    with pytest.warns(RuntimeWarning, match='did not converge'):
        model = DixonColesModel(use_correction=correction, maxiter=1).fit(training_frame())
    assert not model.optimization_['success']
    assert model.optimization_['status'] == 'not_converged'
    assert model.optimization_['iterations'] == 1
    assert model.optimization_['final_nll'] == pytest.approx(model._objective(model.params_)[0])
    json.dumps(model.optimization_, allow_nan=False)
    probabilities = model.predict_proba(training_frame())
    assert np.isfinite(probabilities).all() and (probabilities >= 0).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)


def central_jacobian(function, point, step=1e-6):
    directions = np.eye(len(point)) * step
    return np.asarray([(function(point + direction) - function(point - direction)) / (2 * step)
                       for direction in directions]).T


def test_tau_constraint_analytic_jacobian_matches_central_differences():
    model = DixonColesModel().fit(training_frame())
    params = np.linspace(-0.2, 0.3, len(model.params_))
    params[-1] = -0.07
    actual = model._tau_constraint_jacobian(params)
    expected = central_jacobian(model._tau_constraints, params)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=2e-8)


@pytest.mark.parametrize('method', ['expected_goals', 'predict_proba'])
def test_predicting_before_fit_raises_clear_runtime_error(method):
    with pytest.raises(RuntimeError, match='fit'):
        getattr(DixonColesModel(), method)(training_frame())


@pytest.mark.parametrize('rho', [None, 0.0, -0.12, 0.13, -0.99, 0.99])
def test_likelihood_analytic_gradient_matches_central_differences(rho):
    # Derivative verification of the objective already exercised by fit tests,
    # including finite penalty/continuation for infeasible line-search points.
    frame = training_frame().assign(home_goals=[0, 0, 1, 1, 2, 3] * 4,
                                    away_goals=[0, 1, 0, 1, 0, 2] * 4)
    model = DixonColesModel(use_correction=rho is not None).fit(frame)
    params = np.linspace(-0.1, 0.2, len(model.params_))
    if rho is not None:
        params[-1] = rho
    loss, gradient = model._objective(params)
    step = 1e-5
    numerical = central_jacobian(lambda value: model._objective(value)[0], params, step=step)
    assert np.isfinite(loss) and np.isfinite(gradient).all()
    # Infeasible continuations have large loss; account for cancellation of
    # nearly equal float64 objectives rather than relaxing relative accuracy.
    cancellation = 5 * np.finfo(float).eps * abs(loss) / step
    np.testing.assert_allclose(gradient, numerical, rtol=1e-5, atol=max(2e-6, cancellation))


def test_tiny_valid_half_life_and_zero_goals_remain_numerically_finite():
    frame = training_frame().assign(home_goals=0, away_goals=0)
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        model = DixonColesModel(half_life_days=1e-320).fit(frame)
        rates = model.expected_goals(frame)
        probabilities = model.predict_proba(frame)
    assert np.isfinite(rates).all() and (rates > 0).all()
    assert np.isfinite(probabilities).all() and (probabilities >= 0).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    assert model.sample_weight_[-1] == 1.0
    assert model.sample_weight_[:-1].sum() == 0.0


@pytest.mark.parametrize('kind', ['missing_team', 'null_team', 'blank_team', 'duplicate_columns'])
def test_prediction_validates_fixture_team_columns(kind):
    model = DixonColesModel(use_correction=False).fit(training_frame())
    frame = training_frame()[['home_team', 'away_team']].copy()
    if kind == 'missing_team':
        frame = frame.drop(columns='home_team')
    elif kind == 'null_team':
        frame.loc[0, 'home_team'] = None
    elif kind == 'blank_team':
        frame.loc[0, 'away_team'] = '  '
    else:
        frame = pd.concat([frame, frame[['home_team']]], axis=1)
    with pytest.raises(ValueError):
        model.expected_goals(frame)
