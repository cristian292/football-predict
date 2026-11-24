"""Development-only selection checks on real synthetic football scores."""
import importlib
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, f1_score, log_loss


def api():
    return importlib.import_module('football_predictor.advanced_poisson_selection')


@pytest.fixture
def matches():
    rng = np.random.default_rng(452)
    teams = ['A', 'B', 'C', 'D']
    rows = []
    for day in range(48):
        order = np.roll(teams, day % 4)
        for home, away in ((order[0], order[1]), (order[2], order[3])):
            hg, ag = int(rng.poisson(1.7)), int(rng.poisson(1.1))
            rows.append(dict(date=pd.Timestamp('2020-01-01') + pd.Timedelta(days=day),
                             home_team=home, away_team=away, home_goals=hg, away_goals=ag,
                             result=0 if hg > ag else 1 if hg == ag else 2))
    return pd.DataFrame(rows, index=np.arange(100, 100 + len(rows)) * 3)


def rolling(identifier='rolling-test', window=10, smoothing=5.):
    return dict(id=identifier, kind='rolling_poisson',
                params=dict(window=window, smoothing=smoothing))


def folds_for(frame):
    return [(list(frame.index[:48]), list(frame.index[48:64])),
            (list(frame.index[:64]), list(frame.index[64:96]))]


@pytest.mark.parametrize('case', ['no_folds', 'empty_train', 'empty_valid', 'overlap', 'same_day',
                                        'repeat_valid', 'duplicate_train', 'missing_label', 'bool_label',
                                        'duplicate_frame', 'string_frame', 'invalid_date', 'timezone'])
def test_rejects_invalid_development_partitions(matches, case):
    frame = matches.copy()
    train, valid = list(frame.index[:48]), list(frame.index[48:64])
    folds = [(train, valid)]
    if case == 'no_folds':
        folds = []
    elif case == 'empty_train':
        folds = [([], valid)]
    elif case == 'empty_valid':
        folds = [(train, [])]
    elif case == 'overlap':
        folds = [(train, [train[-1], *valid])]
    elif case == 'same_day':
        frame.loc[train[-1], 'date'] = frame.loc[valid[0], 'date'] + pd.Timedelta(hours=1)
    elif case == 'repeat_valid':
        folds *= 2
    elif case == 'duplicate_train':
        folds = [([*train, train[0]], valid)]
    elif case == 'missing_label':
        folds = [(train, [99999])]
    elif case == 'bool_label':
        folds = [(train, [True])]
    elif case == 'duplicate_frame':
        frame.index = [frame.index[0]] * len(frame)
    elif case == 'string_frame':
        frame.index = frame.index.astype(str)
        folds = [(list(frame.index[:48]), list(frame.index[48:64]))]
    elif case == 'invalid_date':
        frame.loc[valid[0], 'date'] = pd.NaT
    elif case == 'timezone':
        frame['date'] = frame['date'].dt.tz_localize('UTC')
    with pytest.raises(ValueError):
        api().select_advanced_poisson(frame, folds, candidates=[rolling()])


def test_public_rolling_helpers_batch_whole_days_and_ignore_future(matches):
    module, candidate = api(), rolling()
    train, valid = folds_for(matches)[0]
    stamped = matches.copy()
    stamped.loc[stamped.index[1::2], 'date'] += pd.Timedelta(hours=12)
    reference_model = module.fit_poisson_candidate(candidate, matches.loc[train])
    stamped_model = module.fit_poisson_candidate(candidate, stamped.loc[train])
    np.testing.assert_allclose(stamped_model.params_, reference_model.params_, atol=1e-10)
    expected = module.poisson_candidate_probabilities(reference_model, candidate, matches, valid)
    np.testing.assert_allclose(module.poisson_candidate_probabilities(stamped_model, candidate, stamped, valid), expected)
    changed = stamped.astype({'home_goals': float, 'away_goals': float})
    last_date = matches.loc[valid, 'date'].max()
    changed.loc[matches.date >= last_date, ['home_goals', 'away_goals']] = np.inf
    np.testing.assert_allclose(module.poisson_candidate_probabilities(stamped_model, candidate, changed, valid), expected)
    earliest = valid[:2]
    changed.loc[earliest, ['home_goals', 'away_goals']] = 50.
    np.testing.assert_allclose(module.poisson_candidate_probabilities(stamped_model, candidate, changed, earliest), expected[:2])


def test_selector_caches_real_rolling_history_across_folds(matches):
    import sys
    module = api()
    calls = []
    code = module.historical_rates.__code__
    def trace(frame, event, arg):
        if event == 'call' and frame.f_code is code:
            calls.append(len(frame.f_locals['matches']))
    previous = sys.getprofile()
    try:
        sys.setprofile(trace)
        module.select_advanced_poisson(matches, folds_for(matches), candidates=[rolling()])
    finally:
        sys.setprofile(previous)
    # One causal full-history pass, plus independent training-only rates per fold.
    assert sorted(calls) == [48, 64, 96]


@pytest.mark.parametrize('correction', [False, True])
def test_real_dixon_coles_fit_predictions_and_selection(matches, correction):
    from football_predictor.models.dixon_coles import DixonColesModel
    module = api()
    candidate = dict(id='dc-real', kind='dixon_coles', params=dict(half_life_days=180., ridge=.1, use_correction=correction))
    train, valid = folds_for(matches)[0]
    model = module.fit_poisson_candidate(candidate, matches.loc[train])
    direct = DixonColesModel(**candidate['params']).fit(matches.loc[train])
    assert isinstance(model, DixonColesModel)
    assert model.optimization_['success']
    expected = direct.predict_proba(matches.loc[valid])
    np.testing.assert_allclose(module.poisson_candidate_probabilities(model, candidate, matches, valid), expected)
    changed = matches.copy()
    changed.loc[valid, ['home_goals', 'away_goals', 'result']] = 99
    np.testing.assert_allclose(module.poisson_candidate_probabilities(model, candidate, changed, valid), expected)
    report = module.select_advanced_poisson(matches, [(train, valid)], candidates=[candidate, rolling()])
    trial = next(t for t in report['trials'] if t['id'] == candidate['id'])
    assert trial['status'] == 'ok'
    assert trial['fold_metrics'][0]['optimization'] == direct.optimization_
    assert trial['validation_log_loss'] == log_loss(matches.loc[valid, 'result'], expected, labels=[0, 1, 2])
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize('failure', ['invalid_window', 'dc_nonconvergence', 'nonfinite_parameters',
                                    'nonfinite_likelihood', 'nonfinite_probability', 'bad_probability_sum'])
def test_failed_candidates_are_reported_without_suppressing_valid_models(matches, monkeypatch, failure):
    module = api()
    bad = rolling('bad')
    if failure == 'invalid_window':
        bad['params']['window'] = -1
    elif failure == 'dc_nonconvergence':
        bad = dict(id='bad', kind='dixon_coles', params=dict(half_life_days=365., ridge=.1, use_correction=True, maxiter=1))
    else:
        original = module.PoissonModel.fit
        def corrupt_fit(self, x, goals):
            model = original(self, x, goals)
            # Only the bad candidate uses this smoothing, so the valid fit remains real.
            if x.iloc[2, 0] == bad_rates.iloc[2, 0]:
                if failure == 'nonfinite_parameters':
                    model.params_[0] = np.nan
                elif failure == 'nonfinite_likelihood':
                    model.optimization_['final_nll'] = np.inf
                else:
                    predict = model.predict_proba
                    def corrupt_predict(frame):
                        p = predict(frame)
                        if failure == 'nonfinite_probability':
                            p[0, 0] = np.nan
                        else:
                            p[0] *= 2
                        return p
                    model.predict_proba = corrupt_predict
            return model
        bad['params']['smoothing'] = 3.
        train = folds_for(matches)[0][0]
        bad_rates = module.historical_rates(matches.loc[train], **bad['params'])
        monkeypatch.setattr(module.PoissonModel, 'fit', corrupt_fit)
    report = module.select_advanced_poisson(matches, [folds_for(matches)[0]], candidates=[bad, rolling('good')])
    assert report['selected']['id'] == 'good'
    trial = next(t for t in report['trials'] if t['id'] == 'bad')
    assert trial['status'] == 'error'
    assert trial['error']
    if failure != 'invalid_window':
        assert 'optimization' in trial['fold_metrics'][0]
        assert trial['fold_metrics'][0]['optimization']['n_train'] == 48
    json.dumps(report, allow_nan=False)


def test_all_failed_candidates_return_explicit_empty_selection(matches):
    bad = rolling()
    bad['params']['window'] = -1
    report = api().select_advanced_poisson(matches, folds_for(matches), candidates=[bad])
    assert report['selected'] is None
    assert report['oof_records'] == []
    assert report['trials'][0]['status'] == 'error'


def test_default_grid_runs_real_models_and_selects_deterministically(matches):
    module = api()
    messages = []
    report = module.select_advanced_poisson(matches, [folds_for(matches)[0]], progress=messages.append)
    assert len(report['trials']) == 44
    assert len(messages) == 44
    assert all(trial['status'] == 'ok' for trial in report['trials'])
    winner = min(report['trials'], key=lambda t: (-t['validation_accuracy'], -t['validation_macro_f1'], t['validation_log_loss'], t['id']))
    assert report['selected'] == winner['candidate']
    assert set(report) == {'selected', 'trials', 'oof_records', 'selection_metric'}
    # Identical real fits must use the ID, not caller order, as the last tie break.
    tied = module.select_advanced_poisson(matches, [folds_for(matches)[0]], candidates=[rolling('z'), rolling('a')])
    assert tied['selected']['id'] == 'a'
    assert tied['trials'][0]['validation_log_loss'] == tied['trials'][1]['validation_log_loss']
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize('case', ['empty', 'duplicate_ids', 'bad_id', 'unknown_kind', 'nan_params', 'missing_params'])
def test_rejects_malformed_candidate_sets(matches, case):
    candidates = [rolling()]
    if case == 'empty':
        candidates = []
    elif case == 'duplicate_ids':
        candidates *= 2
    elif case == 'bad_id':
        candidates[0]['id'] = 42
    elif case == 'unknown_kind':
        candidates[0]['kind'] = 'unknown'
    elif case == 'nan_params':
        candidates[0]['params']['smoothing'] = np.nan
    elif case == 'missing_params':
        del candidates[0]['params']
    with pytest.raises(ValueError):
        api().select_advanced_poisson(matches, folds_for(matches), candidates=candidates)


@pytest.mark.parametrize('actual', [1.5, -1., 3., np.nan])
def test_rejects_invalid_validation_labels_without_integer_coercion(matches, actual):
    matches = matches.astype({'result': float})
    matches.loc[folds_for(matches)[0][1][0], 'result'] = actual
    with pytest.raises(ValueError, match='Result labels'):
        api().select_advanced_poisson(matches, [folds_for(matches)[0]], candidates=[rolling()])


@pytest.mark.parametrize('helper', ['fit', 'predict'])
def test_public_helpers_reject_unknown_model_kind(matches, helper):
    module = api()
    candidate = rolling()
    train, valid = folds_for(matches)[0]
    model = module.fit_poisson_candidate(candidate, matches.loc[train])
    candidate['kind'] = 'not-a-model'
    with pytest.raises(ValueError, match='kind'):
        if helper == 'fit':
            module.fit_poisson_candidate(candidate, matches.loc[train])
        else:
            module.poisson_candidate_probabilities(model, candidate, matches, valid)


def test_fixed_grid_is_complete_unique_and_json_safe():
    from itertools import product
    grid = api().poisson_candidates()
    assert len(grid) == 44
    assert len({c['id'] for c in grid}) == 44
    assert all(set(c) == {'id', 'kind', 'params'} for c in grid)
    actual_rolling = {(c['params']['window'], c['params']['smoothing']) for c in grid if c['kind'] == 'rolling_poisson'}
    actual_dc = {(c['params']['half_life_days'], c['params']['ridge'], c['params']['use_correction']) for c in grid if c['kind'] == 'dixon_coles'}
    assert actual_rolling == set(product((5, 10, 20, 40, 60), (3., 5., 10., 20.)))
    assert actual_dc == set(product((180., 365., 730., None), (.01, .1, 1.), (False, True)))
    assert json.loads(json.dumps(grid, allow_nan=False)) == grid
    grid[0]['params']['window'] = 999
    assert api().poisson_candidates()[0]['params']['window'] != 999


def test_rolling_selection_returns_pooled_real_oof_scores(matches):
    module = api()
    candidate = rolling()
    folds = folds_for(matches)
    report = module.select_advanced_poisson(matches, folds, candidates=[candidate])
    assert report['selected'] == candidate
    assert report['selection_metric'] == 'pooled_validation_accuracy'
    assert len(report['oof_records']) == 48
    trial = report['trials'][0]
    assert trial['status'] == 'ok'
    assert len(trial['fold_metrics']) == 2
    records = report['oof_records']
    p = np.array([[r['p_home'], r['p_draw'], r['p_away']] for r in records])
    truth = np.array([r['actual'] for r in records])
    assert trial['validation_accuracy'] == accuracy_score(truth, p.argmax(axis=1))
    assert trial['validation_macro_f1'] == f1_score(truth, p.argmax(axis=1), labels=[0, 1, 2], average='macro', zero_division=0)
    assert trial['validation_log_loss'] == log_loss(truth, p, labels=[0, 1, 2])
    for number, (train, valid) in enumerate(folds):
        model = module.fit_poisson_candidate(candidate, matches.loc[train])
        expected = module.poisson_candidate_probabilities(model, candidate, matches, valid)
        np.testing.assert_allclose(p[sum(len(v) for _, v in folds[:number]):sum(len(v) for _, v in folds[:number + 1])], expected)
        assert trial['fold_metrics'][number]['optimization']['n_train'] == len(train)
    json.dumps(report, allow_nan=False)
