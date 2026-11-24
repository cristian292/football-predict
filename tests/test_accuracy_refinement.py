"""Synthetic validation-only refinement tests; never run a real campaign."""
from copy import deepcopy
import hashlib
import json

import pytest


def rules():
    return {
        'KNN': {'top': 3, 'neighbors': [201, 301, 501], 'smoothing': [.03, .1]},
        'Random Forest': {'top': 3, 'n_estimators': [300, 600],
                          'leaf_refinement': {'n_estimators': 300, 'min_samples_leaf': [1, 20]}},
        'Poisson': {'top_per_kind': 2,
                    'dixon_coles': {'ridge': [.003, .03], 'half_life_days': [90., 540.]},
                    'rolling_poisson': {'window': [30, 90], 'smoothing': [1., 15.]}}}


def candidate(identifier, **extra):
    return dict(id=identifier, feature_set='form', params={'n_neighbors': 31, 'weights': 'distance'},
                train_window_days=730, **extra)


def trial(config, accuracy=.5, f1=.4, loss=1., status='ok'):
    return dict(id=config['id'], candidate=config, status=status,
                validation_accuracy=accuracy, validation_macro_f1=f1, validation_log_loss=loss)


def selections():
    return {name: {'selected': config, 'trials': [], 'oof_records': []} for name, config in {
        'KNN': candidate('knn-winner'),
        'Random Forest': {'id': 'rf-winner', 'feature_set': 'strength',
                          'params': {'n_estimators': 100, 'min_samples_leaf': 5}},
        'Poisson': {'id': 'poisson-winner', 'kind': 'rolling_poisson',
                    'params': {'window': 20, 'smoothing': 5.}}}.items()}


def test_original_winners_are_always_present_as_fresh_clones():
    from football_predictor.accuracy_refinement import build_refinement_candidates
    source = selections()
    before = deepcopy(source)
    result = build_refinement_candidates(source, rules())
    assert result == {name: [selection['selected']] for name, selection in source.items()}
    result['KNN'][0]['params']['n_neighbors'] = 999
    assert source == before


def test_knn_uses_ranked_successful_top_three_independent_rule_overrides():
    from football_predictor.accuracy_refinement import build_refinement_candidates
    source = selections()
    configs = [candidate(identifier, probability_smoothing=.02) for identifier in 'abcde']
    for number, config in enumerate(configs):
        config['feature_set'] = f'family-{number}'
    source['KNN']['trials'] = [trial(configs[4], .99, status='failed'),
                             trial(configs[3], .5, .4, .8),
                             trial(configs[2], .5, .5, .9),
                             trial(configs[1], .6), trial(configs[0], .6)]
    custom = rules()
    custom['KNN']['neighbors'] = [211, 311]
    custom['KNN']['smoothing'] = [.04]
    before = deepcopy(source)
    grid = build_refinement_candidates(source, custom)['KNN']
    # a,b win the ID tiebreak; c wins macro-F1 over d; failures never compete.
    assert len(grid) == 10
    for config in configs[:3]:
        variants = [item for item in grid if item['feature_set'] == config['feature_set']]
        assert {item['params']['n_neighbors'] for item in variants} == {31, 211, 311}
        assert {(item['params']['n_neighbors'], item['probability_smoothing'])
                for item in variants} == {(211, .02), (311, .02), (31, .04)}
        assert all(item['train_window_days'] == 730 for item in variants)
    assert source == before


def test_forest_independent_tree_counts_and_leaf_refinement_preserve_decay():
    from football_predictor.accuracy_refinement import build_refinement_candidates
    source = selections()
    config = source['Random Forest']['selected']
    config['sample_half_life_days'] = 365
    config['params']['class_weight'] = 'balanced'
    source['Random Forest']['trials'] = [trial(config)]
    custom = rules()
    custom['Random Forest']['n_estimators'] = [310, 610]
    custom['Random Forest']['leaf_refinement'] = {'n_estimators': 320, 'min_samples_leaf': [2, 21]}
    grid = build_refinement_candidates(source, custom)['Random Forest']
    assert {(item['params']['n_estimators'], item['params']['min_samples_leaf']) for item in grid} == {
        (100, 5), (310, 5), (610, 5), (320, 2), (320, 21)}
    assert all(item['sample_half_life_days'] == 365 and
               item['params']['class_weight'] == 'balanced' for item in grid)


def test_poisson_refines_top_two_successes_per_kind_without_cross_products():
    from football_predictor.accuracy_refinement import build_refinement_candidates
    source = selections()
    configs = [dict(id=f'{kind}-{i}', kind=kind, params=params(i))
               for kind, params in [
                   ('dixon_coles', lambda i: dict(ridge=.1, half_life_days=365., use_correction=bool(i))),
                   ('rolling_poisson', lambda i: dict(window=10 + i, smoothing=5. + i))]
               for i in range(3)]
    source['Poisson']['trials'] = [trial(config, .8 - i * .05) for i, config in enumerate(configs)]
    custom = rules()
    custom['Poisson']['dixon_coles'] = {'ridge': [.004], 'half_life_days': [91.]}
    custom['Poisson']['rolling_poisson'] = {'window': [33], 'smoothing': [1.5]}
    grid = build_refinement_candidates(source, custom)['Poisson']
    assert len(grid) == 9
    for config in grid[1:]:
        params = config['params']
        if config['kind'] == 'dixon_coles':
            assert (params['ridge'], params['half_life_days']) in {(.004, 365.), (.1, 91.)}
        else:
            assert (params['window'], params['smoothing']) in {(33, 5.), (33, 6.), (10, 1.5), (11, 1.5)}
    assert {item['params']['use_correction'] for item in grid if item['kind'] == 'dixon_coles'} == {False, True}


def test_semantic_deduplication_preserves_winner_id_and_stable_independent_clones():
    from football_predictor.accuracy_refinement import build_refinement_candidates
    source = selections()
    winner = source['KNN']['selected']
    alias = {**deepcopy(winner), 'id': 'alias'}
    source['KNN']['trials'] = [trial(alias), trial(winner)]
    custom = rules()
    custom['KNN']['neighbors'] = [31, 301, 301]
    custom['KNN']['smoothing'] = []
    first = build_refinement_candidates(source, custom)
    source['KNN']['trials'].reverse()
    second = build_refinement_candidates(source, custom)
    assert first == second
    assert len(first['KNN']) == 2
    assert first['KNN'][0]['id'] == 'knn-winner'
    assert len({item['id'] for item in first['KNN']}) == 2
    first['KNN'][1]['params']['weights'] = 'uniform'
    assert first['KNN'][0]['params']['weights'] == 'distance'
    assert second['KNN'][1]['params']['weights'] == 'distance'


@pytest.fixture
def base_campaign(tmp_path):
    from football_predictor.accuracy_campaign import freeze_campaign_protocol
    base = tmp_path / 'base'
    data = tmp_path / 'synthetic.csv'
    data.write_text('synthetic runner fixture,not benchmark data\n')
    protocol = {'source_sha256': hashlib.sha256(data.read_bytes()).hexdigest(),
                'development_indices': [1, 2], 'test_indices': [3],
                'folds': [{'train': [1], 'validation': [2]}]}
    digest = freeze_campaign_protocol(base, protocol)
    registered = rules()
    registered['base_protocol_sha256'] = digest
    (base / 'refinement_rules.json').write_text(json.dumps(registered))
    for model, selection in selections().items():
        path = base / f"selection_{model.lower().replace(' ', '_')}.json"
        path.write_text(json.dumps({'protocol_sha256': digest, 'selection': selection}))
    return base, data, digest, registered


def test_runner_reads_only_allowlisted_inputs_and_freezes_plan_before_delegation(base_campaign, monkeypatch):
    from pathlib import Path
    from football_predictor import accuracy_campaign
    from football_predictor.accuracy_refinement import run_refinement
    base, data, digest, registered = base_campaign
    allowed = {'refinement_rules.json', 'protocol.json', 'selection_knn.json',
               'selection_random_forest.json', 'selection_poisson.json'}
    real_read = Path.read_text
    reads = []

    def guarded_read(path, *args, **kwargs):
        if path.parent == base:
            assert path.name in allowed
            reads.append(path.name)
        return real_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'read_text', guarded_read)
    calls = []
    sentinel = object()
    progress = lambda message: None

    def campaign(csv_path, output_dir, **kwargs):
        plan = json.loads((output_dir / 'refinement_plan.json').read_text())
        assert plan['base_protocol_sha256'] == digest
        assert plan['source_plan_sha256'] == hashlib.sha256(
            json.dumps(registered, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()
        assert plan['grid_counts'] == {'KNN': 1, 'Random Forest': 1, 'Poisson': 1}
        assert plan['candidate_sets'] == kwargs['candidate_sets']
        calls.append((csv_path, output_dir, kwargs))
        return sentinel

    monkeypatch.setattr(accuracy_campaign, 'run_accuracy_campaign', campaign)
    assert run_refinement(data, base, workers=2, progress=progress) is sentinel
    assert set(reads) == allowed
    assert len(calls) == 1
    assert calls[0][0:2] == (data.resolve(), base / 'refined')
    assert calls[0][2]['workers'] == 2
    assert calls[0][2]['progress'] is progress


@pytest.fixture
def reject_before_campaign(base_campaign, monkeypatch):
    from football_predictor import accuracy_campaign
    from football_predictor.accuracy_refinement import run_refinement
    base, data, _, _ = base_campaign
    calls = []
    monkeypatch.setattr(accuracy_campaign, 'run_accuracy_campaign',
                        lambda *args, **kwargs: calls.append((args, kwargs)))

    def reject(message, output=None):
        target = output if output is not None else base / 'refined'
        existed = target.exists()
        with pytest.raises(ValueError, match=message):
            run_refinement(data, base, output_dir=target)
        assert calls == []
        if not existed:
            assert not target.exists()
    return reject


@pytest.mark.parametrize('tamper', ['content', 'digest', 'missing_content', 'missing_digest', 'nan'])
def test_runner_verifies_base_protocol_content(base_campaign, reject_before_campaign, tamper):
    base, _, _, _ = base_campaign
    path = base / 'protocol.json'
    frozen = json.loads(path.read_text())
    if tamper == 'content':
        frozen['protocol']['development_indices'].append(42)
    elif tamper == 'digest':
        frozen['sha256'] = '0' * 64
    elif tamper == 'missing_content':
        del frozen['protocol']
    elif tamper == 'missing_digest':
        del frozen['sha256']
    else:
        frozen['protocol']['invalid'] = float('nan')
    path.write_text(json.dumps(frozen))
    reject_before_campaign('base protocol')


def test_runner_rejects_source_csv_with_different_bytes(base_campaign, reject_before_campaign):
    _, data, _, _ = base_campaign
    data.write_text('changed synthetic input\n')
    reject_before_campaign('source CSV')


@pytest.mark.parametrize('registered_digest', [None, '0' * 64])
def test_runner_requires_matching_registration_digest(base_campaign, reject_before_campaign, registered_digest):
    base, _, _, registered = base_campaign
    if registered_digest is None:
        del registered['base_protocol_sha256']
    else:
        registered['base_protocol_sha256'] = registered_digest
    (base / 'refinement_rules.json').write_text(json.dumps(registered))
    reject_before_campaign('registration.*digest')


@pytest.mark.parametrize('slug', ['knn', 'random_forest', 'poisson'])
@pytest.mark.parametrize('cache_digest', [None, '0' * 64])
def test_runner_requires_each_cache_to_match_protocol(base_campaign, reject_before_campaign, slug, cache_digest):
    base, _, _, _ = base_campaign
    path = base / f'selection_{slug}.json'
    cache = json.loads(path.read_text())
    if cache_digest is None:
        del cache['protocol_sha256']
    else:
        cache['protocol_sha256'] = cache_digest
    path.write_text(json.dumps(cache))
    reject_before_campaign('cache.*digest')


@pytest.mark.parametrize('slug', ['knn', 'random_forest', 'poisson'])
def test_runner_rejects_missing_cache(base_campaign, reject_before_campaign, slug):
    base, _, _, _ = base_campaign
    (base / f'selection_{slug}.json').unlink()
    reject_before_campaign('Missing.*cache')


@pytest.mark.parametrize('slug', ['knn', 'random_forest', 'poisson'])
@pytest.mark.parametrize('missing', ['selection', 'selected', 'null', 'empty'])
def test_runner_rejects_missing_selected_winner(base_campaign, reject_before_campaign, slug, missing):
    base, _, _, _ = base_campaign
    path = base / f'selection_{slug}.json'
    cache = json.loads(path.read_text())
    if missing == 'selection':
        del cache['selection']
    elif missing == 'selected':
        del cache['selection']['selected']
    else:
        cache['selection']['selected'] = None if missing == 'null' else {}
    path.write_text(json.dumps(cache))
    reject_before_campaign('selected winner')


@pytest.mark.parametrize('alias', ['direct', 'dot', 'symlink'])
def test_runner_forbids_output_resolving_to_base(base_campaign, reject_before_campaign, alias):
    base, _, _, _ = base_campaign
    before = {path.name: path.read_bytes() for path in base.iterdir()}
    output = base if alias == 'direct' else base / '.'
    if alias == 'symlink':
        output = base.parent / 'alias'
        output.symlink_to(base, target_is_directory=True)
    reject_before_campaign('output.*base', output=output)
    assert {path.name: path.read_bytes() for path in base.iterdir()} == before


@pytest.mark.parametrize('existing', ['{"different": true}', '{invalid json'])
def test_runner_preserves_existing_different_plan(base_campaign, reject_before_campaign, existing):
    base, _, _, _ = base_campaign
    output = base / 'refined'
    output.mkdir()
    plan = output / 'refinement_plan.json'
    plan.write_text(existing)
    reject_before_campaign('refinement_plan', output=output)
    assert plan.read_text() == existing
    assert list(output.iterdir()) == [plan]


def test_runner_resumes_identical_plan_without_rewriting_it(base_campaign, monkeypatch):
    from football_predictor import accuracy_campaign
    from football_predictor.accuracy_refinement import run_refinement
    base, data, _, _ = base_campaign
    calls = []
    sentinel = object()

    def campaign(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel

    monkeypatch.setattr(accuracy_campaign, 'run_accuracy_campaign', campaign)
    assert run_refinement(data, base) is sentinel
    plan_path = base / 'refined' / 'refinement_plan.json'
    # Formatting is not semantic: an interrupted run can resume the same plan.
    plan_path.write_text(json.dumps(json.loads(plan_path.read_text()), indent=1))
    before = plan_path.read_bytes()

    def unexpected_write(*args, **kwargs):
        pytest.fail('An identical frozen plan must not be rewritten')

    monkeypatch.setattr(accuracy_campaign, '_write_json', unexpected_write)
    assert run_refinement(data, base) is sentinel
    assert plan_path.read_bytes() == before
    assert len(calls) == 2
    assert calls[0] == calls[1]
