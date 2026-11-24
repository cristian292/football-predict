"""The web default remains legacy unless an explicit matching recipe exists."""
import hashlib
import json

import pandas as pd
import pytest


@pytest.fixture
def matching_recipe(tmp_path):
    source = tmp_path / 'matches.csv'
    rows = [dict(date=(pd.Timestamp('2024-01-01') + pd.Timedelta(days=day)).strftime('%Y-%m-%d'),
                 season=2024, venue='Home', team='A' if day % 2 else 'B',
                 opponent='B' if day % 2 else 'A', gf=day % 3, ga=(day + 1) % 3)
            for day in range(30)]
    pd.DataFrame(rows).to_csv(source, index=False)
    metrics = dict(accuracy=.4, macro_f1=.3, log_loss=1.2, brier=.7, n_test=12)
    candidates = {
        'KNN': dict(id='synthetic-knn', feature_set='legacy_basic', params=dict(n_neighbors=3)),
        'Random Forest': dict(id='synthetic-rf', feature_set='strength', params=dict(n_estimators=3)),
        'Poisson': dict(id='synthetic-poisson', kind='rolling_poisson', params=dict(window=5, smoothing=3.)),
    }
    config = dict(schema_version=1, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  protocol_sha256='a' * 64, selection_cutoff='2024-01-24',
                  models={name: dict(candidate=candidate, multipliers=[1., 1., 1.],
                                     validation_metrics=metrics) for name, candidate in candidates.items()})
    recipe = tmp_path / 'explicit-recipe.json'
    recipe.write_text(json.dumps(config))
    return source, recipe, config


def test_explicit_matching_recipe_uses_optimized_service(matching_recipe):
    from football_predictor.optimized_service import OptimizedProjectService
    from football_predictor.service_factory import create_project_service
    source, recipe, config = matching_recipe
    service = create_project_service(source, config_path=recipe)
    assert type(service) is OptimizedProjectService
    assert service.source_sha256 == config['source_sha256']
    assert service.report is None


def test_changed_dataset_falls_back_without_reusing_frozen_settings(matching_recipe):
    from football_predictor.service import ProjectService
    from football_predictor.service_factory import create_project_service
    source, recipe, _ = matching_recipe
    frame = pd.read_csv(source)
    frame.loc[0, 'gf'] += 1
    frame.to_csv(source, index=False)
    service = create_project_service(source, config_path=recipe)
    assert type(service) is ProjectService
    assert int(service.matches.iloc[0].home_goals) == int(frame.iloc[0].gf)


def test_cli_explicit_recipe_reaches_real_web_workspace(matching_recipe, tmp_path, monkeypatch):
    import sys
    import types
    from football_predictor.cli import main
    from football_predictor.optimized_service import OptimizedProjectService
    source, recipe, _ = matching_recipe
    observed = []

    def serve(app, **kwargs):
        workspace = app.extensions['workspace']
        observed.append(workspace)
        assert type(workspace.service) is OptimizedProjectService
        response = app.test_client().get('/api/state')
        assert response.status_code == 200
        assert response.get_json()['data']['n_matches'] == 30

    monkeypatch.setitem(sys.modules, 'waitress', types.SimpleNamespace(serve=serve))
    assert main(['--data', str(source), '--model-config', str(recipe),
                 '--work-dir', str(tmp_path / 'web')]) == 0
    assert len(observed) == 1 and observed[0].closed


def test_evaluate_cli_applies_same_explicit_recipe(matching_recipe, tmp_path):
    from football_predictor.cli import main
    source, recipe, config = matching_recipe
    output = tmp_path / 'report'
    assert main(['--evaluate', '--data', str(source), '--model-config', str(recipe),
                 '--output', str(output)]) == 0
    report = json.loads((output / 'report.json').read_text())
    assert report['protocol_sha256'] == config['protocol_sha256']
    assert report['source_sha256'] == config['source_sha256']
    assert len(report['metrics']) == 4


@pytest.mark.parametrize('payload', ['{', '{}', '{"schema_version": true}'])
def test_existing_invalid_recipe_is_rejected_not_silently_ignored(matching_recipe, payload):
    from football_predictor.service_factory import create_project_service
    source, recipe, _ = matching_recipe
    recipe.write_text(payload)
    with pytest.raises(ValueError):
        create_project_service(source, config_path=recipe)


def test_upload_uses_legacy_and_reset_restores_matching_recipe(matching_recipe, tmp_path):
    from functools import partial
    from football_predictor.optimized_service import OptimizedProjectService
    from football_predictor.service import ProjectService
    from football_predictor.service_factory import create_project_service
    from football_predictor.web import create_app
    source, recipe, config = matching_recipe
    original = source.read_bytes()
    app = create_app(source, tmp_path / 'web',
                     service_factory=partial(create_project_service, config_path=recipe))
    workspace = app.extensions['workspace']
    try:
        assert type(workspace.service) is OptimizedProjectService
        workspace.upload(original + b'\n', 'different-bytes.csv')
        uploaded = workspace._upload_path
        assert type(workspace.service) is ProjectService
        assert workspace.state()['data']['n_matches'] == 30
        workspace.replace(workspace.source, workspace.source.name)
        assert type(workspace.service) is OptimizedProjectService
        assert workspace.service.source_sha256 == config['source_sha256']
        assert not uploaded.exists()
        assert source.read_bytes() == original
    finally:
        workspace.close()


def test_factory_without_recipe_uses_unchanged_legacy_service(tmp_path):
    from football_predictor.service import ProjectService
    from football_predictor.service_factory import create_project_service
    source = tmp_path / 'matches.csv'
    pd.DataFrame([dict(date='2020-01-01', season=2020, venue='Home', team='A',
                       opponent='B', gf=1, ga=0)]).to_csv(source, index=False)
    service = create_project_service(source, config_path=tmp_path / 'absent.json')
    assert type(service) is ProjectService
    assert len(service.matches) == 1
