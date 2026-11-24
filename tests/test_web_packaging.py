"""Web presentation resources must survive non-editable packaging."""
from pathlib import Path
import tomllib


def test_web_templates_and_assets_are_declared_as_package_data():
    root = Path(__file__).resolve().parents[1]
    metadata = tomllib.loads((root / 'pyproject.toml').read_text())
    resources = metadata['tool']['setuptools']['package-data'].get('football_predictor.web', [])
    assert {'templates/*.html', 'static/*.css', 'static/*.js'} <= set(resources)


def test_package_version_matches_web_project_metadata():
    import football_predictor
    root = Path(__file__).resolve().parents[1]
    metadata = tomllib.loads((root / 'pyproject.toml').read_text())
    assert football_predictor.__version__ == metadata['project']['version']


def test_browser_icon_is_linked_served_and_packaged(tmp_path):
    from football_predictor.web import create_app
    app = create_app(data_path=tmp_path / 'unused.csv', work_dir=tmp_path / 'web',
                     service_factory=lambda path: object())
    try:
        client = app.test_client()
        assert b'/static/favicon.svg' in client.get('/').data
        icon = client.get('/static/favicon.svg')
        assert icon.status_code == 200
        assert icon.mimetype == 'image/svg+xml'
        assert b'<svg' in icon.data
        root = Path(__file__).resolve().parents[1]
        metadata = tomllib.loads((root / 'pyproject.toml').read_text())
        assert 'static/*.svg' in metadata['tool']['setuptools']['package-data']['football_predictor.web']
    finally:
        app.extensions['workspace'].close()
