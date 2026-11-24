"""HTTP contract tests; no browser, GTK, or frontend dependency."""
import io
import time
from threading import Event

import numpy as np
import pandas as pd
import pytest
from jinja2 import DictLoader


@pytest.fixture
def csv_path(tmp_path):
    path = tmp_path / 'matches.csv'
    rows = []
    teams = ['Alpha', 'Beta', 'Gamma', 'Delta']
    for i in range(48):
        home, away = teams[i % 4], teams[(i + 1 + (i // 4) % 3) % 4]
        rows.append(dict(date=(pd.Timestamp('2024-01-01') + pd.Timedelta(days=i)).strftime('%Y-%m-%d'),
                         season=2024, venue='Home', team=home, opponent=away,
                         gf=[2, 1, 0][i % 3], ga=[0, 1, 2][i % 3]))
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


@pytest.fixture
def app(csv_path, tmp_path):
    from football_predictor.web import create_app
    application = create_app(csv_path, tmp_path / 'web')
    application.config['TESTING'] = True
    application.jinja_loader = DictLoader({'index.html': '<meta name="csrf-token" content="{{ csrf_token }}">'})
    yield application
    application.extensions['workspace'].close()


@pytest.fixture
def client(app):
    return app.test_client()


def token(client):
    return client.get('/').text.split('content="')[1].split('"')[0]


def post(client, path, **kwargs):
    return client.post(path, headers={'X-CSRF-Token': token(client)}, **kwargs)


def finished(client, response):
    assert response.status_code == 202, response.json
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        job = client.get('/api/jobs/' + response.json['job_id']).json
        if job['status'] in ('complete', 'error'):
            return job
        time.sleep(.01)
    pytest.fail('Job did not finish')


def test_root_csrf_and_security_headers(client):
    secret = token(client)
    assert len(secret) >= 32
    for path in ('/api/train', '/api/predict', '/api/upload', '/api/reset', '/api/export'):
        assert client.post(path).status_code == 403
        assert client.post(path, headers={'X-CSRF-Token': 'wrong'}).status_code == 403
    response = client.get('/api/state')
    assert response.headers['Cache-Control'] == 'no-store'
    assert response.headers['X-Content-Type-Options'] == 'nosniff'
    assert "default-src 'self'" in response.headers['Content-Security-Policy']
    assert "frame-ancestors 'none'" in response.headers['Content-Security-Policy']
    assert 'Access-Control-Allow-Origin' not in response.headers


def test_training_job_returns_real_report(client):
    assert client.get('/api/report').status_code == 409
    job = finished(client, post(client, '/api/train', json={}))
    assert job['kind'] == 'train'
    assert job['status'] == 'complete', job
    assert job['progress'] == 1
    assert job['error'] is None
    assert len(job['result']['metrics']) == 4
    assert client.get('/api/report').json == job['result']
    assert client.get('/api/state').json['trained'] is True
    assert client.get('/api/jobs/missing').status_code == 404


def test_busy_jobs_reject_concurrent_work_and_errors_are_sanitized(csv_path, tmp_path):
    from football_predictor.web import create_app
    from football_predictor.service import ProjectService
    entered, release = Event(), Event()
    class BlockingService(ProjectService):
        def train(self, progress=None):
            progress('Preparing chronological train / validation / test partitions')
            entered.set()
            release.wait(10)
            raise RuntimeError('/secret/internal.py password=hidden')
    application = create_app(csv_path, tmp_path / 'web', BlockingService)
    application.jinja_loader = DictLoader({'index.html': '<meta content="{{ csrf_token }}">'})
    c = application.test_client()
    try:
        response = post(c, '/api/train', json={})
        assert entered.wait(3)
        state = c.get('/api/state').json
        assert state['busy'] and state['active_job'] == response.json['job_id']
        job = c.get('/api/jobs/' + response.json['job_id']).json
        assert job['status'] == 'running' and 0 < job['progress'] < 1
        assert job['message'] == 'Pregătire partiții cronologice de antrenare, validare și test'
        assert post(c, '/api/train', json={}).status_code == 409
        release.set()
        job = finished(c, response)
        assert job['status'] == 'error'
        assert 'secret' not in str(job) and 'password' not in str(job)
        assert c.get('/api/state').json['busy'] is False
    finally:
        release.set()
        application.extensions['workspace'].close()


def test_prediction_validates_input_and_runs_real_models(client):
    payload = dict(home='Alpha', away='Beta', date='2024-02-18')
    assert post(client, '/api/predict', json=payload).status_code == 409
    assert finished(client, post(client, '/api/train', json={}))['status'] == 'complete'
    for change in ({'home': 'Beta'}, {'home': 'Unknown'}, {'date': '2024-02-17'},
                   {'date': '2024-2-18'}, {'date': '2024-02-30'}, {'date': None}, {'home': []}):
        assert post(client, '/api/predict', json=payload | change).status_code == 400
    assert post(client, '/api/predict', json=[]).status_code == 400
    job = finished(client, post(client, '/api/predict', json=payload))
    assert job['status'] == 'complete', job
    result = job['result']
    assert result['home'] == 'Alpha' and result['date'] == payload['date']
    assert len(result['scores']) == 10 and len(result['expected_goals']) == 2
    for values in result['probabilities'].values():
        assert sum(values) == pytest.approx(1)
    assert result['features'] and result['score_matrix']


def test_upload_is_atomic_and_reset_restores_original(client, csv_path):
    original = client.get('/api/state').json['data']
    assert post(client, '/api/upload').status_code == 400
    assert post(client, '/api/upload', data={'file': (io.BytesIO(b'x'), 'bad.txt')}).status_code == 400
    bad = finished(client, post(client, '/api/upload', data={'file': (io.BytesIO(b'not,a,dataset\n1,2,3'), 'bad.csv')}))
    assert bad['status'] == 'error'
    assert client.get('/api/state').json['data'] == original
    raw = pd.read_csv(csv_path).iloc[:30].to_csv(index=False).encode()
    good = finished(client, post(client, '/api/upload', data={'file': (io.BytesIO(raw), '../../new.csv')}))
    assert good['status'] == 'complete', good
    assert good['result']['data']['n_matches'] == 30
    assert good['result']['data']['name'] == 'new.csv'
    assert good['result']['trained'] is False
    reset = finished(client, post(client, '/api/reset', json={}))
    assert reset['status'] == 'complete'
    assert reset['result']['data'] == original
    assert client.get('/api/state').json['data'] == original


def test_standings_rank_real_fixtures_and_validate_season(client, app):
    from football_predictor.data import standings
    expected = standings(app.extensions['workspace'].service.matches).to_dict('records')
    assert client.get('/api/standings').json == {'rows': expected}
    assert client.get('/api/standings?season=2024').json == {'rows': expected}
    for value in ('no', '2024.5', '2023', ''):
        assert client.get('/api/standings?season=' + value).status_code == 400


def test_history_filters_orders_limits_and_validates(client, app):
    matches = app.extensions['workspace'].service.matches
    expected = matches.loc[matches.home_team.eq('Alpha') | matches.away_team.eq('Alpha')].sort_values('date', ascending=False)
    response = client.get('/api/history?team=Alpha&limit=3')
    assert response.status_code == 200
    assert response.json['total'] == len(expected)
    assert response.json['rows'] == [dict(date=row.date.strftime('%Y-%m-%d'), home_team=row.home_team,
                                         away_team=row.away_team, home_goals=int(row.home_goals),
                                         away_goals=int(row.away_goals)) for row in expected.head(3).itertuples()]
    assert len(client.get('/api/history?team=Alpha').json['rows']) == min(20, len(expected))
    for query in ('', '?team=Unknown', '?team=Alpha&limit=0', '?team=Alpha&limit=201',
                  '?team=Alpha&limit=1.5', '?team=Alpha&limit='):
        assert client.get('/api/history' + query).status_code == 400


def test_export_real_reports_and_whitelisted_downloads(client):
    import json
    import zipfile
    assert post(client, '/api/export', json={}).status_code == 409
    assert finished(client, post(client, '/api/train', json={}))['status'] == 'complete'
    job = finished(client, post(client, '/api/export', json={}))
    assert job['status'] == 'complete', job
    result = job['result']
    assert len(result['export_id']) == 32
    files = {entry['name']: entry['url'] for entry in result['files']}
    assert set(files) == {'metrics.csv', 'tuning.csv', 'predictions_test.csv', 'report.json', 'report.zip'}
    for name, url in files.items():
        response = client.get(url)
        assert response.status_code == 200
        assert 'attachment;' in response.headers['Content-Disposition']
        assert name in response.headers['Content-Disposition']
        assert response.data
    report = json.loads(client.get(files['report.json']).data)
    assert len(report['metrics']) == 4
    assert len(pd.read_csv(io.BytesIO(client.get(files['metrics.csv']).data))) == 4
    with zipfile.ZipFile(io.BytesIO(client.get(files['report.zip']).data)) as archive:
        assert set(archive.namelist()) == set(files) - {'report.zip'}
    base = '/api/exports/' + result['export_id'] + '/'
    for url in (base + 'other.csv', base + '../matches.csv', base + '%2e%2e%2fmatches.csv',
                '/api/exports/unknown/report.json'):
        assert client.get(url).status_code == 404


def test_upload_has_a_hard_request_size_limit(client, app):
    assert app.config['MAX_CONTENT_LENGTH'] == 10 * 1024 * 1024
    response = post(client, '/api/upload', data={'file': (io.BytesIO(b'x' * (10 * 1024 * 1024)), 'large.csv')})
    assert response.status_code == 413
    assert isinstance(response.json['error'], str)
    assert client.get('/api/state').json['busy'] is False


def test_rejects_untrusted_hosts_before_serving_csrf(client):
    for host in ('attacker.example', 'localhost.attacker.example', '127.0.0.1.attacker.example'):
        for path in ('/', '/api/state', '/healthz'):
            response = client.get(path, base_url='http://' + host)
            assert response.status_code == 400
            assert response.is_json
    for host in ('localhost:8000', '127.0.0.1:8000', '[::1]:8000'):
        assert client.get('/healthz', base_url='http://' + host).status_code == 200


def test_predict_malformed_bodies_are_json_400(client):
    for data, content_type in (('{', 'application/json'), ('null', 'application/json'),
                               ('[]', 'application/json'), ('hi', 'text/plain'), ('', None)):
        response = post(client, '/api/predict', data=data, content_type=content_type)
        assert response.status_code == 400
        assert isinstance(response.json['error'], str)


def test_unexpected_server_errors_are_sanitized_json(client, app, monkeypatch):
    def broken():
        raise RuntimeError('/secret/server.py password=hidden')
    monkeypatch.setattr(app.extensions['workspace'], 'state', broken)
    response = client.get('/api/state')
    assert response.status_code == 500
    assert isinstance(response.json['error'], str)
    assert 'secret' not in response.text and 'password' not in response.text


def test_export_failure_is_sanitized_and_removes_partial_artifacts(client, app, monkeypatch):
    workspace = app.extensions['workspace']
    workspace.service.report = {'trained': True}
    def fail(directory):
        directory.mkdir(parents=True)
        (directory / 'metrics.csv').write_text('partial')
        raise OSError('/secret/output.csv password=hidden')
    monkeypatch.setattr(workspace.service, 'export_report', fail)
    job = finished(client, post(client, '/api/export', json={}))
    assert job['status'] == 'error'
    assert 'secret' not in str(job) and 'password' not in str(job)
    assert not workspace.exports
    assert not list((workspace.work_dir / 'exports').iterdir())
    assert client.get('/api/state').json['busy'] is False


def test_state_snapshot_waits_for_workspace_lock(app):
    from threading import Thread
    workspace = app.extensions['workspace']
    entered, done = Event(), Event()
    def read():
        entered.set()
        workspace.state()
        done.set()
    with workspace.lock:
        thread = Thread(target=read)
        thread.start()
        assert entered.wait(1)
        assert not done.wait(.1)
    thread.join(2)
    assert done.is_set()


def test_job_history_is_bounded_and_snapshots_are_independent(app):
    workspace = app.extensions['workspace']
    ids = []
    for index in range(105):
        job_id = workspace.submit('check', lambda progress: {'value': 1})
        ids.append(job_id)
        deadline = time.monotonic() + 3
        while workspace.state()['busy'] and time.monotonic() < deadline:
            time.sleep(.001)
        assert workspace.job(job_id)['status'] == 'complete'
    assert len(workspace.jobs) == 100
    assert ids[0] not in workspace.jobs
    snapshot = workspace.job(ids[-1])
    snapshot['result']['value'] = 999
    assert workspace.job(ids[-1])['result']['value'] == 1
    assert workspace.job(ids[-1])['message'] == 'Operație încheiată'


def test_upload_retains_only_current_dataset_without_changing_source(client, app, csv_path):
    import hashlib
    original_hash = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    for size in (30, 36):
        raw = pd.read_csv(csv_path).iloc[:size].to_csv(index=False).encode()
        job = finished(client, post(client, '/api/upload', data={'file': (io.BytesIO(raw), 'data.csv')}))
        assert job['status'] == 'complete'
    directory = app.extensions['workspace'].work_dir / 'uploads'
    assert len(list(directory.glob('*.csv'))) == 1
    assert hashlib.sha256(csv_path.read_bytes()).hexdigest() == original_hash
    assert finished(client, post(client, '/api/reset', json={}))['status'] == 'complete'
    assert list(directory.glob('*.csv')) == []
    assert hashlib.sha256(csv_path.read_bytes()).hexdigest() == original_hash


def test_export_retention_keeps_latest_hundred_bundles(client, app):
    workspace = app.extensions['workspace']
    assert finished(client, post(client, '/api/train', json={}))['status'] == 'complete'
    exports = [workspace.export(workspace.service) for _ in range(101)]
    assert len(workspace.exports) == 100
    assert len(list((workspace.work_dir / 'exports').iterdir())) == 100
    assert client.get(exports[0]['files'][0]['url']).status_code == 404
    assert client.get(exports[-1]['files'][0]['url']).status_code == 200
    assert client.get(exports[1]['files'][0]['url']).status_code == 200


def test_close_waits_for_active_work_and_rejects_later_jobs(app):
    from threading import Thread
    from werkzeug.exceptions import Conflict
    workspace = app.extensions['workspace']
    entered, release, closed = Event(), Event(), Event()
    def action(progress):
        entered.set()
        assert release.wait(3)
        return 'done'
    job_id = workspace.submit('check', action)
    assert entered.wait(1)
    def close():
        workspace.close()
        closed.set()
    thread = Thread(target=close)
    thread.start()
    try:
        assert not closed.wait(.1)
    finally:
        release.set()
        thread.join(3)
    assert closed.is_set()
    assert workspace.job(job_id)['status'] == 'complete'
    with pytest.raises(Conflict):
        workspace.submit('check', action)
    assert workspace.state()['busy'] is False


def test_job_lifecycle_messages_are_romanian(app):
    workspace = app.extensions['workspace']
    entered, release = Event(), Event()
    def action(progress):
        entered.set()
        release.wait(3)
    try:
        with workspace.lock:
            job_id = workspace.submit('check', action)
            assert workspace.job(job_id)['message'] == 'În așteptare'
        assert entered.wait(1)
        assert workspace.job(job_id)['message'] == 'În lucru'
    finally:
        release.set()


def test_http_500_descriptions_are_not_exposed(client, app, monkeypatch):
    from werkzeug.exceptions import InternalServerError
    def broken():
        raise InternalServerError('/secret/server.log password=hidden')
    monkeypatch.setattr(app.extensions['workspace'], 'state', broken)
    response = client.get('/api/state')
    assert response.status_code == 500
    assert response.is_json
    assert 'secret' not in response.text and 'password' not in response.text


def test_state_describes_real_dataset(client):
    assert client.get('/healthz').json == {'status': 'ok'}
    state = client.get('/api/state').json
    assert state == {'ready': True, 'trained': False, 'busy': False,
                     'active_job': None, 'error': None,
                     'data': {'name': 'matches.csv', 'n_matches': 48, 'n_teams': 4,
                              'teams': ['Alpha', 'Beta', 'Delta', 'Gamma'], 'seasons': [2024],
                              'start_date': '2024-01-01', 'end_date': '2024-02-17',
                              'default_prediction_date': '2024-02-18'}}
