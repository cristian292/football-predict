"""Flask application factory."""
from pathlib import Path
import secrets
import re
from datetime import date as calendar_date
import pandas as pd
from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException
from werkzeug.utils import secure_filename
from .workspace import Workspace


def create_app(data_path=None, work_dir=None, service_factory=None):
    root = Path(__file__).resolve().parents[3]
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024
    app.config['TRUSTED_HOSTS'] = ['localhost', '127.0.0.1', '[::1]']
    original_wsgi = app.wsgi_app

    def loopback_wsgi(environ, start_response):
        # Werkzeug 3.1.9 truncates IPv6 trusted-list entries at the first colon.
        # Canonicalize only the exact loopback literal, never arbitrary IPv6 or
        # a suffix match. Native TRUSTED_HOSTS still validates every request.
        host = environ.get('HTTP_HOST', '')
        if re.fullmatch(r'\[::1\](?::[0-9]{1,5})?', host):
            environ = environ.copy()
            environ['HTTP_HOST'] = host.replace('[::1]', 'localhost', 1)
        return original_wsgi(environ, start_response)

    app.wsgi_app = loopback_wsgi
    workspace = Workspace(data_path or root / 'matches.csv', work_dir or root / 'data/webapp', service_factory)
    app.extensions['workspace'] = workspace
    csrf_token = secrets.token_urlsafe(32)

    @app.before_request
    def protect_mutations():
        if request.method not in ('GET', 'HEAD', 'OPTIONS'):
            supplied = request.headers.get('X-CSRF-Token', '')
            if not secrets.compare_digest(supplied.encode(), csrf_token.encode()):
                return jsonify(error='Missing or invalid CSRF token.'), 403

    @app.after_request
    def security_headers(response):
        response.headers['Content-Security-Policy'] = ("default-src 'self'; script-src 'self'; style-src 'self'; "
                                                     "frame-ancestors 'none'; base-uri 'self'; form-action 'self'")
        response.headers['X-Content-Type-Options'] = 'nosniff'
        if request.path.startswith('/api/') or request.path == '/':
            response.headers['Cache-Control'] = 'no-store'
        return response

    @app.errorhandler(HTTPException)
    def http_error(error):
        message = 'Operația a eșuat. Încercați din nou.' if error.code >= 500 else error.description
        return jsonify(error=message), error.code

    @app.errorhandler(Exception)
    def unexpected_error(error):
        app.logger.exception('Unexpected web API failure')
        return jsonify(error='Operația a eșuat. Încercați din nou.'), 500

    @app.get('/')
    def index():
        return render_template('index.html', csrf_token=csrf_token)

    @app.get('/healthz')
    def health():
        return jsonify(status='ok')

    @app.get('/api/state')
    def state():
        return jsonify(workspace.state())

    @app.get('/api/standings')
    def league_standings():
        from ..data import standings
        season = request.args.get('season')
        with workspace.lock:
            matches = workspace.require_service().matches
            if season is not None:
                if not re.fullmatch(r'\d{4}', season) or int(season) not in matches.season.unique():
                    return jsonify(error='Select a known season.'), 400
                season = int(season)
            return jsonify(rows=standings(matches, season=season).to_dict('records'))

    @app.get('/api/history')
    def history():
        team = request.args.get('team')
        limit = request.args.get('limit', '20')
        if not re.fullmatch(r'[0-9]{1,3}', limit) or not 1 <= int(limit) <= 200:
            return jsonify(error='Limit must be an integer from 1 to 200.'), 400
        with workspace.lock:
            matches = workspace.require_service().matches
            if team not in set(matches.home_team) | set(matches.away_team):
                return jsonify(error='Select a known team.'), 400
            selected = matches.loc[matches.home_team.eq(team) | matches.away_team.eq(team)]
            rows = [dict(date=row.date.strftime('%Y-%m-%d'), home_team=row.home_team,
                         away_team=row.away_team, home_goals=int(row.home_goals), away_goals=int(row.away_goals))
                    for row in selected.sort_values('date', ascending=False, kind='stable').head(int(limit)).itertuples()]
            return jsonify(rows=rows, total=len(selected))

    @app.post('/api/train')
    def train():
        with workspace.lock:
            service = workspace.require_service()
            job_id = workspace.submit('train', lambda progress: service.train(progress=progress))
        return jsonify(job_id=job_id), 202

    @app.get('/api/jobs/<job_id>')
    def job(job_id):
        return jsonify(workspace.job(job_id))

    @app.get('/api/report')
    def report():
        with workspace.lock:
            return jsonify(workspace.require_service(trained=True).report)

    @app.post('/api/predict')
    def predict():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify(error='Expected a JSON object.'), 400
        home, away, day = (payload.get(key) for key in ('home', 'away', 'date'))
        with workspace.lock:
            service = workspace.require_service(trained=True)
            teams = set(service.matches.home_team) | set(service.matches.away_team)
            if not isinstance(home, str) or not isinstance(away, str) or home == away or home not in teams or away not in teams:
                return jsonify(error='Select two different known teams.'), 400
            if not isinstance(day, str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2}', day):
                return jsonify(error='Use a valid date in YYYY-MM-DD format.'), 400
            try:
                parsed = calendar_date.fromisoformat(day)
                valid = parsed > service.matches.date.max().date()
            except ValueError:
                valid = False
            if not valid:
                return jsonify(error='Prediction date must be valid and after the dataset end date.'), 400
            job_id = workspace.submit('predict', lambda progress: service.predict(home, away, day))
        return jsonify(job_id=job_id), 202

    @app.post('/api/upload')
    def upload():
        file = request.files.get('file')
        name = secure_filename(file.filename or '') if file else ''
        if not name or Path(name).suffix.lower() != '.csv':
            return jsonify(error='Upload a CSV file using the file field.'), 400
        content = file.read()
        with workspace.lock:
            job_id = workspace.submit('upload', lambda progress: workspace.upload(content, name))
        return jsonify(job_id=job_id), 202

    @app.post('/api/export')
    def export():
        with workspace.lock:
            service = workspace.require_service(trained=True)
            job_id = workspace.submit('export', lambda progress: workspace.export(service))
        return jsonify(job_id=job_id), 202

    @app.get('/api/exports/<export_id>/<filename>')
    def export_file(export_id, filename):
        from flask import send_file
        with workspace.lock:
            return send_file(workspace.export_path(export_id, filename), as_attachment=True, download_name=filename)

    @app.post('/api/reset')
    def reset():
        job_id = workspace.submit('reset', lambda progress: workspace.replace(workspace.source, workspace.source.name))
        return jsonify(job_id=job_id), 202

    return app
