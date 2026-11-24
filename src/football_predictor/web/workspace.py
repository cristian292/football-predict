"""Single-process web workspace around the unchanged scientific service."""
from pathlib import Path
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from threading import RLock
from uuid import uuid4
from copy import deepcopy
from werkzeug.exceptions import Conflict, NotFound
from ..service import ProjectService


class Workspace:
    def __init__(self, data_path, work_dir, service_factory=None):
        self.source = Path(data_path)
        self.name = self.source.name
        self.work_dir = Path(work_dir)
        self.factory = service_factory or ProjectService
        self.lock = RLock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='football-web')
        self.jobs = {}
        self.exports = {}
        self._upload_path = None
        self.active_job = None
        self.closed = False
        self.service = self.factory(self.source)

    def state(self):
        with self.lock:
            matches = self.service.matches
            end = matches.date.max()
            teams = sorted(set(matches.home_team) | set(matches.away_team))
            return dict(ready=True, trained=self.service.report is not None, busy=self.active_job is not None,
                        active_job=self.active_job, error=None,
                        data=dict(name=self.name, n_matches=len(matches), n_teams=len(teams),
                                  teams=teams, seasons=sorted(map(int, matches.season.unique()), reverse=True),
                                  start_date=matches.date.min().strftime('%Y-%m-%d'),
                                  end_date=end.strftime('%Y-%m-%d'),
                                  default_prediction_date=(end + pd.Timedelta(days=1)).strftime('%Y-%m-%d')))

    def submit(self, kind, action):
        with self.lock:
            if self.closed:
                raise Conflict('Workspace is closed.')
            if self.active_job is not None:
                raise Conflict('Another job is already running.')
            job_id = uuid4().hex
            job = dict(id=job_id, kind=kind, status='queued', progress=0.,
                       message='În așteptare', error=None, result=None)
            self.jobs[job_id] = job
            while len(self.jobs) > 100:
                del self.jobs[next(iter(self.jobs))]
            self.active_job = job_id
            self.executor.submit(self._run, job_id, action)
            return job_id

    def _run(self, job_id, action):
        with self.lock:
            self.jobs[job_id].update(status='running', message='În lucru', progress=.05)
        def progress(message):
            with self.lock:
                job = self.jobs[job_id]
                messages = {
                    'Preparing chronological train / validation / test partitions': 'Pregătire partiții cronologice de antrenare, validare și test',
                    'Tuning KNN on validation log loss': 'Optimizare KNN pe setul de validare',
                    'Tuning Random Forest on validation log loss': 'Optimizare Random Forest pe setul de validare',
                    'Optimizing Poisson training goal likelihood with L-BFGS-B': 'Optimizare model Poisson',
                    'Refitting selected models on train + validation; scoring held-out test once': 'Evaluare modele pe setul de test separat',
                    'Refitting all-history models for strictly future predictions': 'Antrenare pe întregul istoric pentru predicții viitoare',
                    'Training and held-out evaluation complete': 'Antrenarea și evaluarea s-au încheiat',
                }
                job.update(message=messages.get(str(message), 'Procesare modele'), progress=min(.95, job['progress'] + .1))
        try:
            result = action(progress)
            with self.lock:
                self.jobs[job_id].update(status='complete', message='Operație încheiată', progress=1., result=result)
                self.active_job = None
        except Exception:
            with self.lock:
                kind = self.jobs[job_id]['kind']
                error = ('Fișier CSV invalid. Verificați coloanele și datele meciurilor.'
                         if kind in ('upload', 'reset') else 'Operația a eșuat. Verificați datele și încercați din nou.')
                self.jobs[job_id].update(status='error', message='Operație eșuată', error=error)
                self.active_job = None

    def job(self, job_id):
        with self.lock:
            if job_id not in self.jobs:
                raise NotFound('Unknown job.')
            return deepcopy(self.jobs[job_id])

    def require_service(self, trained=False):
        if self.service is None:
            raise Conflict('Load a valid dataset first.')
        if trained and self.service.report is None:
            raise Conflict('Train models first.')
        return self.service

    def replace(self, path, name):
        candidate = self.factory(path)
        with self.lock:
            previous = self._upload_path
            self.service = candidate
            self.name = name
            self._upload_path = Path(path) if Path(path) != self.source else None
            if previous is not None and previous != self._upload_path:
                try:
                    previous.unlink(missing_ok=True)
                except OSError:
                    pass  # Cleanup failure must not invalidate a successful dataset swap.
            return self.state()

    def upload(self, content, name):
        directory = self.work_dir / 'uploads'
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / (uuid4().hex + '.csv')
        path.write_bytes(content)
        try:
            return self.replace(path, name)
        except Exception:
            path.unlink(missing_ok=True)
            raise

    def export(self, service):
        from zipfile import ZipFile, ZIP_DEFLATED
        import shutil
        export_id = uuid4().hex
        directory = self.work_dir / 'exports' / export_id
        names = ('metrics.csv', 'tuning.csv', 'predictions_test.csv', 'report.json')
        try:
            service.export_report(directory)
            paths = {name: directory / name for name in names}
            if not all(path.is_file() for path in paths.values()):
                raise ValueError('Incomplete export')
            with ZipFile(directory / 'report.zip', 'w', compression=ZIP_DEFLATED) as archive:
                for name, path in paths.items():
                    archive.write(path, arcname=name)
            paths['report.zip'] = directory / 'report.zip'
            with self.lock:
                self.exports[export_id] = paths
                while len(self.exports) > 100:
                    oldest = next(iter(self.exports))
                    old_paths = self.exports.pop(oldest)
                    shutil.rmtree(next(iter(old_paths.values())).parent, ignore_errors=True)
            return dict(export_id=export_id,
                        files=[dict(name=name, url=f'/api/exports/{export_id}/{name}') for name in paths])
        except Exception:
            shutil.rmtree(directory, ignore_errors=True)
            raise

    def export_path(self, export_id, filename):
        with self.lock:
            path = self.exports.get(export_id, {}).get(filename)
            if path is None or not path.is_file():
                raise NotFound('Unknown export file.')
            return path

    def close(self):
        with self.lock:
            self.closed = True
        self.executor.shutdown(wait=True)
