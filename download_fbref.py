"""Sequential, resumable FBref HTML acquisition through soccerdata; no CSV writes."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import random
import re
import time
from datetime import datetime, timezone

from lxml import html

DATA = Path.home() / 'soccerdata/data/FBref'
SEASONS = [f'{y}-{y+1}' for y in range(2019, 2025)]


def validate_html(content, filename):
    tree = html.fromstring(content)
    title = ' '.join(tree.xpath('//title/text()')).lower()
    text = tree.text_content().lower()
    if any(x in title for x in ('just a moment', 'access denied', 'attention required', 'too many requests')) or any(x in text for x in ('verify you are human', 'you have been blocked', 'rate limit exceeded')):
        raise RuntimeError('STOP: challenge, access denial, or rate limit; no retries')
    if filename.startswith('matchlogs_'):
        expected = tree.xpath('//table[@id="matchlogs_for"]')
    elif filename.startswith('match_'):
        expected = tree.xpath('//*[contains(concat(" ",normalize-space(@class)," ")," scorebox ")]')
    else:
        expected = tree.xpath('//table')
    if not expected:
        raise RuntimeError(f'STOP: missing expected data in {filename}; not caching')
    return tree


def inspect_matchlog(content):
    tree = validate_html(content, 'matchlogs_validation.html')
    matches = []
    for row in tree.xpath('//table[@id="matchlogs_for"]/tbody/tr'):
        comp = ''.join(row.xpath('./td[@data-stat="comp"]//text()')).strip()
        if comp != 'Premier League':
            continue
        links = row.xpath('./td[@data-stat="match_report"]/a/@href')
        if not links:
            raise RuntimeError('STOP: Premier League fixture has no match report')
        found = re.search(r'/matches/([a-f0-9]{8})/', links[0])
        if not found:
            raise RuntimeError('STOP: malformed match report identifier')
        matches.append({'id': found.group(1), 'url': links[0]})
    if len(matches) != 38 or len({m['id'] for m in matches}) != 38:
        raise RuntimeError(f'STOP: expected 38 distinct league matches, found {len(matches)}')
    return matches


def main():
    from soccerdata import FBref
    from soccerdata._common import _sanitize_filename
    import seleniumbase

    parser = argparse.ArgumentParser()
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    journal = DATA / 'download_2019_2025.jsonl'
    manifest = {'seasons': SEASONS, 'status': 'running', 'teams': [], 'reports': [], 'csv_written': False}

    def event(kind, **fields):
        row = {'time': datetime.now(timezone.utc).isoformat(), 'event': kind, **fields}
        with journal.open('a') as f:
            f.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)

    def checkpoint():
        p = DATA / 'download_2019_2025_manifest.json'
        tmp = p.with_suffix('.json.part')
        tmp.write_text(json.dumps(manifest, indent=2))
        tmp.replace(p)

    class CarefulFBref(FBref):
        @classmethod
        def _all_leagues(cls):
            # BaseReader keys configuration by concrete class name.
            return FBref._all_leagues()

        # Delay browser creation until an actual cache miss.
        def _init_webdriver(self):
            return None

        def get(self, url, filepath=None, max_age=None, no_cache=False, var=None):
            if filepath is not None and filepath.exists():
                content = filepath.read_bytes()
                validate_html(content, filepath.name)
                return io.BytesIO(content)
            if args.audit_only:
                raise RuntimeError(f'AUDIT: missing {filepath}')
            return self._download_and_save(url, filepath, var)

        def _download_and_save(self, url, filepath=None, var=None):
            if filepath is None or var is not None:
                raise RuntimeError('STOP: uncached or non-HTML request refused')
            if self._driver is None:
                binary = Path(seleniumbase.__file__).parent / 'drivers/chrome-linux64/chrome'
                from soccerdata._common import BaseSeleniumReader
                self.path_to_browser = str(binary)
                del self._driver
                self._driver = BaseSeleniumReader._init_webdriver(self)
                self._driver.set_page_load_timeout(90)
            # Deliberately slower than soccerdata's default, one request at a time.
            time.sleep(15 + random.uniform(0, 5))
            event('request', url=url, target=str(filepath))
            self._driver.get(url)
            time.sleep(8)
            content = self._driver.page_source.encode('utf-8')
            validate_html(content, filepath.name)
            tmp = filepath.with_suffix('.html.part')
            tmp.write_bytes(content)
            tmp.replace(filepath)
            event('saved', file=str(filepath), bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
            return io.BytesIO(content)

    reader = None
    try:
        reader = CarefulFBref(leagues='ENG-Premier League', seasons=SEASONS, data_dir=DATA, no_cache=False, no_store=False, headless=True)
        teams = reader.read_team_season_stats()
        event('teams_discovered', count=len(teams))
        if len(teams) != 120:
            raise RuntimeError(f'STOP: expected 120 team-seasons, got {len(teams)}')
        reports = {}
        for (league, code, team), row in teams.iterrows():
            season = f'20{str(code)[:2]}-20{str(code)[2:]}'
            base = row['url'].rsplit('/', 1)[0]
            if len(row['url'].split('/')) != 6:
                base += '/' + season
            entry = {'team': team, 'season': season, 'files': {}}
            for stat in ('schedule', 'shooting', 'keeper', 'misc'):
                path = DATA / f'matchlogs_{_sanitize_filename(team)}_{code}_{stat}.html'
                url = f'https://fbref.com{base}/matchlogs/all_comps/{stat}'
                with reader.get(url, path) as f:
                    content = f.read()
                matches = inspect_matchlog(content)
                entry['files'][stat] = {'path': str(path), 'matches': len(matches), 'sha256': hashlib.sha256(content).hexdigest()}
                if stat == 'schedule':
                    for m in matches:
                        reports.setdefault(season, {})[m['id']] = m['url']
            manifest['teams'].append(entry)
            checkpoint()
            event('team_verified', team=team, season=season, completed=len(manifest['teams']))
        for season, season_reports in sorted(reports.items()):
            if len(season_reports) != 380:
                raise RuntimeError(f'STOP: {season}: expected 380 unique matches, got {len(season_reports)}')
            for match_id, relative_url in sorted(season_reports.items()):
                path = DATA / f'match_{match_id}.html'
                with reader.get('https://fbref.com' + relative_url, path) as f:
                    content = f.read()
                manifest['reports'].append({'season': season, 'id': match_id, 'path': str(path), 'sha256': hashlib.sha256(content).hexdigest()})
                checkpoint()
            event('season_reports_verified', season=season, matches=len(season_reports))
        manifest['status'] = 'complete'
        checkpoint()
        event('complete', team_seasons=len(manifest['teams']), match_reports=len(manifest['reports']))
    except Exception as exc:
        manifest['status'] = 'stopped'
        manifest['error'] = str(exc)
        checkpoint()
        event('stopped', error=str(exc))
        raise
    finally:
        if reader is not None and getattr(reader, '_driver', None) is not None:
            reader._driver.quit()


if __name__ == '__main__':
    main()
