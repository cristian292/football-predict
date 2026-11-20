"""Synthetic, offline contracts for reusable prediction diagnostics."""
import importlib.util

import numpy as np
import pandas as pd
import pytest


COLUMNS = ['date', 'season', 'home_team', 'away_team', 'actual', 'model',
           'variant', 'p_home', 'p_draw', 'p_away', 'predicted']
KEYS = {'per_class', 'calibration', 'confident_errors', 'by_team', 'by_season', 'summary'}


def sample():
    return pd.DataFrame([
        ['2023-01-01', 2023, 'A', 'B', 0, 'KNN', 'optimized', .8, .1, .1, 0],
        ['2023-01-02', 2023, 'B', 'C', 1, 'KNN', 'optimized', .7, .2, .1, 0],
        ['2024-01-01', 2024, 'C', 'A', 2, 'KNN', 'optimized', .1, .2, .7, 2],
    ], columns=COLUMNS)


def test_empty_frame_with_schema_returns_all_empty_diagnostics():
    assert importlib.util.find_spec('football_predictor.diagnostics') is not None
    from football_predictor.diagnostics import build_diagnostics
    assert build_diagnostics(pd.DataFrame(columns=COLUMNS)) == {key: [] for key in KEYS}


def test_per_class_exposes_draw_failure_without_input_mutation():
    from football_predictor.diagnostics import build_diagnostics
    frame = sample()
    original = frame.copy(deep=True)
    result = build_diagnostics(frame)
    assert result['per_class'] == [
        dict(variant='optimized', model='KNN', class_id=0, class_label='Home',
             support=1, precision=.5, recall=1., f1=pytest.approx(2 / 3)),
        dict(variant='optimized', model='KNN', class_id=1, class_label='Draw',
             support=1, precision=0., recall=0., f1=0.),
        dict(variant='optimized', model='KNN', class_id=2, class_label='Away',
             support=1, precision=1., recall=1., f1=1.),
    ]
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize(('column', 'value', 'message'), [
    ('actual', None, 'actual'), ('actual', .5, 'actual'), ('actual', 3, 'actual'),
    ('actual', '0', 'actual'), ('actual', True, 'actual'),
    ('predicted', 2, 'argmax'), ('predicted', .5, 'predicted'),
    ('p_home', np.nan, 'probabilit'), ('p_home', np.inf, 'probabilit'),
    ('p_home', -.1, 'probabilit'), ('p_home', .9, 'probabilit'),
    ('date', None, 'date'), ('date', '2023-02-30', 'date'),
    ('season', None, 'season'), ('season', 2023.5, 'season'),
    ('home_team', None, 'home_team'), ('away_team', '', 'away_team'),
    ('variant', None, 'variant'), ('model', None, 'model'),
])
def test_invalid_source_values_are_rejected(column, value, message):
    from football_predictor.diagnostics import build_diagnostics
    frame = sample().astype({column: object})
    frame.loc[0, column] = value
    with pytest.raises(ValueError, match=message):
        build_diagnostics(frame)


@pytest.mark.parametrize('column', COLUMNS)
def test_missing_required_columns_have_meaningful_errors(column):
    from football_predictor.diagnostics import build_diagnostics
    with pytest.raises(ValueError, match=column):
        build_diagnostics(sample().drop(columns=column))


def test_duplicate_fixture_in_model_variant_is_rejected():
    from football_predictor.diagnostics import build_diagnostics
    with pytest.raises(ValueError, match='[Dd]uplicate'):
        build_diagnostics(pd.concat([sample(), sample().iloc[:1]], ignore_index=True))


def test_top_label_calibration_bins_boundaries_and_weighted_ece():
    from football_predictor.diagnostics import build_diagnostics
    frame = sample()
    frame.loc[0, ['p_home', 'p_draw', 'p_away']] = [1., 0., 0.]
    result = build_diagnostics(frame)
    bins = result['calibration']
    assert len(bins) == 10
    assert [(row['bin_lower'], row['bin_upper']) for row in bins] == [
        (i / 10, (i + 1) / 10) for i in range(10)]
    assert [row['count'] for row in bins] == [0, 0, 0, 0, 0, 0, 0, 2, 0, 1]
    assert bins[7]['mean_confidence'] == pytest.approx(.7)
    assert bins[7]['accuracy'] == .5
    assert bins[7]['gap'] == pytest.approx(.2)
    assert bins[0]['mean_confidence'] is None
    assert bins[0]['accuracy'] is None and bins[0]['gap'] is None
    assert result['summary'][0]['ece'] == pytest.approx(.4 / 3)
    assert result['summary'][0]['n_matches'] == 3
    assert result['summary'][0]['n_errors'] == 1
    assert result['summary'][0]['draw_recall'] == 0


def test_confident_errors_preserve_source_columns_and_sort_globally():
    from football_predictor.diagnostics import build_diagnostics
    frame = sample()
    frame['source'] = pd.Series([None, None, 'archive'], dtype=object)
    second = frame.iloc[[0]].assign(variant='legacy', actual=2, p_home=.95, p_draw=.04, p_away=.01)
    frame = pd.concat([frame, second], ignore_index=True)
    errors = build_diagnostics(frame)['confident_errors']
    assert [row['confidence'] for row in errors] == [.95, .7]
    assert errors[0]['actual_label'] == 'Away' and errors[0]['predicted_label'] == 'Home'
    assert errors[1]['actual_label'] == 'Draw'
    assert all(row['source'] is None for row in errors)
    for record in errors:
        original = frame[(frame.variant == record['variant']) & (frame.date == record['date'])].iloc[0]
        assert all(record[column] == original[column] for column in COLUMNS)


def test_team_and_season_aggregation_counts_both_sides_with_finite_loss():
    from football_predictor.diagnostics import build_diagnostics
    frame = sample()
    frame.loc[1, ['p_home', 'p_draw', 'p_away']] = [1., 0., 0.]
    frame = pd.concat([frame, frame.assign(variant='legacy'),
                       frame.assign(model='Poisson')], ignore_index=True)
    result = build_diagnostics(frame)
    assert len(result['by_team']) == 9
    assert len(result['by_season']) == 6
    teams = {r['team']: r for r in result['by_team'] if r['variant'] == 'optimized' and r['model'] == 'KNN'}
    assert {name: r['n_matches'] for name, r in teams.items()} == {'A': 2, 'B': 2, 'C': 2}
    assert {name: r['accuracy'] for name, r in teams.items()} == {'A': 1., 'B': .5, 'C': .5}
    assert teams['A']['mean_log_loss'] == pytest.approx(-np.log([.8, .7]).mean())
    assert teams['B']['mean_log_loss'] == pytest.approx(-np.log([.8, np.finfo(float).eps]).mean())
    seasons = [r for r in result['by_season'] if r['variant'] == 'optimized' and r['model'] == 'KNN']
    assert [(r['season'], r['n_matches'], r['accuracy']) for r in seasons] == [(2023, 2, .5), (2024, 1, 1.)]
    assert all(np.isfinite(r['log_loss']) for r in result['by_season'])
    assert len(result['summary']) == 3
    assert all(r['accuracy'] == pytest.approx(2 / 3) for r in result['summary'])
    assert all(np.isfinite(r['log_loss']) for r in result['summary'])


@pytest.mark.parametrize('empty', [False, True])
def test_exports_are_readable_json_and_csv_with_stable_empty_headers(tmp_path, empty):
    import json
    from pathlib import Path
    import football_predictor.diagnostics as diagnostics
    assert hasattr(diagnostics, 'export_diagnostics')
    frame = sample()
    frame['source'] = pd.Series([pd.NA, np.nan, None], dtype=object)
    if empty:
        frame = frame.iloc[:0]
    destination = tmp_path / 'nested' / 'diagnostics'
    paths = diagnostics.export_diagnostics(frame, destination)
    expected = {'per_class': 'per_class.csv', 'calibration': 'calibration.csv',
                'confident_errors': 'confident_errors.csv', 'by_team': 'errors_by_team.csv',
                'by_season': 'errors_by_season.csv', 'summary': 'summary.csv',
                'diagnostics': 'diagnostics.json'}
    assert {key: Path(value).name for key, value in paths.items()} == expected
    payload = json.loads(Path(paths['diagnostics']).read_text(),
                         parse_constant=lambda x: pytest.fail(f'Nonstandard JSON constant {x}'))
    assert payload == diagnostics.build_diagnostics(frame)
    for key in KEYS:
        exported = pd.read_csv(paths[key])
        assert len(exported) == len(payload[key])
        assert {'variant', 'model'} <= set(exported.columns)
    errors = pd.read_csv(paths['confident_errors'])
    assert set(COLUMNS + ['source', 'confidence', 'actual_label', 'predicted_label']) == set(errors.columns)
    if not empty:
        assert payload['confident_errors'][0]['source'] is None
    # A second export is byte-deterministic and does not add an index column.
    before = {key: Path(value).read_bytes() for key, value in paths.items()}
    diagnostics.export_diagnostics(frame, destination)
    assert before == {key: Path(value).read_bytes() for key, value in paths.items()}
