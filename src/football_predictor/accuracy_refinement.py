"""Deterministic second-stage candidates derived only from validation selections."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def _semantic(candidate):
    return _canonical({key: value for key, value in candidate.items() if key != 'id'})


def _rank(trial):
    return (-trial['validation_accuracy'], -trial['validation_macro_f1'],
            trial['validation_log_loss'], trial['id'])


def build_refinement_candidates(selection_results, rules):
    """Return fresh grids, retaining each original validation-selected winner."""
    grids = {}
    for name, result in selection_results.items():
        grid = [deepcopy(result['selected'])]
        seen = {_semantic(grid[0])}
        successful = sorted((trial for trial in result['trials'] if trial['status'] == 'ok'), key=_rank)

        def add(config, params=None, **extra):
            clone = deepcopy(config)
            clone['params'].update(params or {})
            clone.update(extra)
            semantic = _semantic(clone)
            if semantic in seen:
                return
            seen.add(semantic)
            clone['id'] = 'refined-' + hashlib.sha256(semantic.encode()).hexdigest()
            grid.append(clone)

        if name == 'KNN':
            for trial in successful[:rules[name]['top']]:
                config = trial['candidate']
                for neighbors in rules[name]['neighbors']:
                    add(config, {'n_neighbors': neighbors})
                for smoothing in rules[name]['smoothing']:
                    add(config, probability_smoothing=smoothing)
        elif name == 'Random Forest':
            for trial in successful[:rules[name]['top']]:
                config = trial['candidate']
                for trees in rules[name]['n_estimators']:
                    add(config, {'n_estimators': trees})
                leaf_rule = rules[name]['leaf_refinement']
                for leaf in leaf_rule['min_samples_leaf']:
                    add(config, {'n_estimators': leaf_rule['n_estimators'], 'min_samples_leaf': leaf})
        elif name == 'Poisson':
            for kind in ('dixon_coles', 'rolling_poisson'):
                ranked = [trial for trial in successful if trial['candidate']['kind'] == kind]
                for trial in ranked[:rules[name]['top_per_kind']]:
                    for parameter, values in rules[name][kind].items():
                        for value in values:
                            add(trial['candidate'], {parameter: value})
        grids[name] = grid
    return grids


def run_refinement(csv_path, base_dir, output_dir=None, workers=3, progress=None):
    """Run registered validation refinement in its own frozen campaign directory.

    Only the rules, base protocol and three validation-selection caches are read
    from the base directory. Missing caches fail immediately, without polling.
    The campaign engine owns chronology, final selection and target evaluation.
    """
    from .accuracy_campaign import run_accuracy_campaign, _write_json

    base = Path(base_dir).resolve()
    source = Path(csv_path).resolve()
    output = Path(output_dir).resolve() if output_dir is not None else (base / 'refined').resolve()
    if output == base:
        raise ValueError('refinement output must differ from base directory')
    rules = json.loads((base / 'refinement_rules.json').read_text())
    frozen = json.loads((base / 'protocol.json').read_text())
    try:
        protocol = frozen['protocol']
        digest = hashlib.sha256(_canonical(protocol).encode()).hexdigest()
        if not isinstance(protocol, dict) or frozen['sha256'] != digest:
            raise ValueError('base protocol digest mismatch')
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('Invalid base protocol or digest mismatch') from exc
    if hashlib.sha256(source.read_bytes()).hexdigest() != protocol.get('source_sha256'):
        raise ValueError('source CSV digest does not match base protocol')
    if not isinstance(rules, dict) or rules.get('base_protocol_sha256') != digest:
        raise ValueError('refinement registration base protocol digest mismatch')
    selections = {}
    for name in ('KNN', 'Random Forest', 'Poisson'):
        slug = name.lower().replace(' ', '_')
        try:
            cached = json.loads((base / f'selection_{slug}.json').read_text())
        except FileNotFoundError as exc:
            raise ValueError(f'Missing {name} validation-selection cache') from exc
        if not isinstance(cached, dict) or cached.get('protocol_sha256') != digest:
            raise ValueError(f'{name} cache protocol digest mismatch')
        selection = cached.get('selection')
        winner = selection.get('selected') if isinstance(selection, dict) else None
        if not isinstance(winner, dict) or not winner:
            raise ValueError(f'{name} cache is missing a selected winner')
        selections[name] = selection
    grids = build_refinement_candidates(selections, rules)
    plan = dict(base_protocol_sha256=frozen['sha256'],
                source_plan_sha256=hashlib.sha256(_canonical(rules).encode()).hexdigest(),
                grid_counts={name: len(grid) for name, grid in grids.items()},
                candidate_sets=grids)
    plan_path = output / 'refinement_plan.json'
    if plan_path.exists():
        try:
            existing = json.loads(plan_path.read_text())
        except ValueError as exc:
            raise ValueError('Existing refinement_plan.json is invalid') from exc
        if existing != plan:
            raise ValueError('Existing refinement_plan.json differs from requested plan')
    else:
        output.mkdir(parents=True, exist_ok=True)
        _write_json(plan_path, plan)
    return run_accuracy_campaign(source, output, candidate_sets=grids,
                                 workers=workers, progress=progress)
