"""Check provenance, saved results, and executed notebooks independently."""

import hashlib
import json
from pathlib import Path
import re

import nbformat
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def check_method(raw_dir, results, method, skip_constant, selected, manifest, saved, summary_rows):
    """Recompute one method without calling rca code; return #rankings checked."""
    assert set(selected['case']) == set(saved.index)
    assert len(selected) == manifest['n_cases']
    for item in manifest['files']:
        data = (raw_dir / item['path']).read_bytes()
        assert len(data) == item['bytes']
        assert hashlib.sha256(data).hexdigest() == item['sha256']
    assert not any(path.parent.name.startswith(('re1ss_', 're1tt_', 're2', 're3'))
                   for path in raw_dir.rglob('metrics.parquet'))

    # Reference calculation deliberately does not call rca.baselines or metrics.
    n_recomputed = 0
    for record in selected.itertuples():
        raw = pd.read_parquet(raw_dir / record.case / 'metrics.parquet')
        injection = int((raw_dir / record.case / 'inject_time.txt').read_text())
        row = saved.loc[record.case]
        mask = raw['time'].to_numpy() < injection
        if mask.sum() < 2 or (~mask).sum() < 1:
            assert row['status'] == 'invalid_window'
            assert json.loads(row['predicted_ranking']) == []
            assert np.isinf(row['rank_of_true_root'])
            continue
        service_scores = {}
        for column in raw.columns:
            if column == 'time':
                continue
            service = column.rsplit('_', 1)[0]
            service_scores.setdefault(service, -np.inf)
            values = raw[column].to_numpy(dtype=float)
            before, after = values[mask], values[~mask]
            before, after = before[np.isfinite(before)], after[np.isfinite(after)]
            if len(before) < 2 or len(after) < 1:
                continue
            if skip_constant and before.std(ddof=1) == 0:
                continue
            value = abs(after.mean() - before.mean()) / (before.std(ddof=1) + 1e-8)
            service_scores[service] = max(service_scores[service], value)
        ranking = sorted(service_scores, key=lambda name: (-service_scores[name], name))
        assert ranking == json.loads(row['predicted_ranking']), (method, record.case)
        assert row['rank_of_true_root'] == ranking.index(record.root_cause_service) + 1
        n_recomputed += 1

    summary = pd.read_csv(results / f'{method}_summary.csv')
    for row in summary.to_dict('records'):
        subset = saved if row['evaluation_set'] == 'all_cases' else saved.loc[saved['status'].eq('ok')]
        if row['scope'] == 'fault_type':
            subset = subset.loc[subset['fault_type'].eq(row['fault_type'])]
        assert len(subset) == row['n_cases']
        independently_measured = []
        for k in range(1, 6):
            value = sum(subset['rank_of_true_root'] <= k) / len(subset)
            assert np.isclose(value, row[f'AC@{k}'])
            independently_measured.append(value)
        assert np.isclose(np.mean(independently_measured), row['Avg@5'])
    return n_recomputed, summary_rows + len(summary)


def main():
    raw_dir, results = ROOT / 'data/raw', ROOT / 'results'
    manifest = json.loads((results / 'dataset_manifest.json').read_text())
    index = pd.read_parquet(raw_dir / 'cases.parquet')
    selected = index.loc[index['dataset'].eq('RE1-OB')]
    n_recomputed, summary_rows = {}, 0
    for method, skip_constant in [('zscore', False), ('zscore_nonconst', True)]:
        saved = pd.read_csv(results / f'{method}_per_case.csv').set_index('case_id')
        n_recomputed[method], summary_rows = check_method(
            raw_dir, results, method, skip_constant, selected, manifest, saved, summary_rows)

    notebooks = []
    for path in sorted((ROOT / 'notebooks').glob('*.ipynb')):
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        code_cells = [cell for cell in notebook.cells if cell.cell_type == 'code']
        assert all(cell.execution_count is not None for cell in code_cells)
        assert not any(output.output_type == 'error' for cell in code_cells for output in cell.outputs)
        image_count = sum('image/png' in output.get('data', {})
                          for cell in code_cells for output in cell.outputs)
        assert image_count >= (2 if path.name.startswith('01') else 1)
        notebooks.append({'name': path.name, 'executed_code_cells': len(code_cells), 'embedded_images': image_count})

    missing_links = []
    for path in [ROOT / 'README.md', *(ROOT / 'docs').glob('*.md')]:
        for target in re.findall(r'\]\(([^)]+)\)', path.read_text(encoding='utf-8')):
            if target.startswith(('https://', 'http://', '#')):
                continue
            if not (path.parent / target).exists():
                missing_links.append(f'{path.name}: {target}')
    assert not missing_links, missing_links
    outline = (ROOT / 'docs/slide_outline.md').read_text(encoding='utf-8')
    slides = re.split(r'^## \d+\.', outline, flags=re.MULTILINE)[1:]
    assert len(slides) == 8
    for slide in slides:
        count = len(re.findall(r'^- ', slide, flags=re.MULTILINE))
        assert 3 <= count <= 5, count
    report = {'status': 'passed', 'hashed_data_files': len(manifest['files']),
              'indexed_cases_checked': len(selected), 'rankings_independently_recomputed': n_recomputed,
              'invalid_window_cases': int(saved['status'].ne('ok').sum()),
              'summary_rows_independently_checked': summary_rows, 'notebooks': notebooks,
              'slides_checked': len(slides), 'missing_local_links': missing_links}
    (results / 'verification.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
