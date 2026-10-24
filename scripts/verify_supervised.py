"""Audit split isolation and independently refit forests against saved OOF scores."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


ROOT = Path(__file__).resolve().parents[1]


def verify_supervised():
    directory = ROOT / 'results/supervised'
    manifest = json.loads((directory / 'split_manifest.json').read_text())
    for key, path in [('protocol_sha256', ROOT / 'docs/ML_PROTOCOL.md'),
                      ('feature_source_sha256', ROOT / 'src/rca/features.py'),
                      ('experiment_source_sha256', ROOT / 'src/rca/supervised.py')]:
        assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest[key], key
    features = pd.read_csv(directory / 'service_features.csv')
    metadata = pd.read_csv(directory / 'service_metadata.csv')
    incidents = pd.read_csv(directory / 'incidents.csv')
    scores = pd.read_csv(directory / 'random_forest_oof_scores.csv')
    per_case = pd.read_csv(directory / 'per_case.csv')
    assert features.columns.tolist() == manifest['feature_columns']
    assert np.isfinite(features.to_numpy()).all()
    assert len(features) == len(metadata) == manifest['n_service_rows']
    assert not metadata.duplicated(['case_id', 'service']).any()
    assert metadata.groupby('case_id')['target'].sum().eq(1).all()
    assert len(scores) == len(metadata)
    assert not scores.duplicated(['case_id', 'service']).any()
    assert scores['score'].between(0, 1).all()
    assert set(metadata['case_id']) == set(incidents.loc[incidents['status'].eq('ok'), 'case_id'])
    source_index = pd.read_parquet(ROOT / 'data/raw/cases.parquet').set_index('case')
    for row in incidents.itertuples():
        source = source_index.loc[row.case_id]
        assert row.true_root == source.root_cause_service and row.fault_type == source.fault
        assert row.scenario_group == f'{row.true_root}|{row.fault_type}'
    expected_targets = metadata.apply(lambda row: int(row['service'] == source_index.loc[row['case_id'], 'root_cause_service']), axis=1)
    np.testing.assert_array_equal(expected_targets, metadata['target'])
    tested, refit_count = [], 0
    valid = incidents.loc[incidents['status'].eq('ok')].set_index('case_id')
    for fold in manifest['folds']:
        train_ids, test_ids = set(fold['train_case_ids']), set(fold['test_case_ids'])
        assert not train_ids & test_ids
        assert train_ids | test_ids == set(valid.index)
        train_groups = set(valid.loc[list(train_ids), 'scenario_group'])
        test_groups = set(valid.loc[list(test_ids), 'scenario_group'])
        assert not train_groups & test_groups
        assert train_groups == set(fold['train_groups']) and test_groups == set(fold['test_groups'])
        train_rows = metadata['case_id'].isin(train_ids)
        test_rows = metadata['case_id'].isin(test_ids)
        forest = RandomForestClassifier(**manifest['model_parameters'])
        forest.fit(features.loc[train_rows].to_numpy(), metadata.loc[train_rows, 'target'].to_numpy())
        expected = metadata.loc[test_rows, ['case_id', 'service']].copy()
        expected['recomputed_score'] = forest.predict_proba(features.loc[test_rows].to_numpy())[:, list(forest.classes_).index(1)]
        measured = scores.loc[scores['fold'].eq(fold['fold'])].merge(expected, on=['case_id', 'service'], validate='one_to_one')
        assert len(measured) == test_rows.sum()
        np.testing.assert_allclose(measured['score'], measured['recomputed_score'], rtol=0, atol=1e-12)
        tested.extend(test_ids)
        refit_count += 1
    assert len(tested) == len(set(tested)) == len(valid)
    assert set(tested) == set(valid.index)

    rf_rows = per_case.loc[per_case['method'].eq('random_forest')]
    for row in rf_rows.itertuples():
        if row.status != 'ok':
            assert row.fold == -1 and json.loads(row.predicted_ranking) == []
            assert np.isinf(row.rank_of_true_root)
            continue
        candidates = scores.loc[scores['case_id'].eq(row.case_id)].sort_values(['score', 'service'], ascending=[False, True])
        ranking = candidates['service'].tolist()
        assert ranking == json.loads(row.predicted_ranking)
        assert row.rank_of_true_root == ranking.index(row.true_root) + 1
        assert len(ranking) == row.n_candidates
        assert candidates['fold'].eq(row.fold).all()
    for method in ['zscore', 'zscore_nonconst']:
        original = pd.read_csv(ROOT / f'results/{method}_per_case.csv').set_index('case_id')
        comparison = per_case.loc[per_case['method'].eq(method)].set_index('case_id')
        assert set(original.index) == set(comparison.index)
        assert original['predicted_ranking'].equals(comparison.loc[original.index, 'predicted_ranking'])
    summary = pd.read_csv(directory / 'summary.csv')
    fold_summary = pd.read_csv(directory / 'fold_summary.csv')
    for table, is_fold in [(summary, False), (fold_summary, True)]:
        for row in table.to_dict('records'):
            subset = per_case.loc[per_case['method'].eq(row['method'])]
            if is_fold:
                subset = subset.loc[subset['fold'].eq(row['fold'])]
            if row['evaluation_set'] == 'evaluable_only':
                subset = subset.loc[subset['status'].eq('ok')]
            if row['scope'] == 'fault_type':
                subset = subset.loc[subset['fault_type'].eq(row['fault_type'])]
            assert len(subset) == row['n_cases']
            cutoffs = [sum(subset['rank_of_true_root'] <= k) / len(subset) for k in range(1, 6)]
            for k, value in enumerate(cutoffs, 1):
                assert np.isclose(row[f'AC@{k}'], value)
            assert np.isclose(row['Avg@5'], np.mean(cutoffs))
    report = {'status': 'passed', 'scenario_groups': manifest['n_scenario_groups'],
              'folds_independently_refitted': refit_count, 'held_out_incidents': len(valid),
              'candidate_scores_verified': len(scores), 'numeric_features': len(features.columns),
              'source_hashes_verified': True, 'summary_rows_checked': len(summary) + len(fold_summary)}
    (directory / 'verification.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report


if __name__ == '__main__':
    print(json.dumps(verify_supervised(), indent=2))
