"""Independently refit the audit, verify partitions and recompute saved metrics."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = 'full_features_scenario_cv'


def verify_robustness():
    directory, upstream = ROOT / 'results/robustness', ROOT / 'results/supervised'
    manifest = json.loads((directory / 'manifest.json').read_text())
    original = json.loads((upstream / 'split_manifest.json').read_text())
    assert manifest['model_parameters'] == original['model_parameters']
    assert manifest['hash_encoding'] == 'SHA256 of UTF-8 text bytes with CRLF normalized to LF'
    for name, expected in manifest['sha256'].items():
        assert hashlib.sha256((ROOT / name).read_bytes().replace(b'\r\n', b'\n')).hexdigest() == expected, name
    features = pd.read_csv(upstream / 'service_features.csv')
    metadata = pd.read_csv(upstream / 'service_metadata.csv')
    incidents = pd.read_csv(upstream / 'incidents.csv')
    valid = incidents.loc[incidents['status'].eq('ok')].set_index('case_id')
    scores = pd.read_csv(directory / 'candidate_scores.csv')
    cases = pd.read_csv(directory / 'per_case.csv').fillna({'held_out_root': ''})
    methods = {REFERENCE, 'availability_only', 'behavior_only', 'held_out_root'}
    assert set(scores['method']) == set(cases['method']) == methods
    assert len(scores) == len(metadata) * len(methods)
    assert not scores.duplicated(['method', 'case_id', 'service']).any()
    assert scores['score'].between(0, 1).all()
    assert len(cases) == len(incidents) * len(methods)
    assert not cases.duplicated(['method', 'case_id']).any()
    # Also compare the cached feature matrix with a fresh extraction from raw data.
    from rca.supervised import build_feature_dataset
    fresh_features, fresh_metadata, _ = build_feature_dataset()
    np.testing.assert_allclose(features.to_numpy(), fresh_features.to_numpy(), rtol=1e-12, atol=1e-12)
    pd.testing.assert_frame_equal(metadata, fresh_metadata)
    ref = scores.loc[scores['method'].eq(REFERENCE)].sort_values(['case_id', 'service'])
    old = pd.read_csv(upstream / 'random_forest_oof_scores.csv').sort_values(['case_id', 'service'])
    pd.testing.assert_frame_equal(ref[old.columns].reset_index(drop=True), old.reset_index(drop=True))
    old_cases = pd.read_csv(upstream / 'per_case.csv').query("method == 'random_forest'").set_index('case_id')
    ref_cases = cases.loc[cases['method'].eq(REFERENCE)].set_index('case_id')
    assert old_cases['predicted_ranking'].equals(ref_cases.loc[old_cases.index, 'predicted_ranking'])
    refits = 0
    for variant in manifest['variants']:
        method, columns = variant['method'], variant['columns']
        if method == 'availability_only':
            expected = [c for c in features if c.endswith('__available')]
        elif method == 'behavior_only':
            expected = [c for c in features if c.split('__')[1] in ['signed_log_shift', 'relative_shift', 'log_std_ratio']]
        else:
            assert method == 'held_out_root'
            expected = features.columns.tolist()
        assert columns == expected
        if method != 'held_out_root':
            assert variant['folds'] == original['folds']
        else:
            assert {f['held_out_root'] for f in variant['folds']} == set(valid['true_root'])
        tested = []
        for fold in variant['folds']:
            train_ids, test_ids = set(fold['train_case_ids']), set(fold['test_case_ids'])
            assert not train_ids & test_ids
            assert train_ids | test_ids == set(valid.index)
            if method == 'held_out_root':
                assert not valid.loc[list(train_ids), 'true_root'].eq(fold['held_out_root']).any()
                assert valid.loc[list(test_ids), 'true_root'].eq(fold['held_out_root']).all()
                saved_cases = cases.loc[cases['method'].eq(method) & cases['fold'].eq(fold['fold'])]
                assert saved_cases['held_out_root'].eq(fold['held_out_root']).all()
            else:
                assert not set(valid.loc[list(train_ids), 'scenario_group']) & set(valid.loc[list(test_ids), 'scenario_group'])
            train = metadata['case_id'].isin(train_ids)
            test = metadata['case_id'].isin(test_ids)
            forest = RandomForestClassifier(**manifest['model_parameters'])
            forest.fit(features.loc[train, columns].to_numpy(), metadata.loc[train, 'target'].to_numpy())
            expected = metadata.loc[test, ['case_id', 'service']].copy()
            expected['expected'] = forest.predict_proba(features.loc[test, columns].to_numpy())[:, list(forest.classes_).index(1)]
            measured = scores.loc[scores['method'].eq(method) & scores['fold'].eq(fold['fold'])].merge(
                expected, on=['case_id', 'service'], validate='one_to_one')
            assert len(measured) == test.sum()
            np.testing.assert_allclose(measured['score'], measured['expected'], rtol=0, atol=1e-12)
            tested.extend(test_ids)
            refits += 1
        assert len(tested) == len(set(tested)) == len(valid)
        assert set(tested) == set(valid.index)
    indexed_incidents = incidents.set_index('case_id')
    for row in cases.itertuples():
        incident = indexed_incidents.loc[row.case_id]
        assert row.true_root == incident.true_root and row.fault_type == incident.fault_type
        assert row.n_candidates == incident.n_candidates and row.status == incident.status
        if row.status != 'ok':
            assert row.fold == -1 and json.loads(row.predicted_ranking) == [] and np.isinf(row.rank_of_true_root)
            assert not ((scores['method'] == row.method) & (scores['case_id'] == row.case_id)).any()
            continue
        candidates = scores.loc[scores['method'].eq(row.method) & scores['case_id'].eq(row.case_id)].sort_values(
            ['score', 'service'], ascending=[False, True])
        ranking = candidates['service'].tolist()
        expected_candidates = metadata.loc[metadata['case_id'].eq(row.case_id), 'service']
        assert set(ranking) == set(expected_candidates) and len(ranking) == row.n_candidates
        assert ranking == json.loads(row.predicted_ranking)
        assert ranking.index(row.true_root) + 1 == row.rank_of_true_root
        assert candidates['fold'].eq(row.fold).all()
    summary, fold_summary = pd.read_csv(directory / 'summary.csv'), pd.read_csv(directory / 'fold_summary.csv')
    for table, by_fold in [(summary, False), (fold_summary, True)]:
        for row in table.to_dict('records'):
            subset = cases.loc[cases['method'].eq(row['method'])]
            if by_fold:
                subset = subset.loc[subset['fold'].eq(row['fold'])]
            if row['evaluation_set'] == 'evaluable_only':
                subset = subset.loc[subset['status'].eq('ok')]
            if row['scope'] == 'fault_type':
                subset = subset.loc[subset['fault_type'].eq(row['fault_type'])]
            assert len(subset) == row['n_cases']
            assert subset['status'].eq('ok').sum() == row['n_evaluable']
            values = [(subset['rank_of_true_root'] <= k).mean() for k in range(1, 6)]
            np.testing.assert_allclose(values, [row[f'AC@{k}'] for k in range(1, 6)], atol=1e-12, rtol=0)
            assert np.isclose(np.mean(values), row['Avg@5'])
    expected_pairs = cases.loc[cases['status'].eq('ok')].pivot(index='case_id', columns='method', values='rank_of_true_root')
    pairs = pd.read_csv(directory / 'paired_ranks.csv').set_index('case_id')
    np.testing.assert_array_equal(pairs.loc[expected_pairs.index, expected_pairs.columns], expected_pairs)
    failures = pd.read_csv(directory / 'failures.csv')
    expected_failures = cases.loc[cases['status'].eq('ok') & cases['rank_of_true_root'].gt(1)]
    assert list(zip(failures.method, failures.case_id)) == list(zip(expected_failures.method, expected_failures.case_id))
    report = {'status': 'passed', 'forests_independently_refitted': refits,
              'candidate_scores_verified': len(scores), 'rankings_verified': len(cases),
              'summary_rows_checked': len(summary) + len(fold_summary),
              'original_predictions_unchanged': True, 'raw_features_recomputed': True,
              'input_source_hashes_verified': len(manifest['sha256'])}
    (directory / 'verification.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report


if __name__ == '__main__':
    print(json.dumps(verify_robustness(), indent=2))
