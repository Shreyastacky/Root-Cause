"""Create the fourth notebook without clearing any earlier executed notebook."""

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]


def main():
    md = lambda text: nbf.v4.new_markdown_cell(dedent(text).strip())
    code = lambda text: nbf.v4.new_code_cell(dedent(text).strip())
    notebook = nbf.v4.new_notebook(cells=[
        md("""
        # 04 — Does the model generalize, or recognize measurement patterns?

        The previous forest used 48 features and scenario-grouped CV. Before
        adding complexity, we test two feature ablations and a harder split.
        This notebook reruns the entire fixed audit. The data has already been
        explored: this is a diagnostic study, not a fresh final test.
        Read `docs/ROBUSTNESS_PROTOCOL.md` for the pre-run specification.
        """),
        code("""
        import json
        from pathlib import Path
        import matplotlib.pyplot as plt
        import pandas as pd
        from IPython.display import display
        from rca.robustness import OUTPUT_DIR, REFERENCE, AVAILABILITY, BEHAVIOR, run_experiment
        from rca.robustness_report import overall_table, plot_audit, build_report
        print('Availability features:', len(AVAILABILITY), '| behavior features:', len(BEHAVIOR))
        display(pd.DataFrame({'availability_only': pd.Series(AVAILABILITY),
                              'behavior_only': pd.Series(BEHAVIOR)}))
        """),
        md("""
        ## 1. Fix the question before fitting

        Availability-only uses eight finite-observation flags, without change
        values. Behavior-only uses 24 shift/variability features, without explicit
        availability/constant/onset flags. Both use the exact original scenario
        folds and the same forest settings. Missing-feature zeros can still
        expose schema, so behavior-only is not guaranteed shortcut-free.

        The harder task holds out all incidents rooted at one service, across
        every fault family and repetition. That service can still occur as a
        negative candidate in training. This tests a new fault location inside
        Online Boutique, not an entirely unseen service or application.
        """),
        code("""
        cases, summary, folds = run_experiment()
        manifest = json.loads((OUTPUT_DIR / 'manifest.json').read_text())
        source = pd.read_csv('results/supervised/incidents.csv').set_index('case_id')
        split_rows = []
        for variant in manifest['variants']:
            tested = []
            for fold in variant['folds']:
                train_ids, test_ids = fold['train_case_ids'], fold['test_case_ids']
                assert not set(train_ids) & set(test_ids)
                if variant['method'] == 'held_out_root':
                    assert not source.loc[train_ids, 'true_root'].eq(fold['held_out_root']).any()
                    assert source.loc[test_ids, 'true_root'].eq(fold['held_out_root']).all()
                else:
                    assert not set(source.loc[train_ids, 'scenario_group']) & set(source.loc[test_ids, 'scenario_group'])
                tested.extend(test_ids)
                split_rows.append({'experiment': variant['method'], 'fold': fold['fold'],
                                   'train_incidents': len(train_ids), 'test_incidents': len(test_ids),
                                   'held_out_root': fold.get('held_out_root', '')})
            assert len(tested) == len(set(tested)) == 123
        display(pd.DataFrame(split_rows))
        """),
        md("""
        ## 2. Show every result, including regressions

        The first three rows share identical folds; the last row changes the
        training task. We report per-incident root ranks, not binary accuracy
        over mostly negative candidates. The two invalid recordings stay as
        explicit misses in the all-125 table. No subset is chosen as a winner.
        """),
        code("""
        display(overall_table(summary))
        display(overall_table(summary, 'all_cases'))
        figure = plot_audit(summary, folds)
        plt.show()
        display(folds.query("method == 'held_out_root'")[
            ['held_out_root', 'n_cases', 'AC@1', 'AC@3', 'AC@5', 'Avg@5']])
        """),
        md("""
        ## 3. Inspect failures instead of hiding them in an average

        Each score is produced by a model whose training incidents exclude that
        test case. Availability-only performance is evidence that measurement
        patterns contain predictive information; it does not prove label leakage
        or causation. Root-service holdout is still within one application.
        """),
        code("""
        failures = cases.query("method == 'held_out_root' and status == 'ok' and rank_of_true_root > 1")
        display(failures[['case_id', 'fault_type', 'held_out_root', 'rank_of_true_root']])
        display(cases.query("method == 'held_out_root' and status != 'ok'")[['case_id', 'status', 'error']])
        paired = pd.read_csv(OUTPUT_DIR / 'paired_ranks.csv')
        for method in ['availability_only', 'behavior_only', 'held_out_root']:
            gains = int((paired[method].eq(1) & paired[REFERENCE].ne(1)).sum())
            losses = int((paired[method].ne(1) & paired[REFERENCE].eq(1)).sum())
            print(method, '| top-one gains vs reference:', gains, '| losses:', losses)
        """),
        md("""
        ## 4. Interpretation and evidence

        All settings are fixed. These data have already informed project design,
        so additional decisions based on this audit remain exploratory. No
        confidence interval, causal effect, online detection, or cross-system
        result is claimed. The next step is to reserve genuinely new data and
        specify that evaluation before further model selection. Short observation
        windows, robust statistics, and recording-regime sensitivity remain open.

        `scripts/verify_project.py` independently refits all 15 new forests,
        verifies candidate scores, rankings, split isolation, hashes and metrics.
        """),
        code("""
        report = build_report()
        print('Saved docs/ROBUSTNESS_AUDIT.md, figure, full scores, folds and failures.')
        """)
    ], metadata={
        'kernelspec': {'display_name': 'Python (RCA project)', 'language': 'python', 'name': 'rca-project'},
        'language_info': {'name': 'python', 'version': '3.11.9'}})
    nbf.write(notebook, ROOT / 'notebooks/04_robustness.ipynb')
    print('Created notebooks/04_robustness.ipynb')


if __name__ == '__main__':
    main()
