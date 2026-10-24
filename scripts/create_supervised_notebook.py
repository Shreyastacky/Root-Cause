"""Create the supervised stage notebook without clearing the original notebooks."""

from pathlib import Path
import textwrap

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]


def main():
    def md(source):
        return nbf.v4.new_markdown_cell(textwrap.dedent(source).strip())

    def code(source):
        return nbf.v4.new_code_cell(textwrap.dedent(source).strip())

    cells = [
        md("""
        # 03 · First supervised model: learn to rank candidate services

        **Question:** Can a fixed Random Forest combine service-change features
        more effectively than the two heuristic rules on held-out scenarios?

        We preserve the original baselines and raw data. This notebook reruns the
        five grouped folds specified in [ML_PROTOCOL.md](../docs/ML_PROTOCOL.md).
        The dataset has already been explored, so this is exploratory within-system
        cross-validation, not a pristine final test or cross-system result.
        """),
        code("""
        from pathlib import Path
        import sys
        import os
        import json
        ROOT = Path.cwd()
        if not (ROOT / 'pyproject.toml').is_file():
            ROOT = ROOT.parent
        sys.path.insert(0, str(ROOT / 'src'))
        os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.mplconfig'))
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        from IPython.display import display, Markdown
        from rca.data import list_cases, load_case
        from rca.features import FEATURE_COLUMNS, extract_service_features
        from rca.supervised import run_experiment, OUTPUT_DIR
        from rca.ml_report import overall_table, plot_grouped_comparison, build_report
        """),
        md("""
        ## 1. Features describe behavior, not names

        Each candidate service gets 48 numbers: six features for each of eight
        measurement types. They describe availability, signed standardized mean
        change, relative mean change, variability change, constant normal behavior,
        and a zero-to-nonzero onset. Missing measurements use an explicit
        availability flag; they are not filled across the fault boundary.

        Names identify rows for reporting only. Labels are stored separately and
        sklearn receives only the numeric feature matrix. These summaries do not
        preserve the full time-series order: this is a tabular learned baseline,
        not yet a temporal encoder or GNN.
        """),
        code("""
        example = list_cases()[0]
        measurements, injection, root, _, _ = load_case(example)
        example_features = extract_service_features(measurements, injection)
        print('One row per candidate; numeric features:', len(FEATURE_COLUMNS))
        display(example_features[[name for name in FEATURE_COLUMNS if name.startswith('cpu__')]])
        """),
        md("""
        ## 2. Split scenarios before training

        A scenario is a root-service / fault-type combination. All its repeated
        incidents stay together. We use five GroupKFold folds and expand each
        training/test incident set into candidate rows only afterward. This keeps
        both the same incident and repeated scenarios out of their model's training
        set. Every evaluable incident is predicted once by a model that held it out.

        ## 3. Fit the fixed Random Forest

        In each training fold, the candidate label is 1 for the injected root and
        0 for other candidates. The forest learns combinations of feature values.
        We use 300 trees, depth at most 6, minimum leaf size 3, balanced_subsample
        class weights, seed 42, and one worker. No search or best-seed selection.

        Class-1 predict_proba scores determine ranking; they are not calibrated
        incident-level confidence. The configuration and split manifest are saved
        **before** the first fit. Original heuristic rules are evaluated on the
        same held-out incidents for a fair candidate/denominator comparison.
        """),
        code("""
        per_case, summary, fold_summary = run_experiment()
        manifest = json.loads((OUTPUT_DIR / 'split_manifest.json').read_text())
        split_rows = []
        tested = []
        for fold in manifest['folds']:
            assert not set(fold['train_case_ids']) & set(fold['test_case_ids'])
            assert not set(fold['train_groups']) & set(fold['test_groups'])
            tested.extend(fold['test_case_ids'])
            split_rows.append({'fold': fold['fold'], 'train_incidents': len(fold['train_case_ids']),
                               'test_incidents': len(fold['test_case_ids']), 'test_scenarios': len(fold['test_groups'])})
        assert len(tested) == len(set(tested)) == manifest['n_evaluable']
        display(pd.DataFrame(split_rows))
        print('All incident and scenario isolation checks passed.')
        """),
        md("""
        ## 4. Compare pooled held-out rankings and inspect variability

        The primary table uses only the 123 evaluable incidents. We separately
        retain a table for all 125 supplied incidents, where invalid recordings
        count as misses. An empty ranking for an invalid recording is not a model
        prediction. AC@k measures root localization per incident, not ordinary
        binary accuracy over the many non-root service rows.
        """),
        code("""
        display(overall_table(summary))
        display(summary.query("scope == 'overall' and evaluation_set == 'all_cases'")[
            ['method', 'n_cases', 'n_evaluable', 'AC@1', 'AC@3', 'AC@5', 'Avg@5']])
        display(fold_summary[['method', 'fold', 'n_cases', 'AC@1', 'AC@3', 'Avg@5']])
        figure = plot_grouped_comparison(summary, fold_summary)
        plt.show()
        """),
        md("""
        ## 5. What did the forest use, and where did it fail?

        These importance values are impurity-based importance averaged over the
        five training folds. They describe the fitted forests, not causal effects.
        Correlated features and measurement-schema shortcuts can affect them.
        We also list every held-out incident whose root was not ranked first.
        """),
        code("""
        importance = pd.read_csv(OUTPUT_DIR / 'feature_importance.csv')
        display(importance.head(12))
        failures = per_case.query("method == 'random_forest' and status == 'ok' and rank_of_true_root > 1")
        display(failures[['case_id', 'fault_type', 'fold', 'true_root', 'rank_of_true_root']])
        paired = pd.read_csv(OUTPUT_DIR / 'paired_case_comparison.csv')
        gained = int((paired['rf_top1'] & ~paired['heuristic_top1']).sum())
        lost = int((~paired['rf_top1'] & paired['heuristic_top1']).sum())
        print('Top-one gains versus nonconstant heuristic:', gained, '| losses:', lost)
        display(per_case.query("method == 'random_forest' and status != 'ok'")[['case_id', 'status', 'error']])
        """),
        md("""
        ## 6. Interpretation and next experiment

        The forest learns from labeled incidents and makes held-out predictions;
        its performance is now an ML baseline. This does not establish performance
        on a new application or production workload. The data was already explored,
        there are only 25 scenario groups, and feature availability may indirectly
        reveal service roles. Scores still use complete post-injection windows.

        Keep both improvements and regressions visible. Next, reserve genuinely
        new data and study feature/recording shortcuts before choosing a more
        complex temporal or graph model. An application dependency graph has not
        been built and no new fault injection was run.
        """),
        code("""
        report = build_report()
        print('Saved docs/SUPERVISED_BASELINE.md, comparison figure, failures and fold evidence.')
        """)
    ]
    notebook = nbf.v4.new_notebook(cells=cells, metadata={
        'kernelspec': {'display_name': 'Python (RCA project)', 'language': 'python', 'name': 'rca-project'},
        'language_info': {'name': 'python', 'version': '3.11.9'}})
    nbf.write(notebook, ROOT / 'notebooks/03_supervised.ipynb')
    print('Created notebooks/03_supervised.ipynb')


if __name__ == '__main__':
    main()
