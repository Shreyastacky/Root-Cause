"""Create the two student-facing notebooks; this does not execute them."""

from pathlib import Path
import textwrap

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]


def md(source):
    return nbf.v4.new_markdown_cell(textwrap.dedent(source).strip())


def code(source):
    return nbf.v4.new_code_cell(textwrap.dedent(source).strip())


SETUP = """
from pathlib import Path
import os
import sys
ROOT = Path.cwd()
if not (ROOT / 'pyproject.toml').is_file():
    ROOT = ROOT.parent
assert (ROOT / 'src/rca').is_dir(), 'Open this notebook from the project folder.'
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.mplconfig'))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
from rca.data import list_cases, load_case, service_names
from rca.analysis import configure_plots
np.random.seed(42)
configure_plots()
RESULTS = ROOT / 'results'
RESULTS.mkdir(exist_ok=True)
"""


def save(name, cells):
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata = {"kernelspec": {"display_name": "Python (RCA project)",
                                      "language": "python", "name": "rca-project"},
                         "language_info": {"name": "python", "version": "3.11.9"}}
    path = ROOT / 'notebooks' / name
    path.parent.mkdir(exist_ok=True)
    nbf.write(notebook, path)


def main():
    save('01_eda.ipynb', [
        md("""
        # 01 · Inspect the failure data

        **Question:** What was recorded, and why might the largest anomaly belong
        to a service other than the one where the fault was injected?

        We use only the official RCAEval **RE1-OB** metrics subset. All counts below
        are recomputed from local files. This notebook does not train a model.
        """),
        code(SETUP),
        md("""
        ## 1. Inspect every recording

        We read the case index for labels and inspect each metrics file for its
        actual dimensions and timestamps. A candidate prefix is a recorded name;
        it is not necessarily a separate architectural microservice.
        """),
        code("""
        from rca.analysis import audit_dataset
        overview, audit = audit_dataset()
        display(pd.DataFrame({
            'Quantity': ['Downloaded cases', 'Evaluable cases', 'Annotated root services',
                         'Distinct candidate prefixes', 'Metrics per case', 'Candidates per case',
                         'Rows per case', 'Missing cells', 'Cases with missing cells'],
            'Observed': [audit['n_cases'], audit['n_evaluable'], len(audit['root_services']),
                         len(audit['candidate_services']), str(audit['metric_count_range']),
                         str(audit['service_count_range']), str(audit['row_count_range']),
                         audit['missing_values'], audit['cases_with_missing_values']]
        }))
        display(overview.groupby('fault_type').agg(
            cases=('case_id', 'size'), evaluable=('evaluable', 'sum'),
            min_metrics=('n_metrics', 'min'), max_metrics=('n_metrics', 'max'),
            min_rows=('n_timesteps', 'min'), max_rows=('n_timesteps', 'max')
        ))
        print('Annotated roots:', ', '.join(audit['root_services']))
        print('Observed metric suffixes:', ', '.join(audit['metric_names']))
        """),
        md("""
        ## 2. Check data quality before scoring

        Two supplied recordings lack a usable before/after comparison. We preserve
        their timestamps and report them. We do not guess where injection occurred.
        Missing observations remain NaN: means and standard deviations use the
        available values separately within each window. No values are filled from
        one side of injection into the other.
        """),
        code("""
        display(overview.loc[~overview['evaluable'], [
            'case_id', 'inject_time', 'time_start', 'time_end', 'normal_rows', 'anomalous_rows'
        ]])
        display(overview.loc[overview['missing_values'].gt(0), ['case_id', 'missing_values']])
        print('Duplicate timestamps:', audit['duplicate_timestamps'])
        print('Infinite values:', audit['infinite_values'])
        """),
        md("""
        ## 3. Open one incident

        The loader returns the measurements and label metadata separately. The
        ranker receives only the measurements and known injection time. The
        two-level columns make it easy to select all measurements for one service.
        """),
        code("""
        cases = list_cases()
        case_by_id = {case.case_id: case for case in cases}
        example = cases[0]
        metrics, inject_time, root, fault, system = load_case(example)
        print('Case:', example.case_id, '| System:', system, '| Fault:', fault)
        print('Annotated root:', root, '| Injection Unix seconds:', inject_time)
        print('Measurement shape:', metrics.shape)
        display(metrics.xs(root, axis=1, level='service').head())
        """),
        md("""
        ## 4. Compare the root with four other candidates

        The dashed line is the supplied injection time. We plot CPU, memory, and
        an available latency measurement. Comparisons are selected by anomaly
        score. Values stay in their raw recorded units; labels do not establish
        whether memory is bytes or latency is seconds.
        """),
        code("""
        from rca.analysis import plot_case_metrics
        fig = plot_case_metrics(example)
        plt.show()
        """),
        md("""
        ## 5. Search all cases for a louder non-root

        For every evaluable case, score services first and only then compare with
        the annotation. Scores here **ignore metrics that were constant before
        injection** (section 6 shows why). Every case where a non-root service
        scores strictly higher than the injected root is listed.

        The plotted example, `re1ob_productcatalogservice_loss_3`, was **chosen by
        hand** from this list: in Online Boutique, `checkoutservice` calls
        `productcatalogservice`, so it is the clearest dependency to explain.
        A higher score alone does **not** prove the fault caused the caller's
        latency rise; that is the hypothesis the project will test.
        """),
        code("""
        from rca.analysis import find_louder_cases, plot_louder_case
        hits = find_louder_cases(skip_constant=True)
        print(f'{len(hits)} of {audit["n_evaluable"]} evaluable cases: a non-root outscores the injected root.')
        display(hits.sort_values('ratio', ascending=False).head(8))
        EXAMPLE = 're1ob_productcatalogservice_loss_3'
        assert EXAMPLE in set(hits['case_id']), 'The chosen example is no longer a non-root win.'
        fig = plot_louder_case(
            case_by_id[EXAMPLE], skip_constant=True,
            title='Packet loss in productcatalogservice: its caller looks 11x more anomalous',
            caption=('Purple = injected root; orange = top-ranked non-root. checkoutservice calls productcatalogservice in Online Boutique. '
                     'Constant metrics ignored. Propagation is a hypothesis, not proven by this plot.'),
            selection_rule='hand-picked from victim_search_zscore_nonconst.csv: clearest documented caller -> callee dependency')
        plt.show()
        """),
        md("""
        ## 6. Why constant metrics must be ignored

        The first baseline divides by the normal-window standard deviation plus
        $10^{-8}$. When a metric never moved before injection (for example an error
        counter stuck at 0), any later blip is divided by $10^{-8}$ and dwarfs a real
        CPU spike. Re-running the search **with** constant metrics shows how often
        this happens, and the first such case shows the artifact directly.
        """),
        code("""
        artifact_hits = find_louder_cases(skip_constant=False)
        print(f'With constant metrics kept: {len(artifact_hits)} non-root wins '
              f'(vs {len(hits)} when they are ignored).')
        fig = plot_louder_case(
            case_by_id[artifact_hits.sort_values('case_id').iloc[0]['case_id']],
            output=RESULTS / 'fig_zero_variance_artifact.png', skip_constant=False,
            title='Artifact: two error blips (0 to 0.002) outscore a CPU jump from 1.6 to 97',
            caption='frontend errors were exactly 0 before injection, so the z-score divides by 1e-8. This motivates the second baseline.',
            selection_rule='first case_id alphabetically among non-root wins with constant metrics kept')
        plt.show()
        """),
        md("""
        ## 7. What we learned

        A large anomaly is not automatically the root cause. Ignoring constant
        metrics removes a scaling artifact, but non-root services still outscore
        the injected root in many packet-loss cases, where callers show the
        symptoms. That remaining gap is what a dependency-aware learned model must
        close, and it must be compared against this stronger statistical baseline.

        Outputs in `results/`: `dataset_overview.csv`, `dataset_audit.json`,
        `victim_search_zscore_nonconst.csv`, `victim_search_zscore.csv`,
        `fig_case_metrics.png`, `fig_victim_louder_than_culprit.png`,
        `fig_zero_variance_artifact.png`.

        Source: [official RCAEval dataset](https://huggingface.co/datasets/phamquiluan/RCAEval).
        """)
    ])

    failure_text = (ROOT / 'docs/failure_analysis.md').read_text(encoding='utf-8')
    save('02_baseline.ipynb', [
        md("""
        # 02 · Evaluate two transparent baselines

        **Question:** How often does the largest standardized mean shift identify
        the service where a fault was injected?

        Both baselines are heuristics, not trained ML models. Their measured
        failures motivate the later learning problem. All tables are computed below.
        """),
        code(SETUP),
        md(r"""
        ## 1. The rule and evaluation setting

        For metric $m$, $z_m = |\mu_{post,m}-\mu_{pre,m}|/(s_{pre,m}+10^{-8})$.
        A service's score is the **maximum** of its metric scores. We use sample
        standard deviation ($ddof=1$), all available `time < inject_time` rows,
        and all available `time >= inject_time` rows. Ties use alphabetical order.

        - `zscore` keeps every metric.
        - `zscore_nonconst` ignores metrics whose normal-window standard deviation
          is exactly zero. For those, the denominator is just $10^{-8}$, so any
          blip after injection gets an enormous score (see notebook 01, section 6).

        Injection time is supplied. No root labels, fault labels, case names,
        or future cases are input features. Full post-injection recordings are
        available, so this experiment is **offline localization**, not online
        incident detection or a held-out learned-model test.
        """),
        code("""
        from rca.evaluate import run_evaluation
        runs = {method: run_evaluation(method=method) for method in ['zscore', 'zscore_nonconst']}
        per_case, summary = runs['zscore_nonconst']
        comparison = pd.concat({method: result[1] for method, result in runs.items()}, names=['method']).reset_index(level=0)
        display(comparison.loc[comparison['scope'].eq('overall'),
                               ['method', 'evaluation_set', 'n_cases', 'AC@1', 'AC@3', 'AC@5', 'Avg@5']]
                .style.format({name: '{:.2%}' for name in ['AC@1', 'AC@3', 'AC@5', 'Avg@5']}).hide(axis='index'))
        """),
        md("""
        ## 2. Read the metrics correctly

        **AC@k** is the fraction of cases with the annotated root in the first k
        positions. **Avg@5** is the mean of AC@1, AC@2, AC@3, AC@4, and AC@5;
        it is not average precision or mean reciprocal rank.

        `all_cases` includes all 125 supplied cases, counting invalid-window cases
        as misses. `evaluable_only` reports the 123 cases with valid windows.
        Both denominators are shown to prevent silent dataset exclusions.
        The chart compares both baselines per fault type on all 125 cases.
        """),
        code("""
        invalid = per_case.loc[per_case['status'].ne('ok'), ['case_id', 'fault_type', 'status', 'error']]
        display(invalid)
        from rca.analysis import plot_baseline_comparison
        fig = plot_baseline_comparison({method: result[1] for method, result in runs.items()})
        plt.show()
        """),
        md("""
        ## 3. The five worst ranked, evaluable incidents (stronger baseline)

        Sort by the annotated root's rank under `zscore_nonconst`, descending,
        then case ID. These are the failures that remain after the scaling fix.
        The two unscorable recordings are listed above rather than being
        described as model-generated rankings.
        """),
        code("""
        def worst_cases(frame):
            return frame.loc[frame['status'].eq('ok')].sort_values(
                ['rank_of_true_root', 'case_id'], ascending=[False, True]).head(5)
        for method, (frame, _) in runs.items():
            worst_cases(frame).to_csv(RESULTS / f'{method}_worst_cases.csv', index=False)
        worst = worst_cases(per_case)
        display(worst[['case_id', 'fault_type', 'true_root', 'rank_of_true_root', 'predicted_ranking']])
        """),
        md("""
        ## 4. Explain two failures using actual measurements

        The following Markdown notes were generated from this pinned dataset's
        measured `zscore_nonconst` scores. The verification cell below checks that they still
        agree with the recomputed worst cases. Interpretations are hypotheses.
        """),
        md(failure_text),
        code("""
        from rca.analysis import worst_case_explanation
        case_by_id = {case.case_id: case for case in list_cases()}
        regenerated = '# Two measured baseline failure cases\\n\\n' + '\\n\\n'.join(
            worst_case_explanation(case_by_id[row.case_id], skip_constant=True) for row in worst.head(2).itertuples())
        saved_explanation = (ROOT / 'docs/failure_analysis.md').read_text(encoding='utf-8')
        assert regenerated == saved_explanation, 'Failure notes changed; regenerate the notebooks.'
        print('Both Markdown explanations agree with the recomputed evidence.')
        """),
        md("""
        ## 5. What comes next

        Ignoring constant metrics was the first scaling fix; next try robust
        scaling (median/MAD) and other aggregations as stronger statistical
        comparisons. Then learn an anomaly representation and compare
        a service-independent ranker with a graph model using a verified dependency
        graph. Divide training and testing by whole incidents; later reserve an
        entire application for cross-system testing. Do not randomly split rows
        from one recording across training and testing.

        A learned GNN, cross-system test, Docker, Kubernetes, and new fault-injection
        runs are **not implemented this week**. BARO is optional and not reproduced
        in this environment; see the README for the package-compatibility reason.

        Outputs in `results/`, for each method `zscore` and `zscore_nonconst`:
        `<method>_per_case.csv`, `<method>_summary.csv`, `<method>_protocol.json`,
        `<method>_worst_cases.csv`; plus `fig_baseline_comparison.png`.
        """)
    ])
    print('Created notebooks/01_eda.ipynb and notebooks/02_baseline.ipynb')


if __name__ == '__main__':
    main()
