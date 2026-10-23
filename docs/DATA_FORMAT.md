# Observed data format: RCAEval RE1-OB

This document describes the files downloaded and inspected for this project,
not a guessed schema. Raw files remain unchanged.

## Source and download

- Official repository: https://github.com/phamquiluan/RCAEval
- Official dataset: https://huggingface.co/datasets/phamquiluan/RCAEval
- Dataset revision: `afeacb11bcc94dadfd1c8f483ee4377b2b8b614e`.
- Method: the official `huggingface_hub.snapshot_download`, with `token=False`.
- First download `cases.parquet`; filter `dataset == "RE1-OB"`; request only
  `metrics.parquet` and `inject_time.txt` under each selected case directory.
- `cases.parquet` contains metadata for 735 cases; **only 125 cases' telemetry**
  was downloaded. No other system, logs, or traces were downloaded.
- The 251 selected data/index files total 31,866,704 bytes (see the machine-readable
  manifest for authoritative bytes and SHA-256 hashes).
- The dataset card states MIT licensing. Cite the [WWW Companion 2025 paper](https://arxiv.org/abs/2412.17015).

## Actual layout

```text
data/raw/
  cases.parquet
  re1ob_adservice_cpu_1/
    inject_time.txt
    metrics.parquet
  ... 124 other RE1-OB case directories ...
  .cache/huggingface/          # downloader metadata, not another dataset
```

Case names have the observed form `re1ob_<service>_<fault>_<repetition>`.
Labels come from the explicit official index; filenames are not model features.
There is no separate `root_cause.txt` in this selected subset.

## Case index

Observed columns: `case`, `dataset`, `suite`, `system`, `system_name`,
`root_cause_service`, `fault`, `fault_description`, `repetition`, `inject_time`,
`n_metrics`, `n_timesteps`, `time_start`, `time_end`, `duration_minutes`,
`normal_timesteps`, `faulty_timesteps`, `has_logs`, `n_logs`, `has_traces`,
`n_traces`, `has_root_cause_file`.

For `re1ob_adservice_cpu_1`: dataset=`RE1-OB`, system=`ob`,
system_name=`Online Boutique`, root_cause_service=`adservice`, fault=`cpu`,
repetition=1, inject_time=1685202688. The case file's injection time agrees with
the index; the loader checks this for every case.

## Metrics and timestamps

Each metrics file is a wide Parquet table: integer Unix timestamp `time` plus
floating-point metric columns. The first inspected case has 4,201 rows and
50 columns including time (49 metric columns). Example metric names:

| Raw column | Service/candidate prefix | Metric name |
| --- | --- | --- |
| `adservice_cpu` | `adservice` | `cpu` |
| `frontend_mem` | `frontend` | `mem` |
| `PassthroughCluster_load` | `PassthroughCluster` | `load` |
| `frontend-external_error` | `frontend-external` | `error` |
| `productcatalogservice_latency-90` | `productcatalogservice` | `latency-90` |

The last example occurs in later cases, not the first CPU case. Split on the last
underscore. Observed metric suffixes are `cpu`, `mem`, `load`, `workload`,
`error`, `latency`, `latency-50`, `latency-90`. The subset has **no disk-I/O
measurement**, even though disk stress is a fault label. Do not invent a `disk`
metric or infer units from column names. Figures use raw recorded units.

The text file holds a single integer Unix timestamp, e.g. `1685202688`.
Normal rows are `time < inject_time`; anomalous rows are `time >= inject_time`.
We use the entire available recording on each side, with no fixed-length crop.
Recordings have a median timestamp step of one second.

## Measured overview

| Quantity | Observed value |
| --- | --- |
| Cases | 125 |
| Evaluable before/after windows | 123 |
| Distinct annotated root services | 5 |
| Metric columns per case, excluding time | 49-59 |
| Candidate prefixes per case | 13-19 |
| Distinct candidate prefixes across the subset | 20 |
| Rows per case | 63-4,201 |
| Missing metric cells | 5,243 in 10 cases |
| Infinite cells / duplicate timestamps | 0 / 0 |
| Root service present among candidates | All 125 cases |

| Fault label | Description in index | Cases |
| --- | --- | --- |
| cpu | CPU stress | 25 |
| delay | network delay | 25 |
| disk | disk I/O stress | 25 |
| loss | network packet loss | 25 |
| mem | memory stress | 25 |

Annotated roots are `adservice`, `cartservice`, `checkoutservice`,
`currencyservice`, and `productcatalogservice`.

Keep **every observed prefix** as a candidate rather than restricting candidates
to these five known roots. Prefixes include infrastructure-like names (`main`,
`redis`, `PassthroughCluster`) and unexpected names such as `carts`, `catalogue`,
`front-end`, and `orders` in source recordings. These names are preserved and
reported, not silently reinterpreted or removed. Twenty observed prefixes does
not mean the application has twenty verified architectural microservices.

## Two cases cannot be scored as supplied

| Case | Injection timestamp | Recording start/end | Pre/post rows |
| --- | ---: | --- | --- |
| `re1ob_currencyservice_loss_1` | 16933142 | 1693313863-1693314583 | 0 / 721 |
| `re1ob_productcatalogservice_cpu_3` | 1685373255 | 1685371737-1685371799 | 63 / 0 |

The official index and text files agree on these values. We do not guess corrected
times, extrapolate missing observations, or fabricate rankings. `load_case` still
returns these cases for auditing. The ranker rejects their insufficient windows.
The evaluator records `status=invalid_window`, an empty ranking, and infinity
for root rank. The `all_cases` summary counts them as misses in a denominator of
125; `evaluable_only` reports the separate denominator of 123. This is a reporting
policy, not a claim that the algorithm produced a prediction for those two cases.

## Loader output and missing-value policy

`load_case(case)` returns `(metrics_df, inject_time, root_cause_service,
fault_type, system)`; system is the human-readable `Online Boutique`.

The metrics frame has a sorted numeric `time` index and two-level
`(service, metric)` columns. Get one service with:

```python
metrics_df.xs("adservice", axis=1, level="service")
```

Label metadata is returned separately and is never attached to the model's input.
NaN stays missing; infinities would become NaN with an audit count. We do not
forward-fill, back-fill, interpolate across injection, or replace missing cells
with zero. Each window's mean and sample standard deviation use its own observed
values. A metric requires at least two normal and one anomalous observation.
Otherwise it has no usable score. A service without usable metrics stays at the
bottom of the ranking rather than disappearing from the candidate set.

Zero-variance normal metrics are **retained** by the `zscore` baseline, which
uses `std + 1e-8`. This can create very large scores and is a measured
limitation, not evidence that a candidate is the root cause. The second
baseline, `zscore_nonconst`, ignores those metrics; the loader itself keeps
them either way.

Rebuild the audit by executing notebook 01. Detailed case statistics are in
`results/dataset_overview.csv`; hashes are in `results/dataset_manifest.json`.
