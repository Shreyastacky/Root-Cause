# Related work for the progress review

These are reading references, not methods reproduced in this project. Reported
numbers from different papers are not directly comparable to our experiment.

| Work | Main idea | Why it matters here | Primary source |
| --- | --- | --- | --- |
| MicroRCA (NOMS 2020) | Uses an attributed graph and correlations between performance symptoms and resource measurements to localize faults. | Establishes a graph-based approach to anomaly propagation. | [Paper](https://doi.org/10.1109/NOMS47738.2020.9110353) |
| RCD (NeurIPS 2022) | Treats failures as interventions and uses localized, hierarchical causal discovery. | Shows why learning an entire causal graph may be unnecessary. | [Author-hosted paper](https://azamikram.github.io/files/rcd.pdf) |
| CIRCA (KDD 2022) | Identifies changes in conditional distributions under a causal model informed by system architecture. | Separates a root cause indicator from changes explained by its parents. | [Paper](https://arxiv.org/abs/2206.05871) |
| BARO (FSE 2024) | Robust RCA using multivariate Bayesian online change-point detection. | Relevant additional baseline; the optional package is separate from our z-score implementation. | [Paper](https://doi.org/10.1145/3660805) |
| Eadro (ICSE 2023) | Jointly learns anomaly detection and root cause localization using metrics, logs, traces, and service dependencies. | An important predecessor to a learned encoder and graph model; our proposed architecture is not automatically novel. | [Paper](https://arxiv.org/abs/2302.05092) |
| RCAEval (WWW Companion 2025) | Supplies annotated microservice failure datasets and a reproducible RCA evaluation framework. | Source of the data; our current task ranks services rather than individual metrics. | [Paper](https://arxiv.org/abs/2412.17015), [official repository](https://github.com/phamquiluan/RCAEval) |

The BARO entry is based on the title and official RCAEval reference; its full
publisher page was not accessible during setup. We do not claim a full-paper
reproduction of any of these methods.

## Proposed study, not a completed result

Start with a learned anomaly encoder, add message passing over a separately
verified service dependency graph, and predict one score per service. Compare
against the same encoder without graph edges. A graph is an architectural
dependency model, not automatically a causal graph.

Training can later use controlled fault-injection runs: fault scenarios are
artificially induced, but the resulting measurements are collected from running
applications. Purely simulated time series, if explored, must be identified
separately and validated on real application telemetry.

No new infrastructure or fault injection is part of this week's work. RE1-OB
alone cannot establish cross-system generalization. A later test must reserve an
entire other system, freeze model choices before testing, and avoid using service
names or case filenames as predictive features.
