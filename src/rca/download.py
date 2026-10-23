"""Download only the indexed RE1-OB cases, anonymously, from the official host."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from huggingface_hub import snapshot_download


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_REVISION = "afeacb11bcc94dadfd1c8f483ee4377b2b8b614e"
DATASET_REPO = "phamquiluan/RCAEval"


def download(raw_dir: Path, manifest_path: Path) -> dict:
    """Use the snapshot_download method documented in the RCAEval README.

    token=False is deliberate: it avoids using or changing a cached login.
    The tiny whole-benchmark index is needed to select the requested telemetry.
    """
    options = dict(repo_id=DATASET_REPO, repo_type="dataset",
                   revision=DATASET_REVISION, local_dir=str(raw_dir),
                   token=False, max_workers=4)
    snapshot_download(**options, allow_patterns=["cases.parquet"])
    index = pd.read_parquet(raw_dir / "cases.parquet")
    selected = index.loc[index["dataset"].eq("RE1-OB")].sort_values("case")
    if selected.empty or selected["case"].duplicated().any():
        raise ValueError("The official index has no unambiguous RE1-OB subset.")
    files = ["cases.parquet"]
    for case_id in selected["case"]:
        # Only the two files actually observed in a metrics-only case are needed.
        files += [f"{case_id}/metrics.parquet", f"{case_id}/inject_time.txt"]
    snapshot_download(**options, allow_patterns=files[1:])
    records = []
    for name in files:
        path = raw_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Incomplete download: {path}")
        records.append({"path": name, "bytes": path.stat().st_size,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest = {"repo_id": DATASET_REPO, "repo_type": "dataset",
                "revision": DATASET_REVISION, "subset": "RE1-OB",
                "download_method": "huggingface_hub.snapshot_download(token=False)",
                "n_cases": len(selected), "files": records,
                "total_bytes": sum(row["bytes"] for row in records)}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Downloaded/verified {len(selected)} RE1-OB cases; "
          f"{manifest['total_bytes'] / 1_000_000:.2f} MB including the index.")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data/raw")
    args = parser.parse_args()
    download(args.data_dir, PROJECT_ROOT / "results/dataset_manifest.json")


if __name__ == "__main__":
    main()
