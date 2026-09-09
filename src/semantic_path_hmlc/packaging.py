"""Package results for download/publication: bundles per-model/per-seed
metrics, mean +/- SD and LaTeX summary tables, subgroup and paired-bootstrap
results, per-model predictions, the taxonomy audit, and
``paper_evidence.json`` into one zip. Raw model checkpoints are intentionally
left out. Unchanged from the original notebook.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path


LEGACY_RUN_PATTERN = re.compile(
    r"(?:^|__)(?:path_hmlc_wo_balance|path_hmlc_wo_graph|path_hmlc_wo_calibration)(?:__|\.|$)"
)
LEGACY_ENCODER_PATH_PATTERN = re.compile(
    r"^(?:metrics__|history__|subgroups__|qualitative_examples__)?path_hmlc__(?!seed)"
)


def package_results(out_dir: Path, bundle_root: Path) -> Path:
    out_dir = Path(out_dir)
    bundle_root = Path(bundle_root)
    bundle_dirs = ["audit", "tables", "predictions"]
    bundle_files = ["paper_evidence.json"]

    if bundle_root.exists():
        shutil.rmtree(bundle_root)
    bundle_root.mkdir(parents=True)

    for d in bundle_dirs:
        src = out_dir / d
        if not src.exists():
            continue
        for source_file in src.rglob("*"):
            if not source_file.is_file():
                continue
            if (
                "__smoke" in source_file.name
                or LEGACY_RUN_PATTERN.search(source_file.name)
                or LEGACY_ENCODER_PATH_PATTERN.search(source_file.name)
            ):
                continue
            destination = bundle_root / d / source_file.relative_to(src)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_file, destination)
    for fname in bundle_files:
        src = out_dir / fname
        if src.exists():
            shutil.copy2(src, bundle_root / fname)

    zip_path = Path(shutil.make_archive(str(out_dir / "results_bundle"), "zip", root_dir=bundle_root))
    size_mb = zip_path.stat().st_size / 2**20
    n_files = sum(1 for p in bundle_root.rglob("*") if p.is_file())
    print(f"Results bundle: {zip_path} ({size_mb:.1f} MB, {n_files} files)")
    return zip_path
