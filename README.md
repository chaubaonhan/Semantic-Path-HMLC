# Semantic Path-HMLC

<!-- ============================================================ -->
<!-- PAPER TEMPLATE -- fill in / update the placeholders below     -->
<!-- as the manuscript moves through review.                       -->
<!-- ============================================================ -->

> **Title:** Semantic Path-HMLC: Frozen Label Semantics and Taxonomy Graph Fusion for Single-Path Hierarchical Text Classification
>
> **Authors:** Bao Nhan Chau¹, Phuoc Tran¹ (corresponding author)
> ¹ Faculty of Information Technology, Ton Duc Thang University, Ho Chi Minh City, Vietnam
>
> **Venue / status:** `<!-- e.g. Submitted to Neurocomputing, under review -- update on acceptance -->`
> **DOI:** `<!-- fill in once assigned -->`
> **Preprint:** `<!-- arXiv / SSRN link, if any -->`

<details>
<summary><b>Abstract</b> (click to expand)</summary>

> Hierarchical text classification is challenging when a taxonomy is large,
> imbalanced, and described by short label names. We study 434,478 Vietnamese
> news articles annotated with one of 981 valid complete paths across five
> levels. Semantic Path-HMLC integrates frozen label-derived features,
> taxonomy-graph propagation, shared action scoring, semantic-logit fusion,
> and an auxiliary semantic objective. Under PhoBERT-base-v2, it achieves the
> highest exact-path accuracy (0.7464 ± 0.0020), supported-path macro-F1
> (0.4575 ± 0.0043), and hierarchical F1 (0.8605 ± 0.0010) among five core
> variants over three seeds. Class balancing substantially improves macro-F1
> at the expense of aggregate accuracy. Disabling the auxiliary loss reduces
> all three primary metrics, whereas jointly disabling label-feature fusion,
> semantic scoring, and auxiliary supervision improves them; thus, the
> ablations do not establish an independent benefit from every component.
> Across five encoders, the macro-F1 advantage over Flat persists on this
> corpus, while exact-accuracy and hierarchical-F1 comparisons are
> encoder-dependent. A constructed known-taxonomy class-holdout evaluation
> achieves 0.7691 restricted-decoding accuracy on 30 unseen-test paths, but
> full-space generalized recognition remains weak. These findings support the
> integrated configuration as a long-tail-oriented approach for the studied
> setting, while leaving cross-dataset generalization and open-world
> recognition unresolved.

</details>

**Keywords:** Hierarchical text classification · Single-path classification · Label semantics · Taxonomy graph neural network · Long-tail classification · Zero-example generalization

**Citation:**
```bibtex
@article{chau_semantic_path_hmlc,
  title   = {Semantic Path-HMLC: Frozen Label Semantics and Taxonomy Graph Fusion for Single-Path Hierarchical Text Classification},
  author  = {Chau, Bao Nhan and Tran, Phuoc},
  journal = {},   %% TODO: fill in on acceptance
  year    = {},   %% TODO
  volume  = {},   %% TODO
  pages   = {},   %% TODO
  doi     = {}    %% TODO
}
```

<!-- ============================================================ -->
<!-- END PAPER TEMPLATE                                             -->
<!-- ============================================================ -->

---

## What this repository is

This repository is a cleaned, general-purpose extraction of the code used to
produce the results reported in the paper above. It is reorganized from the
original research notebooks into an installable package plus a sequence of
CLI scripts, one per paper section, so the pipeline can be inspected, re-run,
and adapted without Google Colab or Google Drive.

**What changed relative to the research notebooks:** only execution
plumbing -- Colab/Drive mounting is gone, hardcoded personal filesystem paths
became configurable (CLI flags / environment variables), and the code is
split into modules. **What did not change:** every model architecture, loss
function, hyperparameter default, random seed, split procedure, evaluation
metric, and ablation switch. See [`src/semantic_path_hmlc`](src/semantic_path_hmlc)
docstrings for the paper section each module implements.

**What this repository does *not* include:** the underlying news corpus
(434,478 VnExpress articles), any trained model checkpoint, or any
prediction/output file produced by running the pipeline. See
[Data availability](#data-availability) below.

## Repository structure

```
src/semantic_path_hmlc/
  config.py              Config dataclass (paths, hyperparameters, seeds, variant lists)
  common.py               seeding, device/AMP setup, small text utilities
  data.py                 schema resolution + taxonomy-path parsing (Sec. "Problem Formulation")
  audit.py                single-path task audit + assertion (Sec. 3.1)
  splits.py                leakage-safe grouped/stratified train-val-test split (Sec. 5.3)
  taxonomy.py              Taxonomy graph + action factorization + class weights (Sec. 3.2-3.3)
  tokenization.py          token-cache build (offline) and load (train-time), datasets/samplers
  models.py                Flat, Hierarchical-Softmax/Path-HMLC, Label-Semantic, Semantic Path-HMLC (Sec. 4)
  metrics.py               exact/macro-F1/hierarchical-F1/calibration metrics (Sec. 5.5)
  training.py              model construction, loaders, the single-run training loop (Sec. 5, Alg. 1-2)
  tables.py                mean±SD tables, subgroup/seen-unseen tables, paired bootstrap (Sec. 6.1-6.4)
  zero_shot.py             controlled known-taxonomy zero/few-shot protocol (Sec. 5.6, 6.5)
  encoder_robustness.py    five-encoder robustness grid (Sec. 6.6)
  evidence.py              machine-readable evidence-package export
  qualitative.py           qualitative example / error-analysis tables (Sec. 7.1)
  packaging.py             bundle tables/predictions/audit into one zip

scripts/
  01_tokenize_cache.py                Phase 01: build token caches (run once, per encoder)
  02_run_core_and_ablations.py        Phase 02: train 5 core variants + 5 ablations x 3 seeds
  03_build_tables.py                  Phase 03: subgroup/seen-unseen/calibration/bootstrap tables
  04_zero_shot_few_shot.py            Phase 04: controlled zero-example + few-shot evaluation
  05_encoder_robustness.py            Phase 05: 5-encoder robustness grid
  06_export_evidence_and_qualitative.py  Phase 06: evidence package + qualitative tables
  07_package_results.py               Phase 07: zip everything (excluding checkpoints)

tests/
  smoke_test.py                  offline test of taxonomy/model forward+loss/metrics (no GPU, no network)
  smoke_test_data_pipeline.py    offline test of schema/audit/splits/tokenization plumbing
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tested with the dependency versions pinned in `requirements.txt`. The
original experiments used Python 3.13, PyTorch 2.11 (CUDA 12.8), and an
NVIDIA A100-80GB GPU; training also runs on a CPU or a smaller GPU, just
much slower, and the batch size auto-tunes down accordingly
(`semantic_path_hmlc.config.autotune_for_device`).

Run the offline smoke tests any time (no dataset, GPU, or network required)
to check your environment before committing to a full run:

```bash
python tests/smoke_test.py
python tests/smoke_test_data_pipeline.py
```

## Data availability

This repository does not redistribute the underlying article corpus. To run
the pipeline, provide your own parquet file with (at minimum) a text column
and a taxonomy-path column. By default (`Config.text_col="content"`,
`Config.paths_col="sublabel"`) it expects the same schema used for the
paper's dataset:

| column     | meaning                                                              |
|------------|-----------------------------------------------------------------------|
| `content`  | raw article text                                                       |
| `sublabel` | the article's full gold taxonomy path, root -> leaf, e.g. `["giao duc", "tuyen sinh"]` |
| `year`     | (optional) publication year, used only for the diagnostic temporal split |

Point `Config.data_path` (or the `SPHMLC_DATA_PATH` environment variable, or
`--data-path` on every script) at your parquet file. If your schema differs,
either rename your columns or set `Config.text_col` / `Config.paths_col` /
`Config.level_cols` / `Config.path_delimiter` accordingly -- see
`src/semantic_path_hmlc/data.py`.

## Reproducing the paper

All configuration lives in `semantic_path_hmlc.config.Config`; every script
below accepts `--data-path`, `--output-dir`, `--token-cache-dir`, and
`--run-mode {paper,smoke}` (or the equivalent `SPHMLC_*` environment
variables) instead of the notebooks' hardcoded Drive paths. `--run-mode
smoke` runs a fast, small-scale sanity pass (a `smoke_rows`-sized sample,
one seed, two variants) -- never use it to produce numbers for a paper or
report.

Run the phases in order from the repository root:

```bash
export SPHMLC_DATA_PATH=/path/to/your/dataset.parquet
export SPHMLC_OUTPUT_DIR=./outputs
export SPHMLC_TOKEN_CACHE_DIR=./token_cache

python scripts/01_tokenize_cache.py            # once per encoder; required before any training
python scripts/02_run_core_and_ablations.py    # 5 core variants + 5 ablations x 3 seeds -> outputs/tables/
python scripts/03_build_tables.py              # subgroup / seen-unseen / calibration / bootstrap tables
python scripts/04_zero_shot_few_shot.py        # controlled known-taxonomy zero- and few-shot protocol
python scripts/05_encoder_robustness.py        # retrain flat + semantic_path_hmlc on 5 encoders
python scripts/06_export_evidence_and_qualitative.py   # paper_evidence.json + qualitative tables
python scripts/07_package_results.py           # zip outputs/{audit,tables,predictions} for archiving
```

Each phase is independently re-runnable: `train_one` / `train_controlled_one`
skip a (variant, seed[, encoder]) run whose cached metrics/checkpoint/
prediction files already match the current pipeline-version tag, so
interrupting and resuming a long run is safe. Every training script shares
the same `Config` defaults (seeds, hyperparameters, variant lists), so
running them against the same dataset file reproduces the same taxonomy and
train/val/test split every time.

Outputs land under `Config.output_dir` (default `./outputs`):
`audit/` (taxonomy audit), `splits/` (split manifest + controlled zero-shot
manifest), `checkpoints/`, `predictions/` (per-run `.npz` score arrays),
`tables/` (every CSV/LaTeX/JSON table referenced above), and
`paper_evidence.json` (a single machine-readable summary of every audited
statistic and result, meant to make it easy to double-check a number quoted
in the paper against the artifact that produced it).

## Notes on fidelity to the original experiments

- Hyperparameters, seeds (13/21/42), the five core variants, the five
  ablation switches, the controlled zero-shot protocol parameters, and the
  five-encoder robustness grid are unchanged `Config` defaults -- see
  `src/semantic_path_hmlc/config.py`.
- The single-path task audit and assertion (`audit.run_taxonomy_audit`)
  still runs before any row is dropped or subsampled, so smoke-mode
  sampling can never contaminate reported dataset statistics.
- Post-hoc temperature scaling is calibration-only by construction
  (`metrics.fit_temperature` / `tables.build_calibration_ablation` assert
  that argmax predictions are unchanged before and after scaling).

## License

Code in this repository is released under the [MIT License](LICENSE). This
does not extend to the underlying news corpus, which is not included here.
