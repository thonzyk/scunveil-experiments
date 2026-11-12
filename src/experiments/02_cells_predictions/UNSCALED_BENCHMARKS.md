# Corrected four-experiment runner: results and caches are independent

Use this command for the new comparison:

```bash
bash /workspace/subprojects/03_scunveil-experiments/src/experiments/06_cells_predictions/run_all_experiments.sh --results fresh --cache reuse --max-replicates 100
```

It runs ReSisTrace, tumor-reactive T cells, SCOPE-seq2 and mitochondrial potential
sequentially, and builds each report after that dataset reaches the target. It
recalibrates regularization and trains all methods again. The older
`run_all_pca_comparisons.sh`, `add_pca.py` and `collect_replicates.py` commands are
legacy scaled-probe protocols; do not use them for the corrected comparison.

The defaults are 8 CPU threads and GPU scUNVEIL inference. Override with
`--threads N --device cpu|gpu|auto`; use the same settings on resume. Sklearn heads
and PCA run on CPU. Cached features can avoid inference, but uncached shallow
scUNVEIL measurements require it. No full biological run was executed as part of
implementation; the software checks use small synthetic counts and fake frozen
2048-dimensional embeddings, with real sklearn training.

## The two independent switches

| Argument | Meaning |
| --- | --- |
| `--results fresh` | Create a new suite and new collection in every dataset. Recalibrate C/alpha and refit all predictors, even if identical feature caches exist. Preserve old collections and reports. |
| `--results resume` | Resume the latest corrected suite. Keep its completed repetitions and fixed calibration, and collect up to the requested TOTAL target. Reject changed scientific code, settings, environment or frozen-feature identity. |
| `--cache reuse` | Read only checksummed, matching computed-feature caches; compute and cache misses. Never restore old C/alpha, models, predictions, scores or reports into a fresh collection. |
| `--cache rebuild` | Delete the managed feature cache per dataset before running it. Disable legacy cache imports for this suite, including later resumes. Prepared input data and historical result collections remain intact. |

Resume an interrupted run:

```bash
bash /workspace/subprojects/03_scunveil-experiments/src/experiments/06_cells_predictions/run_all_experiments.sh --results resume --cache reuse --max-replicates 100
```

Recompute all features, PCA fits, calibration and predictors:

```bash
bash /workspace/subprojects/03_scunveil-experiments/src/experiments/06_cells_predictions/run_all_experiments.sh --results fresh --cache rebuild --max-replicates 100
```

`--max-replicates 100` means 100 completed repetitions in total, not 100 additional
ones. A larger target with resume adds repetitions without changing calibration.
A smaller target reports the first N without deleting existing repetitions.
Fresh starts again at repetition zero with the same seeded protocol, allowing
paired comparisons across collections; it does not invent different random seeds.

Fresh creates a uniquely named suite JSON under
`/workspace/results/analysis/benchmark_suites/`; its printed path pins the four
source collections and four new destination collections. `LATEST_UNSCALED.json`
selects the default resume suite. Use `--suite /absolute/path/to/suite.json` with
resume to select an older suite. Starting a later fresh suite never deletes an
earlier one. Concurrent suite runners and concurrent dataset writers are rejected
with OS locks. Ctrl+C/SIGTERM stops the sequence; completed work remains immutable.

Rebuild with resume preserves completed results/calibration and regenerates only
the features needed for remaining evaluations. For a complete recomputation use
fresh plus rebuild. A resumed run rejects a changed PCA basis or encoder identity
rather than mixing different features in one collection. Prepared source matrices
are verified input data, not disposable computed-feature caches; rebuild does not
redownload or reprocess raw reads. Legacy caches inside historical collections are
preserved for reproducibility but are not imported by a rebuild suite.

## Explicit scientific protocol

- Raw RNA: sparse float32 `log1p(integer counts)`, all prepared genes.
- scUNVEIL: all 2048 frozen features returned by the library's native API.
- PCA: gene-centered log1p counts, fit once on every eligible full-depth prepared
  cell, including calibration/train/dev/test without labels. Keep the original
  component variances: no whitening. Use 2048 PCs, capped at `min(n_cells-1,n_genes)`;
  SCOPE-seq2 therefore uses 955. Reuse the fixed mean/loadings to project each
  CURRENT diluted matrix, never substitute full-depth coordinates for shallow RNA.
- Shortcut controls: retain their existing explicit feature definitions, including
  log depth/detected genes, clonotype size, and mitochondrial fraction as applicable.
- **No StandardScaler, downstream normalization, variance rescaling, additional
  PCA or extra centering in ANY predictor pipeline.** Each pipeline contains only
  its estimator. Estimators still fit intercepts; PCA retains its defining gene mean.
  ReSisTrace's L-BFGS training inputs are promoted to float64 to avoid float32
  line-search failures at small C. This exactly preserves cached feature values;
  it does not rescale them. Feature caches remain float32.
- Retain the original class/sample weights: ReSisTrace balances lineage classes
  and weights cells within lineages; tumor-reactivity uses balanced classes;
  mitochondrial potential balances subject/visit/gate groups; SCOPE uses ordinary
  ridge. No class-prior probability correction precedes ReSisTrace lineage averaging.
- Keep the original calibration reservation and first 100 seeded group splits,
  selected cells and independent molecule draws. The 100%, 10%, 1% levels dilute
  ALL train/dev/test measurements. Methods share cells, roles and counts. Splits and
  selected cells are shared across retention levels; molecular draws differ.
- Fit every final predictor freshly from its training data in every repetition.
  Dev/test outcomes never select regularization. Calibration stays fixed on resume;
  repetition SD is conditional on that one-time calibration and frozen encoder.

## Calibration uses the reported objective

| Dataset | Objective and folds | Fixed parameter after calibration |
| --- | --- | --- |
| ReSisTrace | Mean calibration-fold AUROC of pooled lineage-averaged DIRECT probabilities; the original stratified lineage folds | C per condition, retention and representation |
| Tumor-reactive T cells | Mean held-out-calibration-patient AUROC; one seeded cell per clonotype | C per retention and representation |
| SCOPE-seq2 | Mean held-out-calibration-region R², independently for area and circularity | Independent alpha for each target, retention and representation |
| Mitochondrial potential | Mean held-out-visit AUROC in the one reserved calibration subject | C per retention and representation |

The mitochondrial calibration has visit-held-out validation within one subject;
evaluation is subject-held-out. This limitation is retained, not silently replaced.
ReSisTrace final scores are pooled lineage AUROC per condition. Tumor-reactivity
averages scorable held-out patients; mitochondrial evaluation averages visits within
subjects then subjects; SCOPE averages held-out technical regions.

Initial logarithmic grids: C = 1e-7 through 1e3; ridge alpha = 1e-3 through 1e7,
in decade increments. Every representation is independently calibrated. Strictly
improving boundary optima expand one decade at a time, up to eight expansions.
An unresolved boundary then stops evaluation. Gains <=1e-5 objective units at the
edge are recorded as plateaus, avoiding an endless search towards an intercept-only
limit. Exact ties favor stronger regularization (smaller C, larger alpha).
`calibration/tuning.tsv`, `grid_audit.tsv`, `selected.json`, `cells.tsv` and
`folds.tsv.gz` record the actual search, scores, choices and cells. Unconverged fits
raise errors rather than entering results.

## Cache identity, validation and scope

Managed cache: `/workspace/results/analysis/{dataset}/feature_cache/unscaled_v2/`.
Only this subtree is deleted by rebuild; symlink-based deletion is refused.

PCA basis keys cover the verified full prepared-data hash, cell/gene identities and
order, PCA settings/seed, solver code, NumPy/SciPy versions and CPU thread count.
Transformed PCA scores additionally bind the ACTUAL basis-array content hash.
scUNVEIL keys cover counts, cell/gene identities and order, checkpoint files,
inference/extractor code, numerical-library versions, batch size, device/threads,
GPU identity/driver where available and relevant TensorFlow environment settings.
Shallow feature keys also cover the ordered selected cells, retention, dilution seed
and actual realized sparse-count contents. Raw log1p matrices and cheap controls
are reconstructed directly from the paired counts.

All matrix objects have identities and checksum receipts, and are published by
atomic rename. Full-depth embeddings use a transactional per-cell SQLite cache
with an explicit computation-identity header and per-vector checksums. Incomplete
scratch objects are discarded on resume; corrupt completed objects stop the run.
Cache hits are exact stored arrays from a matching computation, not approximately
similar embeddings. Rebuilding on GPU does not promise bitwise determinism across
arbitrary hardware/software changes; changed runtime fingerprints cannot be mixed
into an existing collection.

Reuse can import old PCA bases only after checking source data/row/gene hashes,
solver code, settings, versions, thread count and artifact checksums. Legacy
full-depth scUNVEIL caches additionally require matching source provenance,
checkpoint file hashes, extraction settings and verified arrays/vectors. Missing
provenance causes recomputation, not a guessed match. Legacy shallow embeddings
were generally not saved and must be generated once for the new shared cache.
No legacy fitted predictors, regularization choices or scores are imported.

Historical checkpoint manifests may list multiple models sharing one Hugging Face
snapshot. Validation and inference comparisons check only `models/{model_version}/`:
unrelated absent model directories do not affect this experiment. Missing or changed
files of the SELECTED model remain fatal. The exact checkpoint-validation-only
maintenance patch is allowlisted in `checkpoint_validation_compatibility.json` so
the existing completed ReSisTrace collection and caches remain resumable. The
allowlist is bound to exact source hashes and validator code; unknown edits still
invalidate identities. `computation_code_audit.json` records actual versus canonical
computation hashes without editing historical repetition receipts.

Full-depth PCA scores are calculated once for all cells and indexed thereafter.
Shallow PCA/scUNVEIL matrices are stored per paired draw and shared across fresh
collections with identical feature computations. This trades disk for repeated-run
speed: ReSisTrace alone can use roughly 86 GB for 100 paired shallow PCA/scUNVEIL
draws, plus smaller caches for the other datasets. Available workspace storage was
checked during implementation; users control cache deletion explicitly.

## Reports and safe inspection during collection

New collections are `/workspace/results/analysis/{dataset}/collections/unscaled-{suite_id}/`.
Reports use only completed repetitions from that exact collection and its one
calibration. They verify protocol/calibration identities, score coverage and
checksums. Old metric rows are never merged. Statistics are mean ± sample SD
(ddof=1), plus paired differences; overlapping repetitions are not independent
biological cohorts or confidence intervals. Negative R² remains visible.

Stable outputs are `/workspace/results/analysis/{dataset}/report/REPORT.md` and
`report/paper_figure/` (PDF/SVG/PNG). The stable symlink changes atomically only
after a complete new report snapshot is ready. Historical snapshots remain intact.
Old stable reports remain visible until a new report is built, so check its protocol
and collection header when inspecting a running experiment. The paper project is
not updated automatically.

The existing per-dataset `build_report.py --max-repetitions 100` recognizes the new
format and is safe while collection continues. Pin `--collection PATH` to avoid
ambiguity between concurrent historical collections. Alternatively:

```bash
python /workspace/subprojects/03_scunveil-experiments/src/experiments/06_cells_predictions/benchmark_runner.py --report-only --collection /absolute/path/to/unscaled-COLLECTION --max-replicates 100
```

Progress bars use `smoothing=0`, with separate one-time PCA/inference/calibration and
remaining-repetition phases. Grid expansion adjusts the calibration-fit total.
Repetition ETA uses the average elapsed evaluation time in the current invocation;
there is no reliable pre-run wall-time estimate for four cold GPU/cache pipelines.
