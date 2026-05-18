# MA-SP-SAM

MA-SP-SAM is a staged research codebase for TEM axon-myelin paired instance label construction and future self-prompt SAM experiments.

## Current status

- ASTIH official split indexing and paired label construction are implemented.
- Semantic masks use labels `0=background`, `1=myelin`, `2=axon`.
- Aligned `fiber_instance`, `axon_instance`, `myelin_instance`, and `pair_table.csv` are generated.
- Dataset QC overlays, label-quality reports, and label-only baseline evaluation are implemented.
- `SelfPromptGenerator` V1 is implemented in `src/ma_sp_sam/models/self_prompt.py`.
- `ProposalGenerator` and `PromptSynthesizer` are implemented for prompt candidate generation.
- `SelfPromptLoss` and `scripts/train_self_prompt.py` are implemented for Self-Prompt training V1.
- `scripts/predict_self_prompt.py` is implemented for Self-Prompt inference, prompt summary, and proposal QC diagnostics.
- `SAMAdapter` V1.1 is implemented as an optional inference-only wrapper around Meta Segment Anything's `SamPredictor`.
- `PairRefinementModule` V1 is implemented for converting SAM candidate masks into paired fiber/axon/myelin instances.
- End-to-End V1 pipeline orchestration and GT-QC summary export are implemented.
- Local AimSeg archive indexing is available, but AimSeg is not used for V1 training.

micro-SAM integration, SAM mask decoder training, DANN/CORAL, and formal TEM1-to-TEM2 experiments are not implemented yet.

## Quick checks

```bash
pytest -q
python scripts/inspect_manifest.py --config configs/data/astih_tem.yaml --dry-run
python scripts/prepare_data.py --stage index --datasets TEM1 TEM2
python scripts/prepare_data.py --stage build-labels --datasets TEM1 --limit 5 --export-qc
python scripts/validate_labels.py --processed data/processed/astih_tem --report outputs/reports/astih_label_qc.csv
```

## Self-Prompt training

Run a CPU smoke training pass on a small subset:

```bash
python scripts/train_self_prompt.py --config configs/train/self_prompt.yaml --limit 2 --epochs 1 --device cpu
```

The default checkpoint path is:

```text
checkpoints/self_prompt/best.pt
```

## Self-Prompt inference and QC

After training, run prediction and proposal diagnostics:

```bash
python scripts/predict_self_prompt.py \
  --checkpoint checkpoints/self_prompt/best.pt \
  --config configs/train/self_prompt.yaml \
  --split test \
  --dataset TEM1 \
  --limit 5 \
  --device cpu \
  --out outputs/self_prompt_predictions
```

The prediction pipeline writes per-sample `semantic_pred.png`, raw label-map `semantic_pred_labels.tif`, `center_heatmap.png`, boundary heatmaps, `proposal_labels.tif`, `prompt_summary.json`, and a global `summary.csv` with proposal recall/precision/F1 diagnostics.

## SAMAdapter V1.1

SAMAdapter is optional. Install Meta Segment Anything separately when you want to run SAM inference:

```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

Download the SAM checkpoint yourself, for example a ViT-B checkpoint, and keep it outside git. Do not commit SAM weights to this repository.

By default SAMAdapter sends point and box prompts to SAM. The coarse mask prompt is not passed by default because SAM's `mask_input` is a low-resolution prior, not a full-resolution mask. The converter keeps the coarse mask shape and ring points in prompt metadata for later analysis/refinement. Use `--use-mask-input` only when you want the coarse mask resized to a `256x256` low-resolution SAM prior.

Run SAM from Self-Prompt packages:

```bash
python scripts/predict_sam_from_self_prompt.py \
  --self-prompt-checkpoint checkpoints/self_prompt/best.pt \
  --self-prompt-config configs/train/self_prompt.yaml \
  --sam-checkpoint /path/to/sam_vit_b.pth \
  --sam-model-type vit_b \
  --dataset TEM1 \
  --split test \
  --limit 5 \
  --device cpu \
  --out outputs/sam_predictions
```

Add `--use-mask-input` to enable the resized low-resolution mask prior, and `--save-all-candidates` to export every candidate mask array per instance. The script always writes `sam_candidates.npz`, `sam_scores.csv` with `best_index`, and `sam_prompt_summary.json`.

If `segment-anything` is not installed, the command exits with a clear dependency message instead of a traceback.

## Pair refinement

Convert saved SAM candidates into paired instance predictions:

```bash
python scripts/refine_sam_predictions.py \
  --sam-pred-root outputs/sam_predictions \
  --self-prompt-root outputs/self_prompt_predictions \
  --processed-root data/processed/astih_tem \
  --dataset TEM1 \
  --split test \
  --limit 5 \
  --out outputs/refined_predictions
```

The refinement step writes `refined_fiber_instance.tif`, `refined_axon_instance.tif`, `refined_myelin_instance.tif`, `refined_pair_table.csv`, and a global `summary.csv`. If GT processed labels exist under `--processed-root`, refinement also reports `fiber_iou50_recall`, `fiber_iou50_precision`, `axon_dice`, `myelin_dice`, `pair_accuracy_proxy`, and `g_ratio_mae`. These GT labels are used only for evaluation/QC, never for refinement decisions.

## End-to-End V1 pipeline

Full mode runs Self-Prompt prediction, SAM prediction, pair refinement, and final summary merge:

```bash
python scripts/run_v1_pipeline.py \
  --self-prompt-checkpoint checkpoints/self_prompt/best.pt \
  --self-prompt-config configs/train/self_prompt.yaml \
  --sam-checkpoint /path/to/sam_vit_b.pth \
  --sam-model-type vit_b \
  --dataset TEM1 \
  --split test \
  --limit 5 \
  --device cuda \
  --work-dir outputs/v1_pipeline
```

Skip-SAM mode is for reusing existing SAM candidates already saved under `work-dir/sam`:

```bash
python scripts/run_v1_pipeline.py \
  --self-prompt-checkpoint checkpoints/self_prompt/best.pt \
  --self-prompt-config configs/train/self_prompt.yaml \
  --dataset TEM1 \
  --split test \
  --limit 5 \
  --device cpu \
  --work-dir outputs/v1_pipeline \
  --skip-sam
```

The merged report is written to `outputs/v1_pipeline/summary.csv` with key fields from self-prompt proposals, SAM candidates, refinement counts, and optional GT-QC metrics.

## Running ASTIH V1 experiments

ASTIH V1 experiment runner is a thin orchestration layer for repeated TEM1/TEM2 V1 runs. It records each experiment configuration, writes per-run status files, and builds cross-experiment summaries for later formal experiments and ablations.

Before running full experiments:

1. Prepare ASTIH processed labels with `scripts/prepare_data.py`.
2. Train or provide a Self-Prompt checkpoint, usually `checkpoints/self_prompt/best.pt`.
3. Install optional `segment-anything` and prepare a local SAM checkpoint for full mode.
4. Review `configs/experiments/astih_v1.yaml` and update paths/device/limit as needed.

`configs/experiments/astih_v1.yaml` is the formal full-mode configuration for TEM1 internal and TEM2 external experiments. `configs/experiments/astih_v1_smoke.yaml` is for debug/smoke runs with `mode: skip_sam`; do not treat smoke output as a formal result.

Preview the experiment plan without executing:

```bash
python scripts/run_astih_v1_experiments.py \
  --config configs/experiments/astih_v1.yaml \
  --dry-run
```

Run selected experiments:

```bash
python scripts/run_astih_v1_experiments.py \
  --config configs/experiments/astih_v1.yaml \
  --only tem1_internal,tem2_external
```

## Preflight before ASTIH experiments

Run preflight before a long experiment batch:

```bash
python scripts/run_astih_v1_experiments.py \
  --config configs/experiments/astih_v1.yaml \
  --preflight-only
```

Preflight checks the Self-Prompt checkpoint/config, processed label root, dataset/split processed directory, full-mode SAM checkpoint, `segment-anything` importability, and skip-sam candidate directories. Strict preflight is enabled by default: failed checks mark that experiment as `preflight_failed` and prevent the pipeline for that experiment from running, while the runner continues to the next experiment. Use `--no-strict-preflight` only when you want to record failed checks but still allow the pipeline attempt.

Each experiment writes to `outputs/experiments/{experiment_name}` with `resolved_config.yaml`, `preflight.json`, `run_status.json`, and the V1 pipeline outputs. The runner also writes:

```text
outputs/experiments/experiment_status.csv
outputs/experiments/summary_all.csv
outputs/experiments/metrics_by_experiment.csv
```

`experiment_status.csv` is the first file to check: it records whether each experiment succeeded, failed, or stopped at `preflight_failed`, along with the error message and timestamps. `summary_all.csv` keeps per-sample proposal, SAM, refinement, and GT-QC fields. `metrics_by_experiment.csv` aggregates numeric metrics such as Dice, fiber IoU50 recall/precision, pair accuracy proxy, g-ratio MAE, and proposal recall/precision.

`mode: skip_sam` is intended only for smoke tests and debugging with existing SAM candidate files. It is not a formal experiment result.
