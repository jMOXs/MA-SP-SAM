# Run ASTIH V1 Experiments

This runbook describes the current MA-SP-SAM V1 experiment flow for ASTIH TEM1/TEM2. It does not train SAM, does not use micro-SAM, and does not include DANN/CORAL.

## 1. Prepare ASTIH Processed Labels

Build the official-split ASTIH manifest and paired labels first:

```bash
cd /sydata/js/MA-SP-SAM
python scripts/prepare_data.py --stage index --datasets TEM1 TEM2
python scripts/prepare_data.py --stage build-labels --datasets TEM1 TEM2 --export-qc
```

Expected processed labels live under:

```text
data/processed/astih_tem/TEM1/{train,test}/...
data/processed/astih_tem/TEM2/{train,test}/...
```

The experiment runner uses GT labels only for QC/evaluation, not for refinement decisions.

## 2. Train Self-Prompt Checkpoint

Run a small smoke pass before longer training:

```bash
python scripts/train_self_prompt.py \
  --config configs/train/self_prompt.yaml \
  --limit 2 \
  --epochs 1 \
  --device cpu
```

For real runs, train with your chosen GPU/device and keep the resulting checkpoint path aligned with the experiment config:

```text
checkpoints/self_prompt/best.pt
```

Do not commit checkpoint files.

## 3. Prepare SAM Checkpoint

Install Meta Segment Anything when running full mode:

```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

Download the SAM ViT-B checkpoint yourself and update `sam_checkpoint` in the config. Keep SAM weights outside git.

## 4. Run Preflight Only

Preflight catches missing data, checkpoints, dependencies, and skip-sam inputs before a long run:

```bash
python scripts/run_astih_v1_experiments.py \
  --config configs/experiments/astih_v1.yaml \
  --preflight-only
```

Inspect:

```text
outputs/experiments/{experiment_name}/preflight.json
outputs/experiments/{experiment_name}/run_status.json
outputs/experiments/experiment_status.csv
```

Fix every `preflight_failed` row before starting a full experiment.

## 5. Limit=2 Full-Mode Run

Use the limit-2 template for the first real SAM-backed run:

```bash
python scripts/run_astih_v1_experiments.py \
  --config configs/experiments/astih_v1_limit2.yaml
```

Before running, edit checkpoint paths and device fields to match the server.

## 6. Limit=10 Medium Run

After limit=2 passes output checks, run:

```bash
python scripts/run_astih_v1_experiments.py \
  --config configs/experiments/astih_v1_limit10.yaml
```

Use this stage to inspect runtime, memory, proposal counts, and refined QC metrics before full TEM1/TEM2.

## 7. Full TEM1/TEM2 Run

Use the formal full-mode config:

```bash
python scripts/run_astih_v1_experiments.py \
  --config configs/experiments/astih_v1.yaml
```

`astih_v1.yaml` is for formal full-mode TEM1 internal and TEM2 external experiments. `astih_v1_smoke.yaml` is only for debug/smoke runs with existing SAM predictions.

## 8. Check Experiment Outputs

After each run:

```bash
python scripts/check_experiment_outputs.py \
  --experiments-root outputs/experiments
```

The checker writes:

```text
outputs/experiments/output_check.json
```

`FAIL` means required files are missing. `WARN` means files exist but metrics may be sparse or incomplete. Warnings do not exit nonzero.

## 9. Read Summary Files

Start with:

```text
outputs/experiments/experiment_status.csv
```

Use it to see which experiments are `success`, `failed`, or `preflight_failed`.

Then inspect:

```text
outputs/experiments/summary_all.csv
outputs/experiments/metrics_by_experiment.csv
```

`summary_all.csv` is per-sample and includes proposal counts, SAM counts, refined instance counts, and GT-QC fields such as `axon_dice`, `myelin_dice`, `fiber_iou50_recall`, `pair_accuracy_proxy`, and `g_ratio_mae`.

`metrics_by_experiment.csv` aggregates numeric fields per experiment with mean, median, std, and count.

## 10. Common Failures

- `self_prompt_checkpoint` missing: train or copy the Self-Prompt checkpoint and update the config path.
- `self_prompt_config` missing: confirm `configs/train/self_prompt.yaml` exists or update the experiment config.
- `processed_root` missing: run `scripts/prepare_data.py` and confirm `data/processed/astih_tem` exists.
- processed dataset/split missing: confirm official ASTIH splits were prepared for the requested dataset and split.
- `sam_checkpoint` missing: download SAM weights locally and update `sam_checkpoint`; do not commit weights.
- `segment_anything` missing: install the optional dependency for full mode.
- skip-sam predictions missing: use `astih_v1_smoke.yaml` only after existing SAM candidates are present under `work_dir/sam/{dataset}/{split}`.
- output checker `FAIL`: open `output_check.json`, fix missing required files, then rerun the checker.
- many empty or NaN metrics: inspect `summary_all.csv`, per-experiment `summary.csv`, and per-sample refined outputs to identify failed proposal/SAM/refinement stages.
