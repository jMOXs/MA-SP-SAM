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
- `SAMAdapter` V1 is implemented as an optional inference-only wrapper around Meta Segment Anything's `SamPredictor`.
- Local AimSeg archive indexing is available, but AimSeg is not used for V1 training.

micro-SAM integration, SAM mask decoder training, PairRefinementModule integration, DANN/CORAL, and formal TEM1-to-TEM2 experiments are not implemented yet.

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

The prediction pipeline writes per-sample `semantic_pred.png`, `center_heatmap.png`, boundary heatmaps, `proposal_labels.tif`, `prompt_summary.json`, and a global `summary.csv` with proposal recall/precision/F1 diagnostics.

## SAMAdapter V1

SAMAdapter is optional. Install Meta Segment Anything separately when you want to run SAM inference:

```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

Download the SAM checkpoint yourself, for example a ViT-B checkpoint, and keep it outside git. Do not commit SAM weights to this repository.

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

If `segment-anything` is not installed, the command exits with a clear dependency message instead of a traceback.
