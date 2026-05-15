# MA-SP-SAM

MA-SP-SAM is a staged research codebase for TEM axon-myelin paired instance label construction and future self-prompt SAM experiments.

This first implementation focuses on data plumbing only:

- Index ASTIH official splits from `ASTIH-main/ASTIH-main/data/splits/{TEM1,TEM2}/{train,test}`.
- Build semantic masks with labels `0=background`, `1=myelin`, `2=axon`.
- Derive aligned `fiber_instance`, `axon_instance`, `myelin_instance`, and `pair_table.csv`.
- Export QC overlays and label-quality reports.
- Index local AimSeg archives in `8351731/` without using them for training yet.

No complex SAM training code is implemented in this V1.

## Quick checks

```bash
pytest -q
python scripts/inspect_manifest.py --config configs/data/astih_tem.yaml --dry-run
python scripts/prepare_data.py --stage index --datasets TEM1 TEM2
python scripts/prepare_data.py --stage build-labels --datasets TEM1 --limit 5 --export-qc
python scripts/validate_labels.py --processed data/processed/astih_tem --report outputs/reports/astih_label_qc.csv
```
