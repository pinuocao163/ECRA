# Bundled Research Resources

This directory contains supporting research materials included in the ECRA code submission. [inventory.json](inventory.json) records the current path, size, category, and SHA-256 of each payload. [audit.json](audit.json) records structural checks and classification-report summaries. These resources are included in the same submission archive as the implementation.

## Layout

```text
resources/
├── README.md
├── inventory.json
├── audit.json
├── dataset_metadata/
│   ├── iemocap/
│   │   ├── development.tsv
│   │   └── test.tsv
│   └── meld/
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
├── figures/
│   ├── ecra_framework.pdf
│   ├── corruption_robustness.pdf
│   ├── missing_modality.pdf
│   ├── quality_gate_sensitivity.pdf
│   └── vad_prototypes.pdf
├── results/
│   ├── IEMOCAP.txt
│   └── MELD.txt
└── placeholders/
    ├── features/
    ├── original_readmes/
    └── splits/
```

## LLM cache preparation

Generate prompts, raw judgments, and calibrated caches using the commands in the [main README](../README.md). The pipeline uses training-only evidence retrieval, target-dialogue exclusion, OOF training confidence, and ensemble-mean validation/test confidence.

## Dataset metadata and templates

The TSV/CSV files provide transcript and label metadata. IEMOCAP's `development.tsv` covers development dialogues, and `test.tsv` covers test dialogues. MELD metadata is organized into `train.csv`, `val.csv`, and `test.csv`.

The `placeholders/` directory contains feature filename references, a split template, and original resource readmes. Prepare the binary feature tensors and dialogue partitions according to [the data guide](../docs/DATA.md).

## Classification reports

[IEMOCAP.txt](results/IEMOCAP.txt) and [MELD.txt](results/MELD.txt) each contain five classification reports. Accuracy is read from the report body, weighted F1 from `weighted avg`, and macro F1 from `macro avg`.

| Dataset | Accuracy, mean ± sample SD (%) | Weighted F1, mean ± sample SD (%) | Macro F1, mean ± sample SD (%) |
|---|---:|---:|---:|
| IEMOCAP | 74.712 ± 0.129 | 74.866 ± 0.119 | 74.344 ± 0.188 |
| MELD | 68.342 ± 0.152 | 67.508 ± 0.123 | 51.700 ± 0.211 |

At two decimal places, these summaries correspond to the ECRA row in manuscript Table 3. The seed 5 reports provide the reference-run metrics in Table 2.

## Figures

[ecra_framework.pdf](figures/ecra_framework.pdf) illustrates the ECRA pipeline. The remaining PDFs present corruption robustness, missing-modality evaluation, quality-gate sensitivity, and VAD prototypes.

## Checks and packaging

From the project root, verify the resource inventory and refresh its audit:

```bash
python scripts/audit_resources.py
```

Create one archive containing the implementation, manuscript, and all resources:

```bash
python scripts/package_submission.py
```

The archive is written to `dist/ecra_submission.zip`. Acquired model weights and feature bundles under `models/` and `data/`, runtime outputs, virtual environments, logs, and the packaging directory itself are excluded. Every resource listed in the current inventory is included. The packager verifies ZIP CRCs and prints a SHA-256 checksum. Dataset-derived contents remain subject to their source data terms; see [the source notices](../THIRD_PARTY_NOTICES.md).
