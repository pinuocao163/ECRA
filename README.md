# ECRA: Evidence-Conditioned Reuse of LLM-Generated Annotations

Official code for **ECRA: Evidence-Conditioned Reuse of LLM-Generated Annotations for Multimodal Conversational Emotion Recognition**. The submission manuscript is included as [main.pdf](main.pdf).

ECRA integrates retrieval-grounded LLM judgments into a multimodal conversational emotion classifier. Its central design separates admission of a cached judgment from the magnitude of its representation update. This archive provides the observation encoder, evidence retrieval, structured cache construction and calibration, training objectives, validation-based checkpoint selection, and test evaluation for IEMOCAP and MELD.

## Method

1. Nine causal Transformer branches encode within-modality and cross-modality interactions among text, audio, and visual features. Linear target-state reductions and a scalar three-source gate form the observation-only backbone representation.
2. Dual-cue retrieval combines TF–IDF context similarity with standardized audio-visual statistics. The evidence bank contains training dialogues only and excludes the target dialogue.
3. Qwen2.5-7B-Instruct generates structured class judgments, affective appraisals, and modality hints. Dialogue-level five-fold calibration assigns OOF confidence to training records and the mean of five calibrated probabilities to validation and test records.
4. Evidence support and calibrated confidence determine admission. Four-source allocation proposes a representation, and trust-scaled residual interpolation controls its update. Rejection exactly preserves the backbone representation and shared-classifier prediction in the same forward pass.
5. Training combines classification, unimodal supervision, self-distillation, class-prior alignment, source-weight supervision, and affective geometry objectives.

## Requirements

Use Python 3.10. Direct dependencies are pinned in `requirements.txt` (PyTorch 2.3.1 and Transformers 4.48.3). Install them from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Install a PyTorch build appropriate for the available CUDA runtime. Offline LLM generation requires a CUDA GPU supporting bfloat16 and sufficient memory for Qwen2.5-7B-Instruct. Classifier training, calibration, evaluation, and automated checks also support CPU execution. Complete feature tensors and model weights are external inputs. Supplied transcript metadata, figures, and classification reports are bundled as reference resources; their scope is documented below.

Download the complete [Qwen2.5-7B-Instruct model](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct), including its tokenizer and configuration, into a local directory such as `models/Qwen2.5-7B-Instruct`. Generation uses local files only; it does not contact an inference service.

All commands below are run from the root of this code archive.

## Project structure

```text
.
├── README.md
├── main.pdf
├── requirements.txt
├── pyproject.toml
├── THIRD_PARTY_NOTICES.md
├── configs/
│   ├── iemocap.json
│   └── meld.json
├── ecra/
│   ├── __init__.py
│   ├── cognitive.py
│   ├── data.py
│   ├── io.py
│   ├── layers.py
│   ├── model.py
│   ├── prepare.py
│   ├── retrieval.py
│   └── train.py
├── scripts/
│   ├── run.sh
│   ├── audit_resources.py
│   └── package_submission.py
├── resources/
│   ├── README.md
│   ├── inventory.json
│   ├── audit.json
│   ├── dataset_metadata/
│   ├── figures/
│   ├── results/
│   └── placeholders/
├── tests/
│   ├── __init__.py
│   ├── run_checks.py
│   ├── test_method.py
│   └── test_reports.py
└── docs/
    ├── DATA.md
    └── IMPLEMENTATION.md
```

`ecra/` is self-contained and does not import files from a parent project. New runtime artifacts are written to `outputs/`. Acquired feature bundles, model weights, new checkpoints, and runtime logs are excluded from packaging. The supplied reference materials under `resources/` are included in the same archive.

## Bundled resources

The [resource guide](resources/README.md) describes the transcript metadata, ECRA figures, classification reports, and file templates. `resources/inventory.json` records file sizes and SHA-256 checksums. Generate prompts and calibrated LLM caches through the pipeline below.

The classification reports contain five runs per dataset. `resources/audit.json` provides their accuracy, weighted F1, macro F1, and summary statistics.

## Data preparation

Obtain access to [IEMOCAP](https://sail.usc.edu/iemocap/) and [MELD](https://affective-meld.github.io/) under their respective data terms. The implementation consumes utterance-level pre-extracted features: 1024-dimensional RoBERTa text features, 1582/300-dimensional audio features for IEMOCAP/MELD, and 342-dimensional visual features. It does not include raw-media feature extraction.

Place the compatible feature files under `data/`:

```text
data/
├── iemocap_multimodal_features.pkl
├── meld_multimodal_features.pkl
└── meld_features_roberta.pkl
```

The final file supplies MELD's train/validation/test dialogue IDs; a JSON partition manifest can be used instead. The exact feature schema, label order, feature access reference, and split rules are documented in [docs/DATA.md](docs/DATA.md). Feature dimensions and utterance alignment are checked before preprocessing.

## Complete experiment

Run each dataset with its supplied configuration:

```bash
DATASET=iemocap MODEL_PATH=./models/Qwen2.5-7B-Instruct bash scripts/run.sh
DATASET=meld MODEL_PATH=./models/Qwen2.5-7B-Instruct bash scripts/run.sh
```

The script creates dialogue partitions, obtains confusable class pairs from five-fold training-only Base predictions, builds retrieval-grounded prompts using those pairs, generates raw LLM records, calibrates the cache, and trains the five ECRA seeds. A Base model is trained for each OOF fold before the five final ECRA runs.

Optional environment variables are `PYTHON_BIN`, `FEATURE_ROOT`, `DEVICE`, `WORK`, and `PARTITIONS`. Defaults are `python`, `./data`, `cuda:0`, `./outputs/<dataset>`, and `./data/meld_features_roberta.pkl`, respectively. `PARTITIONS` applies to MELD and can point to the JSON manifest described in the data guide.

## Individual stages

The following commands run the IEMOCAP pipeline explicitly:

```bash
python -m ecra.prepare splits --config configs/iemocap.json --features data/iemocap_multimodal_features.pkl --output outputs/iemocap
python -m ecra.train pairs --work outputs/iemocap --device cuda:0
python -m ecra.prepare prompts --config configs/iemocap.json --features data/iemocap_multimodal_features.pkl --pairs outputs/iemocap/pairs.json --output outputs/iemocap
python -m ecra.prepare generate --work outputs/iemocap --model-path models/Qwen2.5-7B-Instruct --device cuda:0 --resume
python -m ecra.prepare calibrate --work outputs/iemocap
python -m ecra.train fit --work outputs/iemocap --device cuda:0
```

For MELD, use `configs/meld.json`, the MELD feature file, and `--partitions data/meld_features_roberta.pkl` for both `splits` and `prompts`. Use a separate work directory, such as `outputs/meld`.

Generation supports resuming a raw cache with matching prompts and retrieval settings. Manifests record feature hashes, prompt hashes, the manuscript revision, the calibration policy, and dialogue IDs. Feature paths are stored relative to the work directory.

## Training configuration

| Setting | IEMOCAP | MELD |
|---|---:|---:|
| Hidden dimension / attention heads | 1024 / 8 | 1024 / 8 |
| Dropout | 0.5 | 0.5 |
| Training epochs | 100 | 50 |
| Batch size, in dialogues | 16 | 8 |
| Adam learning rate | 1e-4 | 5e-6 |
| Weight decay | 1e-5 | 1e-5 |
| Support / confidence admission thresholds | 0.60 / 0.85 | 0.60 / 0.85 |
| Unimodal / self-distillation weights | 1 / 1 | 1 / 1 |
| Class-prior alignment weight | 0.5 | 0.1 |
| Source-weight supervision weight | 0.1 | 0.1 |
| Affective geometry weight | 0.2 | 0.3 |
| Self-distillation temperature | 0.125 | 0.125 |
| Affective temperature / margin | 0.2 / 0.2 | 0.2 / 0.2 |

Adam uses betas `(0.9, 0.999)` and epsilon `1e-8`. Learning rates remain constant, and gradients are not clipped. The default training seeds are 2026–2030. Models are selected exclusively by validation weighted F1; the test set is evaluated after checkpoint selection.

Retrieval uses five examples, three preceding turns plus the target, and a text-similarity weight of 0.7. Main-experiment cache generation uses greedy decoding, bfloat16, batch size 16, a 4096-token input limit, and a 160-token output limit.

Auxiliary sample weights, their normalization denominators, and source-weight targets are detached as specified in Equations (21)–(23); gradients through final predictions and source allocation remain enabled.

The observation encoder implements Equations (1)–(3) using learned positional and speaker embeddings, nine causal Transformer branches, linear target-state reductions, and scalar three-source fusion.

Further equation mappings, initialization choices, and implementation conventions are provided in [docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md).

## Evaluation and result reproduction

Evaluate a selected checkpoint independently:

```bash
python -m ecra.train evaluate --work outputs/iemocap --checkpoint outputs/iemocap/runs/2026/best.pt --split test --device cuda:0
```

Train the observation-only Base control using the same prepared split:

```bash
python -m ecra.train fit --work outputs/iemocap --base-only --device cuda:0
```

The default `fit` command runs all five seeds and writes:

- `runs/<seed>/best.pt`: validation-selected checkpoint and its configuration.
- `runs/<seed>/history.jsonl`: training and validation metrics.
- `runs/<seed>/test_metrics.json`: accuracy, weighted/macro/class-wise F1, confusion matrix, admission rate, and prediction-flip diagnostics.
- `runs/<seed>/test_predictions.jsonl`: utterance-level predictions, probabilities, and fusion quantities.
- `summary.json`: five-seed means and sample standard deviations for accuracy, weighted F1, and macro F1.

The manuscript reports weighted F1 of **74.87 ± 0.12** on IEMOCAP and **67.51 ± 0.12** on MELD. The corresponding classification reports and five-run summaries are available under `resources/results/` and in the [resource guide](resources/README.md). To run the configured experiments, prepare the specified feature files and local LLM weights, execute the complete pipeline, and aggregate the generated `summary.json` metrics across the configured seeds.

## Verification

Run the method tests without benchmark data or LLM weights:

```bash
python -m unittest discover -s tests -v
```

Run the synthetic end-to-end integration check:

```bash
python -m tests.run_checks
```

Optionally include complete-dimension model checks with real features:

```bash
python -m tests.run_checks --feature-root data
```

The checks cover model parameter counts, scalar fusion, learned-position gradients, causality, exact rejection, residual bounds, objective values, training-only retrieval, OOF calibration, and conditional-harm accounting. Report checks verify metric aggregation and classification-report consistency. The synthetic integration check exercises five-fold Base training, prompt construction, calibration, two short training runs, validation checkpoint selection, checkpoint reload, and test evaluation. Synthetic records are generated in a temporary directory and removed automatically.

## Submission archive

Create the single upload package, including all bundled resources:

```bash
python scripts/audit_resources.py
python scripts/package_submission.py
```

Upload `dist/ecra_submission.zip`. The packager includes all organized source resources, preserves the manuscript, excludes runtime and external-weight directories, verifies archive CRCs, and prints its checksum.

## Attribution

External code provenance and resource terms are recorded in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Dataset files and pretrained model weights remain subject to their original terms and are obtained separately.
