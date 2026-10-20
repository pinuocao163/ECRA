# Data and Feature Specification

## Access

IEMOCAP and MELD are obtained from their dataset maintainers:

- IEMOCAP: <https://sail.usc.edu/iemocap/>
- MELD: <https://affective-meld.github.io/>

The pre-extracted multimodal feature distribution referenced by the SDT/TEN release is available through its [feature access link](https://drive.google.com/drive/folders/1J1mvbqQmVodNBzbiOIxRiWOtkP6qqP-K?usp=sharing). Availability and access permissions are controlled by the data providers. This archive does not redistribute complete multimodal feature bundles. Supplied transcript/label exports are included under `resources/`, as described in the [resource guide](../resources/README.md). When preparing features independently, match the dimensions, utterance ordering, speaker ordering, and dataset partitions below; alternative feature extractors or embeddings constitute a different input setting.

## Feature files

Each multimodal `.pkl` file contains a sequence with these ordered fields:

| Index | Field | Per-dialogue content |
|---:|---|---|
| 0 | Utterance IDs | Utterance identifiers in sequence order |
| 1 | Speakers | IEMOCAP `M`/`F` strings, or speaker one-hot vectors |
| 2 | Labels | Integer emotion labels |
| 3 | Text features | Float array of shape `[T, 1024]` |
| 4–6 | Additional text-feature dictionaries | Preserved input-format fields; not consumed by the model |
| 7 | Audio features | `[T, 1582]` for IEMOCAP or `[T, 300]` for MELD |
| 8 | Visual features | Float array of shape `[T, 342]` |
| 9 | Sentences | Original utterance text in sequence order |
| 10 | Development dialogue IDs | Dialogues available for training and validation |
| 11 | Test dialogue IDs | Held-out test dialogues |
| 12 | Optional additional field | Accepted for compatibility with the MELD distribution |

Fields 0–9 are dictionaries keyed by dialogue ID. Numeric and string IDs are normalized to strings. All sequences belonging to one dialogue must have equal length, and all feature values must be finite. Speaker one-hot vectors use two entries for IEMOCAP and nine for MELD. Load feature pickle files only from the intended data distribution.

### Label indices

| Index | IEMOCAP | MELD |
|---:|---|---|
| 0 | happy | neutral |
| 1 | sad | surprise |
| 2 | neutral | fear |
| 3 | angry | sadness |
| 4 | excited | joy |
| 5 | frustrated | disgust |
| 6 | — | anger |

## Dialogue partitions

IEMOCAP reserves 10% of the 120 development dialogues for validation using the configured split seed. The default split contains 108 training dialogues, 12 validation dialogues, and 31 test dialogues. Its utterance counts are 5278, 532, and 1623, respectively.

MELD retains the supplied training, validation, and test IDs. In the compatible `meld_features_roberta.pkl` sequence, fields 8, 9, and 10 contain the training, test, and validation ID lists, respectively. Only those lists are used; model text features come from field 3 of the multimodal feature file.

Alternatively, supply a JSON file to `--partitions` or the `PARTITIONS` script variable:

```json
{
  "train": ["training_dialogue_id"],
  "val": ["validation_dialogue_id"],
  "test": ["test_dialogue_id"]
}
```

The values above illustrate the schema and must be replaced with the dataset's actual IDs. Partitions must be disjoint and cover the development and test dialogue sets exactly. The feature distribution inspected for this release contains 1038/114/280 nonempty MELD dialogues and 9989/1109/2610 utterances.

## Evidence and cache construction

TF–IDF fitting, audio-visual standardization, evidence labels, confidence calibration, and confusable-pair selection use training dialogues only. Retrieval excludes the entire target dialogue. Every target and evidence context ends at its respective utterance. Validation and test gold labels are never included in LLM prompts or calibration fitting.

The standard workflow derives confusable pairs before constructing prompts. Omitting `--pairs` uses the reference class pairs reported in the manuscript and records that choice in the manifest. Use `scripts/run.sh` for the training-derived protocol.

A prepared work directory contains `manifest.json`, `pairs.json`, `prompts.jsonl`, `raw_cache.jsonl`, and the calibrated `cache.jsonl`, with accompanying metadata files. These are generated locally and are not submission assets. Calibration requires all expected target records, rejects duplicate keys, and checks that every evidence record comes from another training dialogue. Invalid generated labels produce unavailable cognitive inputs.

Manifests store feature paths relative to the work directory. Moving both data and work directories together preserves the reference. Cached records are tied to the feature hash, prompt hash, paper revision, dialogue partitions, and confidence policy.

Feature filename references are available under `resources/placeholders/features/`. Place the binary tensors in `data/` and use the feature-aligned dialogue IDs when preparing partitions.
