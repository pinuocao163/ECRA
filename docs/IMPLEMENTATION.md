# Implementation Details

This document specifies the executable conventions accompanying the ECRA manuscript. The manuscript SHA-256 is `2d7039702643cf3d74ee888b60bcd633171cc8f938b133f24ed4f9cec4bcb745`.

## Equation mapping

| Manuscript | Implementation |
|---|---|
| Equations (1)–(2) | `ecra.model.ECRA` and `CausalBranch`: modality projections, learned positional and speaker embeddings, and nine causal attention branches |
| Equation (3) | Three target-state reductions and the observation-only scalar source gate |
| Equation (4) | `ecra.retrieval.DualCueRetrieval`: training-only dual-cue evidence retrieval |
| Equations (5)–(6) | `ecra.cognitive.prior` and `load_cache`: class prior and the `C+15` cognitive input |
| Equation (7) | Four fixed evidence fields combined by a learned softmax support mixer |
| Equations (8)–(11) | Trust, binary admission, cognitive-logit masking, and four-source allocation |
| Equations (12)–(14) | Trust-scaled residual interpolation and exact same-forward rejection |
| Equation (15) | `ecra.model.compute_loss`: the complete supervised and cognitive objective |
| Equations (16)–(17) | `ecra.cognitive.calibrate_records`: dialogue-level five-fold confidence calibration |
| Equation (19) | Signed guidance in the order `[g_pos, g_low, g_ctrl]` |
| Equation (20) | Branch cross-entropy and stopped-teacher self-distillation |
| Equations (21)–(23) | Confidence/support-weighted class-prior alignment and source-weight supervision |
| Equations (24)–(26) | VAD-weighted same-class attraction and confusable-pair separation |
| Equation (34) | Conditional harmful-flip numerator, denominator, and rate in `ecra.train.run_epoch` |

## Encoder and fusion

Each interaction branch has one attention layer with feed-forward width equal to the hidden dimension. Queries use the target modality; keys and values use the source modality. The attention mask combines future-turn exclusion with padded-key exclusion. For each target modality, the three source states are concatenated in text/audio/visual order and reduced by a linear map. A linear projection of the concatenated target states produces three scalar logits; softmax yields the modality weights used for their weighted sum. Positional and speaker embeddings are learned. The position table has 512 entries, and the model enforces the configured dialogue length.

The cognitive projection is `Linear → ReLU → LayerNorm`. Modality prediction heads use `ReLU → Dropout → Linear`; the final and same-forward backbone predictions share one linear classifier. Linear layers and embeddings use PyTorch initialization. The support mixer starts uniformly; `sigmoid(theta_eta)` starts at 0.05, and the two-input controller has zero weights and bias, giving an initial `xi` of 0.0375.

Unavailable cognitive records are zero-filled. Rejected records have exactly zero cognitive allocation and residual coefficient. A valid rejected record may later re-enter when updates to the shared support mixer raise its support past the threshold. Admission indicators are nondifferentiable.

## Retrieval, parsing, and calibration

TF–IDF uses lowercase unigrams and bigrams with at most 50000 features. Each modality contributes mean absolute value, population standard deviation, maximum absolute value, and mean absolute change from the preceding utterance. The first utterance uses an all-zero preceding feature. Standardization uses training means and `max(training_sd, 1e-6)`; L2 normalization uses a `1e-6` guard. Similarity ties are resolved by dialogue ID and utterance index.

Generation uses the model's chat template, greedy decoding, bfloat16, batch 16, 4096 input tokens, and at most 160 new tokens. Left truncation preserves the target at the end of the prompt. Parsing attempts complete JSON, a brace-delimited extraction, and flat-field recovery without retries. Missing or nonfinite appraisal/hint values use manuscript Table 10 defaults. Raw confidence falls back to 0.85 before clipping. Invalid task labels are unavailable inputs. Executable prompt templates are defined in `ecra/retrieval.py`, and the parser and output-field defaults are specified in `ecra/cognitive.py`. Prompts include causal speaker/utterance context, target statistics, five labeled training references, and confusable-pair guidance.

Calibration fits positive temperatures on four training-dialogue folds and excludes the target's entire fold. Training forwards use the corresponding held-out-fold value; validation/test forwards use the mean of all five calibrator probabilities. Records retain `confidence_oof`, `confidence_mean`, and the actual `confidence` used by the model. Scalar optimization bounds `log(T)` to `[-12, 12]` for numerical stability. This split-specific assignment follows Equation (17), including the out-of-fold values used by training forwards and diagnostics.

## Objectives and evaluation

The self-distillation temperature is the mathematical value `1/8 = 0.125` stated in Appendix A.2 for both datasets, without a squared-temperature multiplier. The self-distillation teacher is stopped. Under Equations (21)–(23), auxiliary sample weights, their normalization denominators, and the source-weight target are recomputed every forward pass and detached during backpropagation. The supplied configurations set `detach_cognitive_targets=true`. Gradients through final predictions and source allocation remain enabled, so admitted records can still update the support mixer through allocation and correction paths.

Final and branch cross-entropy average over real utterances. Cognitive alignment uses `max(sum(weights), 1)` normalization. Class-prior weights are `A*rho*gamma`; source-weight supervision uses `A*clip(rho*gamma,0,1)`. All such weights and denominators are detached. Affective losses use admitted records only, ordered pairs, VAD normalization with a `1e-12` guard, and zero contribution for empty pair sets. Confusable pairs are ranked using first-seed training-only OOF Base confusion counts and fixed VAD prototypes; the top three are shared by all final training seeds. Each OOF Base fit uses an internal fixed-seed validation subset of its training portion for checkpoint selection.

Five final training seeds and split/fold seed 2026 are specified in the configurations. Test evaluation follows validation checkpoint selection. Conditional harm is measured on wrong cached labels with confidence at least 0.85; its denominator includes only examples correctly classified by that same model's backbone. A zero denominator produces `null`. Accuracy changes are checked against `(beneficial_flips - harmful_flips) / utterances`; no weighted-F1 change is inferred from that identity.

## Configuration

The supplied dataset configurations select `causal_scalar_v1` with `learned` position embeddings. They specify the encoder dimensions, optimizer, training schedule, retrieval settings, confidence policy, and objective weights. The ECRA and observation-only Base training commands share the same feature schema and partition manifest. Run `python -m unittest discover -s tests -v` to verify model dimensions, fusion weights, gradient paths, causality, and objective invariants.
