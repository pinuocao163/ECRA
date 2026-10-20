import copy
import json
import unittest
from pathlib import Path

import numpy as np
import torch

from ecra.cognitive import (
    APPRAISALS,
    calibrate_records,
    guidance,
    parse_response,
    prior,
)
from ecra.data import load_config
from ecra.model import ECRA, compute_loss
from ecra.io import digest
from ecra.retrieval import DualCueRetrieval


def tiny_config(dataset="IEMOCAP"):
    config = load_config(
        Path(__file__).resolve().parents[1] / "configs" / (dataset.lower() + ".json")
    )
    config.update(
        hidden_dim=16,
        ff_dim=16,
        heads=4,
        text_dim=8,
        audio_dim=6,
        visual_dim=4,
        dropout=0.0,
        max_positions=32,
    )
    return config


def example_batch(config):
    classes = 6 if config["dataset"] == "IEMOCAP" else 7
    cache = torch.zeros(2, 4, classes + 15)
    for i in range(2):
        for j in range(4):
            label = j % classes
            a = APPRAISALS["happy"]
            vector = np.concatenate(
                [
                    prior(label, 0.95, classes),
                    a,
                    [0.45, 0.25, 0.30],
                    [0.95],
                    [1.0, 0.9, 0.85, 1.0],
                    guidance(a),
                ]
            )
            cache[i, j] = torch.tensor(vector, dtype=torch.float32)
    return dict(
        text=torch.randn(2, 4, config["text_dim"]),
        audio=torch.randn(2, 4, config["audio_dim"]),
        visual=torch.randn(2, 4, config["visual_dim"]),
        speakers=torch.zeros(2, 4, dtype=torch.long),
        labels=torch.tensor([[0, 1, 2, 3], [0, 1, 2, 0]]),
        mask=torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]], dtype=torch.bool),
        cache=cache,
        available=torch.ones(2, 4, dtype=torch.bool),
    )


class MethodTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(2)
        self.config = tiny_config()
        self.batch = example_batch(self.config)

    def test_model_parameter_counts(self):
        root = Path(__file__).resolve().parents[1]
        expected = {
            "iemocap": (69735451, 69776467, 69776467),
            "meld": (68433951, 68475991, 68475991),
        }
        for dataset, (base_count, total_count, trainable_count) in expected.items():
            config = load_config(root / "configs" / (dataset + ".json"))
            with torch.device("meta"):
                base = ECRA(config, base_only=True)
                model = ECRA(config)
            self.assertEqual(sum(p.numel() for p in base.parameters()), base_count)
            self.assertEqual(sum(p.numel() for p in model.parameters()), total_count)
            self.assertEqual(
                sum(p.numel() for p in model.parameters() if p.requires_grad),
                trainable_count,
            )
            self.assertEqual(len(model.branches), 9)

    def test_scalar_fusion_and_learned_position_gradients(self):
        model = ECRA(self.config)
        captured = {}

        def capture_states(module, arguments):
            captured["states"] = arguments[0].reshape(2, 4, 3, 16)

        hook = model.base_gate.register_forward_pre_hook(capture_states)
        output = model(self.batch)
        hook.remove()
        self.assertEqual(output["mu"].shape, (2, 4, 3))
        torch.testing.assert_close(output["mu"].sum(-1), torch.ones(2, 4))
        torch.testing.assert_close(
            output["base"],
            (captured["states"] * output["mu"].unsqueeze(-1)).sum(-2),
        )
        compute_loss(output, self.batch, self.config, [[0, 1]])["total"].backward()
        self.assertGreater(model.base_gate.weight.grad.abs().sum().item(), 0)
        self.assertTrue(model.position_embeddings.weight.requires_grad)
        self.assertGreater(
            model.position_embeddings.weight.grad[:4].abs().sum().item(), 0
        )
        torch.testing.assert_close(
            model.position_embeddings.weight.grad[4:], torch.zeros(28, 16)
        )

    def test_exact_rejection_and_convex_bound(self):
        model = ECRA(self.config).eval()
        self.batch["available"][0, 1] = False
        self.batch["cache"][0, 2, 13] = 0.3
        out = model(self.batch)
        rejected = ~out["admission"]
        self.assertTrue(torch.equal(out["base"][rejected], out["final"][rejected]))
        self.assertTrue(
            torch.equal(out["base_logits"][rejected], out["logits"][rejected])
        )
        self.assertTrue((out["beta"][rejected][:, 3] == 0).all())
        self.assertTrue((out["eta"][rejected] == 0).all())
        self.assertTrue((out["eta"] >= 0).all())
        self.assertTrue((out["eta"] <= out["trust"] + 1e-7).all())
        torch.testing.assert_close(out["beta"].sum(-1), torch.ones_like(out["eta"]))
        torch.testing.assert_close(out["xi"], torch.full_like(out["xi"], 0.0375))

    def test_causality_and_padding(self):
        model = ECRA(self.config).eval()
        changed = copy.deepcopy(self.batch)
        for modality in ("text", "audio", "visual"):
            changed[modality][:, 2:] = torch.randn_like(changed[modality][:, 2:]) * 100
        changed["speakers"][:, 2:] = 1
        changed["cache"][:, 2:] = 0
        with torch.no_grad():
            original = model(self.batch)
            perturbed = model(changed)
        torch.testing.assert_close(
            original["logits"][:, :2], perturbed["logits"][:, :2], rtol=0, atol=0
        )
        for modality in ("text", "audio", "visual"):
            changed[modality] = self.batch[modality].clone()
            changed[modality][1, 3] = 1000
        changed["speakers"] = self.batch["speakers"].clone()
        changed["cache"] = self.batch["cache"].clone()
        with torch.no_grad():
            padded = model(changed)
        torch.testing.assert_close(
            original["logits"][1, :3], padded["logits"][1, :3], rtol=0, atol=0
        )

    def test_loss_gradients_and_zero_admission(self):
        model = ECRA(self.config)
        self.batch["labels"] = torch.tensor([[0, 0, 1, 3], [0, 1, 1, 0]])
        out = model(self.batch)
        losses = compute_loss(out, self.batch, self.config, [[0, 1]])
        self.assertTrue(all(torch.isfinite(value) for value in losses.values()))
        losses["total"].backward()
        for parameter in model.parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertGreater(model.support_logits.grad.abs().sum().item(), 0)
        self.batch["available"].zero_()
        losses = compute_loss(model(self.batch), self.batch, self.config, [[0, 1]])
        for name in ("llm", "rel", "aff"):
            self.assertEqual(losses[name].item(), 0.0)

    def test_detached_auxiliary_targets_and_stopped_teacher(self):
        model = ECRA(self.config)
        output = model(self.batch)
        output["logits"].retain_grad()
        losses = compute_loss(output, self.batch, self.config, [])
        losses["sd"].backward(retain_graph=True)
        self.assertIsNone(output["logits"].grad)
        model.zero_grad(set_to_none=True)
        output = model(self.batch)
        output["support"].retain_grad()
        output["beta"] = output["beta"].detach().requires_grad_()
        losses = compute_loss(output, self.batch, self.config, [])
        losses["rel"].backward()
        self.assertIsNone(output["support"].grad)
        self.assertIsNotNone(output["beta"].grad)
        self.assertGreater(output["beta"].grad.abs().sum().item(), 0)
        model.zero_grad(set_to_none=True)
        output = model(self.batch)
        output["support"].retain_grad()
        output["logits"] = output["logits"].detach().requires_grad_()
        losses = compute_loss(output, self.batch, self.config, [])
        losses["llm"].backward()
        self.assertIsNone(output["support"].grad)
        self.assertGreater(output["logits"].grad.abs().sum().item(), 0)

    def test_closed_form_objectives(self):
        classes = 6
        cache = torch.zeros(1, 2, classes + 15)
        cache[..., :classes] = torch.tensor(prior(0, 0.9, classes))
        cache[..., classes + 4 : classes + 7] = 1 / 3
        cache[..., classes + 7] = 0.9
        batch = dict(
            mask=torch.ones(1, 2, dtype=torch.bool),
            labels=torch.zeros(1, 2, dtype=torch.long),
            cache=cache,
        )
        output = dict(
            logits=torch.zeros(1, 2, classes, requires_grad=True),
            branch_logits=torch.zeros(1, 2, 3, classes, requires_grad=True),
            admission=torch.ones(1, 2, dtype=torch.bool),
            support=torch.ones(1, 2),
            beta=torch.full((1, 2, 4), 0.25),
            final=torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
        )
        losses = compute_loss(output, batch, self.config, [])
        self.assertAlmostEqual(losses["cls"].item(), np.log(6), places=6)
        self.assertAlmostEqual(losses["uni"].item(), 3 * np.log(6), places=6)
        self.assertAlmostEqual(losses["sd"].item(), 0.0, places=6)
        expected_kl = 0.9 * np.log(0.9 * 6) + 0.1 * np.log(0.02 * 6)
        self.assertAlmostEqual(losses["llm"].item(), expected_kl, places=6)
        self.assertAlmostEqual(losses["rel"].item(), 0.25, delta=1e-6)
        self.assertAlmostEqual(losses["aff"].item(), 0.0, places=6)

    def test_conditional_harm_and_flip_accounting(self):
        from ecra.train import run_epoch

        batch = {key: value[:1].clone() for key, value in self.batch.items()}
        batch["vids"] = ["diagnostic_dialogue"]
        batch["labels"] = torch.tensor([[0, 0, 1, 1]])
        for turn, label in enumerate([1, 1, 0, 0]):
            batch["cache"][0, turn, :6] = torch.tensor(prior(label, 0.95, 6))
        base_predictions = torch.tensor([[1, 0, 1, 0]])
        final_predictions = torch.tensor([[0, 1, 1, 0]])

        class FixedPredictions(torch.nn.Module):
            def forward(self, values):
                return dict(
                    logits=torch.nn.functional.one_hot(final_predictions, 6).float()
                    * 5,
                    base_logits=torch.nn.functional.one_hot(base_predictions, 6).float()
                    * 5,
                    branch_logits=torch.zeros(1, 4, 3, 6),
                    admission=torch.ones(1, 4, dtype=torch.bool),
                    support=torch.ones(1, 4),
                    beta=torch.full((1, 4, 4), 0.25),
                    eta=torch.ones(1, 4) * 0.02,
                    trust=torch.ones(1, 4) * 0.9,
                    final=torch.eye(4)[None],
                )

        metrics, rows = run_epoch(FixedPredictions(), [batch], self.config, [], "cpu")
        self.assertEqual(metrics["beneficial_flips"], 1)
        self.assertEqual(metrics["harmful_flips"], 1)
        self.assertEqual(metrics["base_correct"], 2)
        self.assertEqual(metrics["final_correct"], 2)
        self.assertEqual(metrics["accuracy_change"], metrics["flip_accuracy_change"])
        self.assertEqual(
            metrics["conditional_harm"], dict(numerator=1, denominator=2, rate=0.5)
        )
        self.assertEqual(len(rows), 4)

    def test_revision_configuration_and_cue_order(self):
        for dataset, llm, aff in [("iemocap", 0.5, 0.2), ("meld", 0.1, 0.3)]:
            config = load_config(
                Path(__file__).resolve().parents[1] / "configs" / (dataset + ".json")
            )
            self.assertEqual(config["method"], "ECRA")
            self.assertEqual(
                config["paper_sha256"],
                digest(Path(__file__).resolve().parents[1] / "main.pdf"),
            )
            self.assertEqual(config["lambda_llm"], llm)
            self.assertEqual(config["lambda_aff"], aff)
            self.assertEqual(config["lambda_rel"], 0.1)
            self.assertEqual(config["generation_batch_size"], 16)
            self.assertEqual(config["confidence_policy"], "split_specific_oof")
            self.assertTrue(config["detach_cognitive_targets"])
        np.testing.assert_allclose(guidance([-0.8, 0.5, -0.6, 0]), [0.0, 0.0, -0.24])
        np.testing.assert_allclose(guidance([-0.8, -0.5, -0.6, 0]), [0.0, 0.4, 0.0])

    def test_parser_and_prior(self):
        labels = ["happy", "sad", "neutral", "angry", "excited", "frustrated"]
        response = (
            "```json\n"
            + json.dumps(
                dict(
                    label="neutral",
                    confidence=0.9,
                    appraisal=dict(valence=0, arousal=0, dominance=0, certainty=-0.4),
                    modality_hint=dict(text=0, audio=0, visual=0),
                )
            )
            + "\n```"
        )
        record = parse_response(response, labels)
        self.assertEqual(record["appraisal"], [0.0, 0.0, 0.0, -0.4])
        self.assertAlmostEqual(sum(record["modality_hint"]), 1.0)
        invalid = parse_response('{"label":"unknown","confidence":NaN}', labels)
        self.assertIsNone(invalid["label"])
        self.assertEqual(invalid["raw_confidence"], 0.85)
        partial = parse_response(
            "label: angry, confidence: 0.92, rationale: raised voice", labels
        )
        self.assertEqual(partial["label"], "angry")
        values = prior(2, 0.0, 6)
        self.assertEqual(int(values.argmax()), 2)
        self.assertAlmostEqual(float(values.sum()), 1.0, places=6)
        self.assertEqual(guidance([0.8, -0.5, 1.0, 1.0])[0], -0.4)

    def test_train_only_retrieval_and_calibration(self):
        features = {
            field: {}
            for field in ("speakers", "sentences", "audio", "visual", "labels")
        }
        vids = [str(i) for i in range(12)]
        for i, vid in enumerate(vids):
            features["speakers"][vid] = ["M", "F"]
            features["sentences"][vid] = [
                f"hello speaker {i}",
                f"emotional response {i}",
            ]
            features["audio"][vid] = np.array([[i + 1.0, 2.0], [3.0, i + 4.0]])
            features["visual"][vid] = np.array([[i + 2.0, 3.0], [4.0, i + 5.0]])
            features["labels"][vid] = [i % 6, (i + 1) % 6]
        splits = dict(train=vids[:10], val=vids[10:11], test=vids[11:])
        index = DualCueRetrieval(features, splits["train"], self.config)
        records = []
        for vid in vids:
            for turn in range(2):
                _, _, examples = index.retrieve(vid, turn)
                self.assertTrue(
                    all(
                        ex["vid"] in splits["train"] and ex["vid"] != vid
                        for ex in examples
                    )
                )
                record = parse_response(
                    '{"label":"happy","confidence":0.91}',
                    ["happy", "sad", "neutral", "angry", "excited", "frustrated"],
                )
                record.update(vid=vid, turn_id=turn, retrieved_examples=examples)
                records.append(record)
        a, metadata = calibrate_records(
            copy.deepcopy(records), features, splits, self.config
        )
        changed = copy.deepcopy(features)
        for vid in splits["val"] + splits["test"]:
            changed["labels"][vid] = [5, 5]
        b, other = calibrate_records(
            copy.deepcopy(records), changed, splits, self.config
        )
        self.assertEqual(metadata, other)
        self.assertEqual([r["confidence"] for r in a], [r["confidence"] for r in b])
        fold = metadata["heldout_fold"]["0"]
        for vid in splits["train"]:
            if metadata["heldout_fold"][vid] == fold:
                changed["labels"][vid] = [5, 5]
        _, other = calibrate_records(
            copy.deepcopy(records), changed, splits, self.config
        )
        self.assertEqual(metadata["temperatures"][fold], other["temperatures"][fold])
        train_record = next(record for record in a if record["vid"] == "0")
        self.assertEqual(train_record["confidence"], train_record["confidence_oof"])
        self.assertIsNotNone(train_record["confidence_oof"])
        ensemble_config = dict(self.config, confidence_policy="ensemble_mean")
        ensemble_records, _ = calibrate_records(
            copy.deepcopy(records), features, splits, ensemble_config
        )
        self.assertEqual(
            ensemble_records[0]["confidence"], ensemble_records[0]["confidence_mean"]
        )
        for record in a:
            if record["split"] != "train":
                self.assertIsNone(record["confidence_oof"])
                self.assertEqual(record["confidence"], record["confidence_mean"])
        changed["audio"]["11"][:] = 1e6
        changed_index = DualCueRetrieval(changed, splits["train"], self.config)
        np.testing.assert_array_equal(index.mean, changed_index.mean)
        records[0]["retrieved_examples"][0]["vid"] = "0"
        with self.assertRaises(ValueError):
            calibrate_records(records, features, splits, self.config)


if __name__ == "__main__":
    unittest.main()
