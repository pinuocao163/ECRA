import math

import torch
from torch import nn
from torch.nn import functional as F

from .layers import TransformerEncoderLayer


class CausalBranch(nn.Module):
    def __init__(self, config):
        super().__init__()
        h = config["hidden_dim"]
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    h, config["heads"], config["ff_dim"], config["dropout"]
                )
                for _ in range(config["layers"])
            ]
        )

    def forward(self, source, target, forbidden):
        self_attention = source is target
        for index, layer in enumerate(self.layers):
            query = layer.layer_norm(target) if index else target
            memory = query if self_attention else source
            context = layer.self_attn(memory, memory, query, forbidden)
            target = layer.feed_forward(layer.dropout(context) + query)
        return target


class ECRA(nn.Module):
    def __init__(self, config, base_only=False):
        super().__init__()
        self.config = config
        self.base_only = base_only
        if config["observation_encoder"] != "causal_scalar_v1":
            raise ValueError("Unsupported observation encoder configuration")
        if config["position_encoding"] != "learned":
            raise ValueError("The encoder requires learned position embeddings")
        h = config["hidden_dim"]
        self.classes = 6 if config["dataset"] == "IEMOCAP" else 7
        self.projections = nn.ModuleDict(
            {
                m: nn.Linear(config[m + "_dim"], h, bias=False)
                for m in ("text", "audio", "visual")
            }
        )
        self.position_embeddings = nn.Embedding(config["max_positions"], h)
        self.speaker_embeddings = nn.Embedding(
            config["speakers"] + 1, h, padding_idx=config["speakers"]
        )
        self.input_dropout = nn.Dropout(config["dropout"])
        self.branches = nn.ModuleDict(
            {
                f"{target}_{source}": CausalBranch(config)
                for target in ("text", "audio", "visual")
                for source in ("text", "audio", "visual")
            }
        )
        self.reductions = nn.ModuleDict(
            {m: nn.Linear(h * 3, h) for m in ("text", "audio", "visual")}
        )
        self.base_gate = nn.Linear(h * 3, 3)
        self.branch_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ReLU(), nn.Dropout(config["dropout"]), nn.Linear(h, self.classes)
                )
                for _ in range(3)
            ]
        )
        self.classifier = nn.Linear(h, self.classes)
        if not base_only:
            self.cache_projection = nn.Sequential(
                nn.Linear(self.classes + 15, h), nn.ReLU(), nn.LayerNorm(h)
            )
            self.support_logits = nn.Parameter(
                torch.tensor(config["support_init"]).log()
            )
            self.source_gate = nn.Linear(h * 4, 4)
            self.hint_gate = nn.Linear(4, 4, bias=False)
            self.vad_gate = nn.Linear(7, 4, bias=False)
            self.residual_logit = nn.Parameter(
                torch.tensor(
                    math.log(config["residual_init"] / (1 - config["residual_init"]))
                )
            )
            self.controller = nn.Linear(2, 1)
            nn.init.zeros_(self.controller.weight)
            nn.init.zeros_(self.controller.bias)

    def forward(self, batch):
        mask = batch["mask"]
        length = mask.size(1)
        if length > self.config["max_positions"]:
            raise ValueError("Dialogue exceeds configured position encoding size")
        positions = self.position_embeddings(torch.arange(length, device=mask.device))[
            None
        ]
        speakers = batch["speakers"].masked_fill(~mask, self.config["speakers"])
        embedding = positions + self.speaker_embeddings(speakers)
        inputs = {
            m: self.input_dropout(projection(batch[m]) + embedding)
            for m, projection in self.projections.items()
        }
        future = torch.ones(length, length, dtype=torch.bool, device=mask.device).triu(
            1
        )
        forbidden = future[None] | (~mask[:, None, :])
        modalities = ("text", "audio", "visual")
        states = [
            self.reductions[target](
                torch.cat(
                    [
                        self.branches[f"{target}_{source}"](
                            inputs[source], inputs[target], forbidden
                        )
                        for source in modalities
                    ],
                    -1,
                )
            )
            for target in modalities
        ]
        mu = self.base_gate(torch.cat(states, -1)).softmax(-1)
        base = (torch.stack(states, -2) * mu.unsqueeze(-1)).sum(-2)
        output = dict(
            base=base,
            mu=mu,
            branch_logits=torch.stack(
                [head(state) for head, state in zip(self.branch_heads, states)], -2
            ),
            base_logits=self.classifier(base),
        )
        if self.base_only:
            output.update(
                final=base,
                logits=output["base_logits"],
                admission=torch.zeros_like(mask),
            )
            return output
        c = self.classes
        cache = batch["cache"]
        confidence = cache[..., c + 7]
        support = (cache[..., c + 8 : c + 12] * self.support_logits.softmax(-1)).sum(-1)
        admission = (
            mask
            & batch["available"]
            & (support >= self.config["quality_threshold"])
            & (confidence >= self.config["confidence_threshold"])
        )
        trust = confidence * (0.5 + 0.5 * support)
        cognition = self.cache_projection(cache)
        hints = torch.cat([cache[..., c + 4 : c + 7], trust.unsqueeze(-1)], -1)
        vad = torch.cat([cache[..., c : c + 4], cache[..., c + 12 : c + 15]], -1)
        logits = (
            self.source_gate(torch.cat(states + [cognition], -1))
            + self.hint_gate(hints)
            + self.vad_gate(vad)
        )
        cognitive_logit = logits[..., 3].masked_fill(~admission, float("-inf"))
        beta = torch.cat([logits[..., :3], cognitive_logit.unsqueeze(-1)], -1).softmax(
            -1
        )
        proposal = (torch.stack(states + [cognition], -2) * beta.unsqueeze(-1)).sum(-2)
        xi = self.residual_logit.sigmoid() * (
            0.5
            + 0.5
            * self.controller(torch.stack([confidence, support], -1))
            .sigmoid()
            .squeeze(-1)
        )
        eta = admission.to(base.dtype) * trust * xi
        final = base + eta.unsqueeze(-1) * (proposal - base)
        output.update(
            final=final,
            logits=self.classifier(final),
            support=support,
            trust=trust,
            admission=admission,
            beta=beta,
            eta=eta,
            xi=xi,
        )
        return output


def normalize_positive(value):
    value = value.clamp_min(1e-6)
    return value / value.sum(-1, keepdim=True)


def compute_loss(output, batch, config, pairs):
    valid = batch["mask"]
    labels = batch["labels"][valid]
    logits = output["logits"][valid]
    branches = output["branch_logits"][valid]
    loss_cls = F.cross_entropy(logits, labels)
    loss_uni = sum(F.cross_entropy(branches[:, m], labels) for m in range(3))
    temperature = config["sd_temperature"]
    teacher = (logits.detach() / temperature).softmax(-1)
    loss_sd = sum(
        F.kl_div(
            (branches[:, m] / temperature).log_softmax(-1),
            teacher,
            reduction="batchmean",
        )
        for m in range(3)
    )
    zero = logits.sum() * 0
    losses = dict(cls=loss_cls, uni=loss_uni, sd=loss_sd, llm=zero, rel=zero, aff=zero)
    if "beta" in output:
        c = logits.size(-1)
        cache = batch["cache"][valid]
        admitted = output["admission"][valid]
        support = output["support"][valid]
        confidence = cache[:, c + 7]
        weights = admitted * support * confidence
        reliability_weights = admitted * (support * confidence).clamp(0, 1)
        if config.get("detach_cognitive_targets", True):
            weights = weights.detach()
            reliability_weights = reliability_weights.detach()
        denominator = weights.sum().clamp_min(1)
        reliability_denominator = reliability_weights.sum().clamp_min(1)
        losses["llm"] = (
            weights
            * F.kl_div(logits.log_softmax(-1), cache[:, :c], reduction="none").sum(-1)
        ).sum() / denominator
        appraisal = cache[:, c : c + 4]
        valence, arousal, dominance, certainty = appraisal.unbind(-1)
        guidance = cache[:, c + 12 : c + 15]
        pos, low, control = guidance.unbind(-1)
        cue_scores = torch.stack(
            [
                certainty.abs() + low.abs() + (1 - arousal.abs()).relu(),
                arousal.abs() + control.abs(),
                valence.relu() + pos.abs() + dominance.abs(),
            ],
            -1,
        )
        modality_target = normalize_positive(
            cache[:, c + 4 : c + 7] + normalize_positive(cue_scores)
        )
        strength = (
            (appraisal[:, :3].abs().sum(-1) + guidance.abs().sum(-1)) / 3
        ).clamp(0, 1)
        share = confidence * support * strength
        target = torch.cat([(1 - share[:, None]) * modality_target, share[:, None]], -1)
        if config.get("detach_cognitive_targets", True):
            target = target.detach()
        losses["rel"] = (
            reliability_weights * (output["beta"][valid] - target).square().sum(-1)
        ).sum() / reliability_denominator
        reps = F.normalize(output["final"][valid][admitted], dim=-1, eps=1e-12)
        vad = F.normalize(appraisal[admitted, :3], dim=-1, eps=1e-12)
        labs = labels[admitted]
        if reps.size(0) > 1:
            cosine = reps @ reps.T
            kappa = torch.exp(-torch.cdist(vad, vad, p=2))
            diagonal = torch.eye(len(labs), device=labs.device, dtype=torch.bool)
            same = (labs[:, None] == labs[None, :]) & ~diagonal
            scores = cosine / config["aff_temperature"]
            log_prob = scores - scores.masked_fill(diagonal, float("-inf")).logsumexp(
                -1, keepdim=True
            )
            positive = -(kappa[same] * log_prob[same]).sum() / same.sum().clamp_min(1)
            negative_mask = torch.zeros_like(same)
            for a, b in pairs:
                negative_mask |= ((labs[:, None] == a) & (labs[None, :] == b)) | (
                    (labs[:, None] == b) & (labs[None, :] == a)
                )
            negative = (
                kappa[negative_mask]
                * (cosine[negative_mask] - config["aff_margin"]).relu()
            ).sum() / negative_mask.sum().clamp_min(1)
            losses["aff"] = positive + negative
    losses["total"] = losses["cls"] + sum(
        config["lambda_" + name] * losses[name]
        for name in ("uni", "sd", "llm", "rel", "aff")
    )
    return losses
