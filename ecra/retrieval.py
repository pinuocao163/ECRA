import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .data import LABELS


def speaker_name(value):
    array = np.asarray(value)
    return f"Speaker{array.argmax()}" if array.ndim else str(value)


def context(features, vid, turn, window):
    return "\n".join(
        f"{speaker_name(features['speakers'][vid][i])}: {features['sentences'][vid][i]}"
        for i in range(max(0, turn - window), turn + 1)
    )


def utterance_stats(sequence, turn):
    current = np.asarray(sequence[turn], dtype=np.float64)
    previous = (
        np.asarray(sequence[turn - 1], dtype=np.float64)
        if turn
        else np.zeros_like(current)
    )
    return [
        float(np.abs(current).mean()),
        float(current.std()),
        float(np.abs(current).max()),
        float(np.abs(current - previous).mean()),
    ]


def multimodal_stats(features, vid, turn):
    return utterance_stats(features["audio"][vid], turn) + utterance_stats(
        features["visual"][vid], turn
    )


class DualCueRetrieval:
    def __init__(self, features, train_vids, config):
        self.features = features
        self.config = config
        self.records = []
        corpus, statistics = [], []
        labels = LABELS[config["dataset"]]
        for vid in train_vids:
            for turn in range(len(features["sentences"][vid])):
                text = context(features, vid, turn, config["context_window"])
                stats = multimodal_stats(features, vid, turn)
                self.records.append(
                    dict(
                        vid=vid,
                        turn_id=turn,
                        label=labels[int(features["labels"][vid][turn])],
                        context=text,
                        multimodal_stats=stats,
                    )
                )
                corpus.append(text)
                statistics.append(stats)
        self.bank_vids = np.asarray([record["vid"] for record in self.records])
        self.bank_turns = np.asarray([record["turn_id"] for record in self.records])
        self.vectorizer = TfidfVectorizer(
            lowercase=True, ngram_range=(1, 2), max_features=50000
        )
        self.text_matrix = self.vectorizer.fit_transform(corpus)
        statistics = np.asarray(statistics)
        self.mean = statistics.mean(0)
        self.std = statistics.std(0).clip(1e-6)
        self.mm_matrix = self.normalize(statistics)

    def normalize(self, values):
        values = (np.asarray(values) - self.mean) / self.std
        return values / np.linalg.norm(values, axis=-1, keepdims=True).clip(1e-6)

    def retrieve(self, vid, turn):
        query = context(self.features, vid, turn, self.config["context_window"])
        stats = multimodal_stats(self.features, vid, turn)
        text_scores = (
            (self.text_matrix @ self.vectorizer.transform([query]).T).toarray().ravel()
        )
        mm_scores = np.einsum("ij,j->i", self.mm_matrix, self.normalize(stats))
        weight = self.config["text_rag_weight"]
        scores = weight * text_scores + (1 - weight) * mm_scores
        eligible = np.flatnonzero(self.bank_vids != vid)
        ranking = np.lexsort(
            (self.bank_turns[eligible], self.bank_vids[eligible], -scores[eligible])
        )
        order = eligible[ranking[: self.config["rag_k"]]]
        examples = [
            dict(
                self.records[i],
                score=float(scores[i]),
                text_score=float(text_scores[i]),
                multimodal_score=float(mm_scores[i]),
            )
            for i in order
        ]
        return query, stats, examples

    def metadata(self):
        return dict(
            statistic_mean=self.mean.tolist(),
            statistic_std=self.std.tolist(),
            vocabulary_size=len(self.vectorizer.vocabulary_),
            bank_size=len(self.records),
            context_window=self.config["context_window"],
            rag_k=self.config["rag_k"],
            text_rag_weight=self.config["text_rag_weight"],
        )


def make_messages(query, stats, examples, dataset, confusion_pairs=None):
    labels = LABELS[dataset]
    pairs = (
        "happy/excited, angry/frustrated, neutral/sad"
        if dataset == "IEMOCAP"
        else "fear/surprise, anger/disgust, neutral/sadness"
    )
    if confusion_pairs is not None:
        if len(confusion_pairs) != 3 or any(
            a == b or a not in range(len(labels)) or b not in range(len(labels))
            for a, b in confusion_pairs
        ):
            raise ValueError("Expected three valid confusion pairs")
        pairs = ", ".join(f"{labels[a]}/{labels[b]}" for a, b in confusion_pairs)
    demonstrations = "\n\n".join(
        f"Example {i + 1} (gold label: {example['label']}):\n{example['context']}\nAudio-visual statistics: {example['multimodal_stats']}"
        for i, example in enumerate(examples)
    )
    prompt = (
        f"Choose exactly one emotion label from: {', '.join(labels)}.\n"
        "Return only one JSON object with keys label, confidence, rationale, appraisal, modality_hint.\n"
        "confidence is label-correctness confidence in [0,1]. appraisal contains valence, arousal, dominance, certainty in [-1,1]. "
        "certainty is signed affective certainty: negative means hesitation or uncertainty, zero means indeterminate, positive means increasing affective certainty. "
        "modality_hint contains nonnegative text, audio, visual weights summing to one.\n"
        f"Use VAD to separate confusing pairs such as {pairs}.\n"
        "Use retrieved examples to calibrate label boundaries and modality reliability. Do not copy labels blindly.\n"
        f"Retrieved labeled examples from training dialogues:\n{demonstrations}\n\n"
        "Audio-visual statistics order: audio mean absolute value, SD, maximum absolute value, mean absolute change; "
        "then the same four statistics for visual features. These are feature statistics, not direct emotion measurements.\n"
        f"Current audio-visual statistics: {stats}\n\n"
        f"Causal dialogue context ending at the target utterance:\n{query}\n"
        "Predict the emotion of the last utterance only. Do not use markdown or extra text."
    )
    return [
        dict(
            role="system",
            content="You are an expert at multimodal emotion recognition. Reason over causal dialogue context, retrieved evidence, VAD appraisal, and modality reliability.",
        ),
        dict(role="user", content=prompt),
    ]
