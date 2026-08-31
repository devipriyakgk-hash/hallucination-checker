"""
FactCheck-Lite: Hallucination / Factual Consistency Checker

Given a SOURCE document and a CLAIM (e.g. a sentence from an LLM's output),
determines whether the claim is:
  - SUPPORTED     (entailed by the source)
  - CONTRADICTED  (likely a hallucination)
  - UNVERIFIABLE  (source doesn't contain enough info either way)

Primary method: zero-shot Natural Language Inference (NLI) using a pretrained
transformer (facebook/bart-large-mnli). This is the "real" model — it needs
an internet connection the first time it runs, to download model weights
from Hugging Face (a few hundred MB, one-time only, then cached locally).

Fallback method: a lightweight TF-IDF cosine-similarity heuristic that needs
no downloads at all. It's far less accurate but lets the tool run instantly
offline — useful for demos, testing, or environments with no internet. The
checker automatically falls back to this if the transformer model can't be
loaded (e.g. no internet, or `transformers` not installed).
"""
from dataclasses import dataclass


@dataclass
class ConsistencyResult:
    label: str          # "SUPPORTED" | "CONTRADICTED" | "UNVERIFIABLE"
    confidence: float    # 0-1
    method: str          # "nli-transformer" | "tfidf-fallback"


class FactConsistencyChecker:
    def __init__(self, prefer_transformer=True):
        self.pipeline = None
        self.method = "tfidf-fallback"

        if prefer_transformer:
            try:
                from transformers import pipeline
                self.pipeline = pipeline(
                    "zero-shot-classification",
                    model="facebook/bart-large-mnli"
                )
                self.method = "nli-transformer"
            except Exception as e:
                print(f"[FactConsistencyChecker] Falling back to TF-IDF "
                      f"heuristic (transformer unavailable: {e})")

    def check(self, source: str, claim: str) -> ConsistencyResult:
        if self.method == "nli-transformer":
            return self._check_nli(source, claim)
        return self._check_tfidf(source, claim)

    def _check_nli(self, source: str, claim: str) -> ConsistencyResult:
        # Frame as NLI: does the source entail, contradict, or stay neutral
        # about the claim?
        candidate_labels = ["entailment", "contradiction", "neutral"]
        hypothesis_template = "This example is {}."
        result = self.pipeline(
            f"Premise: {source}\nHypothesis: {claim}",
            candidate_labels,
            hypothesis_template=hypothesis_template,
        )
        top_label = result["labels"][0]
        top_score = result["scores"][0]

        label_map = {
            "entailment": "SUPPORTED",
            "contradiction": "CONTRADICTED",
            "neutral": "UNVERIFIABLE",
        }
        return ConsistencyResult(
            label=label_map[top_label],
            confidence=round(top_score, 4),
            method="nli-transformer",
        )

    def _check_tfidf(self, source: str, claim: str) -> ConsistencyResult:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        vectorizer = TfidfVectorizer().fit([source, claim])
        vectors = vectorizer.transform([source, claim])
        similarity = cosine_similarity(vectors[0], vectors[1])[0][0]

        # Heuristic thresholds — crude but functional for a fallback.
        if similarity > 0.35:
            label = "SUPPORTED"
        elif similarity < 0.08:
            label = "UNVERIFIABLE"
        else:
            label = "CONTRADICTED"

        return ConsistencyResult(
            label=label,
            confidence=round(float(similarity), 4),
            method="tfidf-fallback",
        )


if __name__ == "__main__":
    checker = FactConsistencyChecker(prefer_transformer=True)
    print(f"Using method: {checker.method}\n")

    examples = [
        (
            "The Eiffel Tower was completed in 1889 and is located in Paris, France.",
            "The Eiffel Tower is located in Paris.",
        ),
        (
            "The Eiffel Tower was completed in 1889 and is located in Paris, France.",
            "The Eiffel Tower was built in London in 1920.",
        ),
        (
            "The Eiffel Tower was completed in 1889 and is located in Paris, France.",
            "The Eiffel Tower is the tallest building in the world.",
        ),
    ]

    for source, claim in examples:
        result = checker.check(source, claim)
        print(f"SOURCE: {source}")
        print(f"CLAIM:  {claim}")
        print(f"-> {result.label} (confidence={result.confidence}, method={result.method})\n")
