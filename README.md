# FactCheck-Lite 🔍

A lightweight hallucination / factual-consistency checker for LLM outputs, built on zero-shot Natural Language Inference (NLI). Give it a source document and a claim — it tells you whether the claim is actually supported by the source, contradicted (a likely hallucination), or unverifiable.

## Why this exists

LLMs sometimes generate text that sounds fluent but isn't actually backed by the source material they were given (RAG context, retrieved documents, etc.). This tool automates the check: "does this generated sentence actually follow from what we gave the model?"

## How it works

**Primary method — zero-shot NLI:**
Frames the check as a Natural Language Inference problem using a pretrained `facebook/bart-large-mnli` model via Hugging Face `transformers`:
- **Entailment** → claim is SUPPORTED by the source
- **Contradiction** → claim is CONTRADICTED (likely hallucination)
- **Neutral** → UNVERIFIABLE from the given source

No training required — this is a genuinely pretrained model doing real semantic reasoning, not keyword matching.

**Fallback method — TF-IDF cosine similarity:**
If there's no internet connection to download the transformer weights (or `transformers` isn't installed), the tool automatically falls back to a simple TF-IDF + cosine-similarity heuristic so it still runs. **This fallback is intentionally crude** — it can be fooled by shared keywords even when the actual claim is false (see the honest limitation below). It exists purely so the tool is runnable in constrained environments, not as a real alternative to the NLI model.

## Honest limitation (worth mentioning in any writeup / demo)

Tested with:
- Source: *"The Eiffel Tower was completed in 1889 and is located in Paris, France."*
- Claim: *"The Eiffel Tower was built in London in 1920."*

The **TF-IDF fallback incorrectly labels this SUPPORTED** — it sees the shared words "Eiffel Tower" and treats that as high similarity, missing that the actual facts (location, year) are contradicted. This is exactly why semantic NLI models matter over keyword-overlap heuristics for factual consistency checking. The real transformer model correctly catches this as a contradiction.

## Running it

```bash
pip install -r requirements.txt

# CLI demo (3 example checks)
python src/checker.py

# Interactive web demo
streamlit run app.py
```

First run downloads the `facebook/bart-large-mnli` model (~1.6GB) from Hugging Face — needs internet, one-time only, then cached locally.

## Project structure

```
hallucination-checker/
├── src/
│   └── checker.py     # FactConsistencyChecker class (NLI + fallback)
├── app.py              # Streamlit interactive demo
├── requirements.txt
└── README.md
```

## Next steps

- [ ] Benchmark against a labeled hallucination dataset (e.g. TruthfulQA, HaluEval)
- [ ] Add sentence-level granularity — split multi-sentence LLM outputs and check each claim separately
- [ ] Try a purpose-built hallucination model (e.g. `vectara/hallucination_evaluation_model`) and compare against general-purpose NLI
- [ ] Wrap as an API endpoint for use as a RAG pipeline guardrail

---
*Exploring NLP applications in explainability and trustworthy AI. Built by Devipriya G.*
