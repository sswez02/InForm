# InForm - Retrieval

InForm is designed around **trust**: answers are produced from a **curated corpus** of peer-reviewed research and must be supported by **retrieved excerpts**. The system does not browse the web at query time

---

## Corpus representation

The core data asset is a curated corpus of **115 studies** maintained as:

- `studies_master.csv` (metadata, tags, study IDs)
- per-study JSON files containing:
  - Structured metadata (title, year, authors, DOI when available, notes)
  - A list of short, pre-extracted **passages**

---

## Hybrid retrieval (lexical + semantic)

Retrieval is hybrid:

1. **TF-IDF (lexical precision)**
   - Good for exact terminology and specific phrasing
2. **Dense retriever (semantic similarity)**
   - SentenceTransformer: `all-MiniLM-L6-v2`
   - Good for paraphrases and concept matching

### Blend weights

Scores are combined with a weighted scheme:

- **0.4 TF-IDF**
- **0.6 dense**

This improves recall for user phrasing while preserving precision for domain terms

---

## Recency-aware re-ranking

After initial retrieval, results are re-ranked to prefer newer evidence **when relevance is comparable**. This helps surface updated research without overriding relevance

---

## Mode-aware retrieval + response style

InForm supports two user modes that influence **evidence prioritisation** and **output style**:

### Beginner mode

- Emphasises generalisable evidence
- Shorter, lower jargon answers

### Intermediate mode

- Allows more technical detail and programming nuance
- Longer answers where helpful
- Permits more trained-lifter evidence

Mode is part of the API input and shapes retrieval ranking and the response template

---

## Confidence scoring

Confidence is derived from retrieval behaviour:

- **Strength of the top match**
- **Separation between the top and second match**

These are converted into:

- A numeric **confidence score** (0-100)
- A qualitative label (**low / medium / high**)

---

## Grounding and citations

### Citation rules (product-facing contract)

- Answers include inline numeric citations like `[1] [2]`
- Each citation maps to a real study from the retrieved top-K

### Post-generation citation validation

After generation, a validation step:

- Removes hallucinated/out-of-range citation indices
- Renumbers valid citations sequentially
- Guarantees citations map to real studies in the response payload

---

## Generation backends

The generation layer is modular:

1. **Baseline generator (deterministic / extractive)**

   - Template-driven, selects sentences from top passages

2. **LLM generator (production)**
   - Model: `gpt-3.5-turbo`
   - Typical response time: **~3-8s** depending on query length and retrieved context

Both generators are constrained to retrieved context; the LLM is used for fluency and synthesis while citations remain anchored to retrieved studies
