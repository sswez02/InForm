# InForm

<p align="center">
  <a href="https://informapp.dev/" target="_blank" rel="noopener noreferrer">
    <img src="public/README.gif" alt="InForm demo" width="900" />
  </a>
</p>

<p align="center">
  🔗 <a href="https://informapp.dev/">https://informapp.dev/</a>
</p>

**Try it:** Ask “How often should I train each muscle group?” and switch between Beginner and Intermediate modes.

**InForm** is an evidence-based AI assistant for training and nutrition.

It combines a modern conversational UI with a **retrieval-augmented backend grounded in peer-reviewed research**. The project is built as a **production-style, full-stack system** to demonstrate fundamentals, clean API design, and product decisions.

---

## Problem & Approach

Most fitness and nutrition apps rely on generic advice or opaque AI outputs.

InForm addresses this by:

- grounding answers in **retrieved research passages**
- adapting explanation depth via **Beginner / Intermediate modes**
- exposing **citations and confidence** to the user

The focus is correctness, clarity, and trust—not just fluent text generation.

---

## Core Capabilities

- Evidence-based Q&A for training, nutrition, recovery, and supplements
- Retrieval-Augmented Generation (RAG) over curated studies
- Mode-aware responses (beginner vs. intermediate)
- Confidence scoring and explicit citations

---

## Docs

- Architecture: `docs/architecture.md`
- Retrieval & grounding: `docs/retrieval.md`
- Evaluation: `docs/evaluation.md`
- Corpus updates: `docs/corpus_update.md`

---

## Technical Overview

### Frontend

- React + TypeScript (Vite)
- Tailwind CSS
- shadcn/ui primitives
- Custom chat, FAQ, and confidence components

### Backend

- FastAPI (Python)
- Research retrieval pipeline
- Pluggable response generation:
  - deterministic baseline
  - LLM-powered generation

---

## System Architecture

1. Client sends an `/ask` request containing:
   - user query
   - experience mode
   - retrieval parameters
2. Backend:
   - retrieves relevant study passages
   - constructs an answer from retrieved evidence
   - computes confidence and citations
3. Client:
   - renders the answer, confidence bar, and references

This separation keeps the system **modular, testable, and extensible**.

---

## Key Design Choices

- **RAG over fine-tuning:** ensures answers stay grounded in evidence and are easier to audit.
- **Explicit confidence scoring:** communicates uncertainty instead of overclaiming.
- **Mode-based explanations:** same facts, different depth, to serve multiple user skill levels.

---

## License

MIT License
