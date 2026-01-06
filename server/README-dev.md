1. Project Structure (High-Level)

project_root/
src/
core/ # Data models, loading logic, utilities
retrieval/ # TF-IDF, dense retrieval, hybrid scorer
ft/ # Fine-tuning pipeline + LLM generation
eval/ # Metrics + citation checking
api/ # Deployment layer (FastAPI)
scripts/
cli/ # Interactive tools: ask, inspect, test
ft/ # FT dataset builders + LoRA trainer
retrieval/ # Indexing and retrieval evaluation
data/ # PDF/CSV ingestion utilities
eval/ # End-to-end evaluation tools

2. src/ - Library Code (Reusable)

## src/core/ - Core data structures & parsing

models.py # Study, Passage dataclasses
store.py # StudyStore: loads studies into memory
load_studies.py # Helpers for ingesting study JSONs
text_utils.py # Tokenization, normalization helpers
logging_utils.py # Interaction logging + JSONL utilities

    Purpose: Everything related to what a study is and how we load it

## src/retrieval/ - Search engine layer

indexer.py # TF-IDF index construction
dense_retriever.py # Sentence-transformer embedding retriever
hybrid_retriever.py # Weighted fusion of lexical+dense scores
retriever.py # Shared Retriever interface

    Purpose: Retrieve relevant study passages for any query

## src/ft/ - Fine-Tuning & LLM Generation

formatting.py # Build prompt templates for SFT + inference
generation.py # DomainLLM wrapper (base + LoRA inference)
answerer.py # Baseline answerer (non-LLM)
answer_generator.py # Utility functions for composing answers
pdf_ingest.py # Optional PDF → Study JSON utilities
dataset.py # (Optional) Helper for constructing FT datasets
train.py # (Optional) One-line wrapper for LoRA training

    Purpose: Format prompts, Generate answers (LLM & baseline), Run LoRA fine-tuning, Handle prompt building + citation mapping

## src/eval/ - Evaluation utilities

citations.py # Citation validity checker
metrics.py # Overlap metrics, length ratio, etc.

    Purpose: Provide reusable evaluation logic for Baseline vs LLM comparisons, Citation faithfulness, Retrieval effectiveness

## src/api/ - Deployment Layer (FastAPI)

server.py # /ask endpoint, loads StudyStore + LLM + retriever

3. scripts/ - Command-Line Tools

## scripts/cli/ - Interactive tools

ask.py # Ask the full system (baseline or LLM)
ask_dense.py # Debug dense retriever only
ask_hybrid.py # Debug hybrid retriever

    Useful for internal testing and debugging

## scripts/ft/ - Dataset prep + LoRA training

build_ft_dataset.py # Extract interactions into SFT rows
clean_ft_dataset.py # Cleaning & validation
split_ft_dataset.py # Train/val split
ft_stats.py # Dataset stats
train_lora.py # Full LoRA training script
test_llm.py # Smoke-test the fine-tuned LLM

    Run LoRA training:

        python -m scripts.ft.train_lora

## scripts/retrieval/ - Indexing & retrieval experiments

build_index.py
index_stats.py
eval_retrieval.py
eval_recall.py
eval_report_dense.py
test_dense_retriever.py
test_hybrid_retriever.py
tune_hybrid_weights.py
search_passages.py

    Used to tune TF-IDF weights, dense model performance, hybrid balancing, etc

## scripts/data/ - PDF / CSV ingestion

import_pdf.py
build_studies_from_csv.py

    These convert raw source data into study JSON format ready for StudyStore

## scripts/eval/ - End-to-end evaluation flows

eval_finetuned_llm.py # LLM vs baseline comparisons
eval_report.py # Summaries and reporting
prepare_human_eval.py # Creates human eval templates
summarise_human_eval.py # Aggregates human ratings

4. Running the Agent

Query the system (baseline or LLM):
python -m scripts.cli.ask --mode beginner "Is creatine safe?"

Run the deployed API:
uvicorn src.api.server:app --reload

Rebuild FT dataset:
python -m scripts.ft.build_ft_dataset
python -m scripts.ft.clean_ft_dataset
python -m scripts.ft.split_ft_dataset

Train LoRA:
python -m scripts.ft.train_lora

5. Testing Retrieval Components

Test dense retriever:
python -m scripts.retrieval.test_dense_retriever "creatine strength"

Test hybrid retriever:
python -m scripts.retrieval.test_hybrid_retriever "hypertrophy trained lifter
