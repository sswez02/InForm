# InForm - Architecture

InForm is an evidence-grounded Q&A system for training, supplementation, and nutrition. It answers **only using a curated on-disk corpus** of studies, and returns **inline numeric citations** plus a **retrieval-based confidence** score

**Note:** the corpus is stored on-disk on the server; **TF-IDF stats and dense embeddings are built in-memory at startup**

---

## High-level system diagram

```mermaid
flowchart LR
  subgraph Client["Client"]
    U["User (Browser)"]
    FE["Frontend<br/>Vite + React + TypeScript<br/>Mode toggle: Beginner / Intermediate<br/>Shows answer + citations + confidence"]
    U --> FE
  end

  subgraph Cloud["Hosting"]
    direction LR

    subgraph Vercel["Vercel"]
      FEDEP["Frontend deployment"]
    end

    subgraph AWS["AWS Elastic Beanstalk"]
      API["FastAPI backend<br/>POST /ask"]
      GEN["Generation<br/>Baseline generator OR OpenAI Chat Completions"]
      RET["Retrieval<br/>TF-IDF + Dense (SentenceTransformer)"]
      CONF["Confidence scoring<br/>0-100 + label"]
      CITE["Citation validation<br/>remove invalid, renumber sequentially"]
      DATA["On-disk corpus<br/>studies_master.csv + per-study JSON passages"]
      IDX["In-memory indexes built at startup<br/>TF-IDF stats + dense embeddings"]

      API --> RET
      RET --> CONF
      RET --> GEN
      GEN --> CITE
      DATA --> IDX
      IDX --> RET
    end
  end

  FE -->|HTTPS| API
  API -->|JSON response| FE
```

## Request path (runtime)

```mermaid
sequenceDiagram
  autonumber
  participant User as "User"
  participant FE as "Frontend (Vite + React)"
  participant API as "FastAPI (POST /ask)"
  participant RET as "Hybrid retrieval (TF-IDF + Dense)"
  participant GEN as "Generator (baseline or gpt-3.5-turbo)"
  participant CITE as "Citation validation"
  participant CONF as "Confidence scoring"

  User->>FE: Enter question + select mode
  FE->>API: POST /ask {question, mode, retrieval_params}
  API->>RET: Retrieve top-K passages + study IDs
  RET-->>CONF: Retrieval scores + separation
  CONF-->>API: confidence_score + label
  API->>GEN: Generate answer constrained to retrieved context
  GEN-->>CITE: Proposed citations (may include extras)
  CITE-->>API: Filter invalid citations + renumber
  API-->>FE: {answer, citations, studies, confidence, generator}
  FE-->>User: Render answer + inline citations + confidence
```

## Startup path (index build)

```mermaid
sequenceDiagram
  autonumber
  participant App as "FastAPI startup"
  participant Disk as "On-disk corpus"
  participant TF as "TF-IDF builder"
  participant Dense as "Dense embedder"
  participant Mem as "In-memory indexes"

  App->>Disk: Load studies_master.csv + JSON passages
  App->>TF: Fit TF-IDF stats over passages
  App->>Dense: Build dense embeddings over passages
  TF-->>Mem: TF-IDF matrix + vocabulary
  Dense-->>Mem: Embedding matrix
  App-->>App: Ready to serve /ask
```
