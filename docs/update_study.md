# InForm - Corpus updates

The InForm corpus is designed to be **data-driven**: studies are represented in a master CSV plus per-study JSON files containing curated passages

**Credibility note:** the corpus is stored on-disk on the server; TF-IDF stats and dense embeddings are built in-memory at startup

---

## Data model (what you update)

1. `studies_master.csv`

   - Stable study ID (used for citations)
   - Metadata (title, year, authors, DOI when available)
   - Tags (e.g., topic, training status)
   - Notes / fields used for filtering or mode behaviour

2. Per-study JSON file
   - Metadata
   - `passages`: short text chunks extracted from key sections

Passages should be:

- Short enough to retrieve precisely
- Written exactly as extracted

---

## Recommended workflow to add a new study

1. **Assign a stable study ID**

   - IDs should not change once referenced by citations

2. **Add metadata to `studies_master.csv`**

   - title, year, authors, DOI
   - Tags used by retrieval/mode filtering

3. **Create the per-study JSON**

   - Include structured metadata
   - Add a list of curated passages

4. **Run local checks**

   - Load corpus without errors
   - Run a few representative queries that should retrieve the new study
   - Verify citations map to the new study ID correctly

5. **Run evaluation batch (optional but recommended)**
   - Ensures overall behaviour (length, citations, confidence) stays stable

---

## How the system picks up updates

On startup, the backend:

- Loads the on-disk corpus
- Builds TF-IDF stats in memory
- Builds dense embeddings in memory
- Serves requests using those in-memory indexes

Implication:

- Corpus updates generally require a **redeploy/restart** so the server rebuilds indexes from the updated dataset.

---

## Quality and consistency guidelines

### Passage segmentation

- Prefer 2-6 sentence passages
- Avoid mixing multiple unrelated findings in one passage
- Keep units and conditions explicit (population, duration, intervention)

### Metadata hygiene

- Use consistent tag vocab (avoid near-duplicates)
- Ensure year/DOI are correct where possible
- If a field is unknown, leave it blank rather than guessing

### Avoid overfitting

- Don’t add special-case code for one study
- Prefer improving alias/tagging/passage quality over logic branches

---

## Deployment note (practical)

After updating corpus files:

- Rebuild / restart the backend so indexes are rebuilt at startup
- Then re-run a small smoke test:
  - 2-3 “known answer” questions
  - Confirm citations and confidence behave as expected
