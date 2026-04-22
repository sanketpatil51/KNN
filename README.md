# Spend Classification System

An ML-powered system that automatically classifies financial transactions (spend records) into categories using semantic similarity and K-Nearest Neighbors (KNN). Built on OpenAI embeddings and FAISS for fast, accurate categorization of supplier transactions at enterprise scale.

---

## How It Works

```
Historic Data (Excel)
        ↓
  Preprocess & Clean
        ↓
  Generate Embeddings  ←  OpenAI text-embedding-3-small
        ↓
  Build FAISS Indexes  ←  Dual index: Description + GL code
        ↓
════════════════════════════
  Test Data (Excel)
        ↓
  Batch Generate Embeddings
        ↓
  Supplier Matching + Search
        ↓
  Radius Filter → Dynamic K → Voting
        ↓
  Conflict Detection & Re-ranking
        ↓
  Confidence Score + Decision Layer
        ↓
  Output Excel (predictions + confidence)
```

**Three-tier decision output:**
| Decision | Confidence | Meaning |
|----------|------------|---------|
| `AUTO` | ≥ 0.85 | High confidence — accept automatically |
| `REVIEW` | 0.70 – 0.85 | Medium confidence — human review recommended |
| `MANUAL` | < 0.70 | Low confidence — manual classification required |

---

## Features

- **Dual-Index Search** — Separate FAISS indexes for descriptions and GL codes
- **Supplier Affinity** — Searches within known suppliers first, then falls back to global search
- **Dynamic K Selection** — Adapts the number of neighbors based on the similarity curve
- **Conflict Detection & Resolution** — Re-ranks neighbors by direct cosine similarity when they disagree
- **Confidence Scoring** — Hybrid metric: vote agreement (60%) + average similarity (40%)
- **Batch Embedding** — Batches 100 records per API call to minimize OpenAI costs
- **Parallel Classification** — Classifies records using 5 concurrent threads

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Embedding Model | OpenAI `text-embedding-3-small` |
| Vector Search | FAISS (HNSW index) |
| Data Processing | Pandas, NumPy |
| ML Algorithm | Weighted KNN with cosine similarity |
| Data Format | Excel `.xlsx` via openpyxl |
| Concurrency | `ThreadPoolExecutor` |

---

## Project Structure

```
Code/
├── main_build_index.py     # Step 1: Build FAISS indexes from historical data
├── main_predict.py         # Step 2: Classify test data and output results
├── requirements.txt        # Python dependencies
├── Historic_data.xlsx      # Training data (supplier, description, gl, category)
├── Test_data.xlsx          # Input data for prediction
└── Src/
    ├── config.py           # API key, file paths, thresholds
    ├── embeddings.py       # OpenAI embedding generation
    ├── preprocessing.py    # Text cleaning and normalization
    ├── index_builder.py    # FAISS index construction and loading
    ├── search.py           # Similarity search logic
    ├── knn_logic.py        # KNN classification algorithms
    ├── classifier.py       # End-to-end classification pipeline
    └── decision.py         # Confidence-based decision routing
```

---

## Setup

### 1. Prerequisites

- Python 3.11+
- An [OpenAI API key](https://platform.openai.com/api-keys) with access to embedding models

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API Key

Open `Src/config.py` and replace the placeholder with your OpenAI API key:

```python
client = OpenAI("sk-your-api-key-here")
```

### 4. Prepare Data Files

**`Historic_data.xlsx`** — Historical records used to build the index. Required columns:

| Column | Description |
|--------|-------------|
| `supplier` | Supplier / vendor name |
| `description` | Transaction description |
| `gl` | General Ledger code |
| `category` | Known category label (ground truth) |

**`Test_data.xlsx`** — Records to be classified. Required columns:

| Column | Description |
|--------|-------------|
| `supplier` | Supplier / vendor name |
| `description` | Transaction description |
| `gl` | General Ledger code |

---

## Usage

### Step 1 — Build the Index (run once)

```bash
python main_build_index.py
```

Reads `Historic_data.xlsx`, generates embeddings, and creates the FAISS indexes. Generates these artifacts:

```
spend_desc_index.faiss   # FAISS index for description search
spend_gl_index.faiss     # FAISS index for GL code search
metadata.pkl             # Historical records with embeddings
supplier_map.pkl         # Supplier → record ID lookup map
```

> Only needs to be re-run when your historical data changes.

### Step 2 — Classify Records

```bash
python main_predict.py
```

Reads `Test_data.xlsx`, classifies every row, and writes `classification_results.xlsx`.

**Output columns:**

| Column | Description |
|--------|-------------|
| `pred_category` | Predicted spend category |
| `pred_confidence_score` | Confidence score (0–1) |
| `pred_decision` | `AUTO`, `REVIEW`, or `MANUAL` |

---

## Configuration

All parameters are in `Src/config.py`:

```python
# Search
TOP_K_SEARCH = 20          # Candidates retrieved from FAISS
MIN_K = 3                  # Minimum neighbors for voting
MAX_K = 7                  # Maximum neighbors for voting

# Thresholds
SIMILARITY_THRESHOLD = 0.65    # Minimum similarity to include a neighbor
AUTO_THRESHOLD = 0.85          # Confidence required for AUTO decision
REVIEW_THRESHOLD = 0.70        # Confidence required for REVIEW (else MANUAL)
```

---

## Requirements

```
pandas
numpy
faiss-cpu
openai
sentence-transformers
openpyxl
```

Install with:

```bash
pip install -r requirements.txt
```

> For GPU-accelerated FAISS on large datasets, replace `faiss-cpu` with `faiss-gpu`.

---

## License

This project is available for personal and commercial use. See [LICENSE](LICENSE) for details.
