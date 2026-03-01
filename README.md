# SHL Recommender System

## Pipeline Overview
The system implements a two-stage retrieval and reranking pipeline for SHL catalogue assessment recommendations:
1. **Semantic Search (Stage 1):** TF-IDF vectorization with FAISS (FlatL2 indexing). Fetches the top 20 conceptually related assessments.
2. **LLM Reranking (Stage 2):** Passes the Stage 1 candidates to the Gemini LLM to optimize for relevance, diversity spanning distinct areas, unique concepts, and structured output (JSON).

## Folder Structure
```
shl-recommender/
├── app/
│   └── streamlit_app.py           # Frontend user interface
├── data/
│   ├── embeddings/                # FAISS index, Vectorizer, Metadata map
│   ├── evaluation/                # Ground truth datasets and script-generated prediction dumps
│   └── processed/                 # Scraped raw/enhancement JSON datasets
├── scripts/
│   └── generate_submission.py     # Batch prediction CSV generator script
├── src/
│   ├── api/
│   │   └── main.py                # FastAPI endpoints pipeline orchestration
│   ├── crawler/
│   │   ├── clean_dataset.py       # Metadata enhancement via regex
│   │   └── shl_scraper.py         # 2-stage product catalog text scraper 
│   ├── evaluation/
│   │   ├── evaluate.py            # Recall@10 benchmarking calculations
│   │   └── generate_predictions.py# Top-10 output CSV generator
│   ├── reranker/
│   │   └── rerank.py              # Gemini LLM ranking configuration
│   └── retrieval/
│       ├── embedder.py            # Generates TF-IDF sparse > NumPy embeddings
│       ├── search.py              # Executes FAISS and cached vectorization
│       └── vector_store.py        # Initialize/Load FAISS Index
├── tests/
│   └── test_api.py                # API Validation endpoint testing
├── .env                           # GEMINI_API_KEY definitions
└── README.md                      # Pipeline summary
```

## Setup & Dependencies
```bash
# Optional but recommended: Create virtual environment
python -m venv venv
source venv/bin/activate

# Install required packages
pip install requests beautifulsoup4 scikit-learn faiss-cpu numpy pandas fastapi uvicorn streamlit google-generativeai python-dotenv

# Set Environment Variables in `.env`
echo "GEMINI_API_KEY=YOUR_KEY_HERE" > .env
```

## System Commands

**1. Data Crawler & Extractor**
```bash
# Scrape SHL catalog (outputs to data/processed/assessments.json)
python src/crawler/shl_scraper.py

# Clean and extract advanced metadata (outputs to *_clean_enhanced.json)
python src/crawler/clean_dataset.py --input data/processed/assessments.json --output data/processed/assessments_clean_enhanced.json
```

**2. FAISS Retrieval Indexing**
```bash
# Generate Vectorizer and NPY embeddings
python src/retrieval/embedder.py --input data/processed/assessments_clean_enhanced.json

# Build and save FAISS FlatL2 Index
python src/retrieval/vector_store.py
```

**3. API Backend & UI Frontend**
```bash
# Start the FastAPI Server (Port 8000)
uvicorn src.api.main:app --reload

# Start the Streamlit UI (Port 8501)
streamlit run app/streamlit_app.py
```

## Evaluation & Reproduction
Use the predefined pipelines to generate predictions against your holdout/test sets. 

**Generate Prediction Output (Score testing)**:
```bash
# Output top-10 predictions mapped sequentially for scoring (Query, Assessment_url)
python scripts/generate_submission.py --input data/evaluation/test_queries.json --output submission.csv
```

**Run Offline Metrics (Recall@10)**:
```bash
# Computes hit-rate recall across the test batches
python src/evaluation/evaluate.py --ground-truth data/evaluation/ground_truth.json
```

## Metric Explanation (Recall@10)
**Recall@10** measures the percentage of test queries for which the system retrieves at least one expected, relevant assessment URL within its top 10 recommended ranking.
* **Equation:** `(Queries with ≥ 1 Hit at K=10) / (Total Queries)`
* **Interpretation:** Optimizes for surfacing correct domain mappings accurately to the user directly on page one. The integration of TF-IDF bounds scope, while Gemini boosts precision routing to prevent exact-string miss rates.
