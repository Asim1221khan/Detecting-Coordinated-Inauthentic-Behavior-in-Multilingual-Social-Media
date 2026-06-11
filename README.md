# Detecting Coordinated Inauthentic Influence Operations in Multilingual Social Media

An end-to-end research-grade NLP and Network Anomaly framework designed to detect coordinated inauthentic behavior (CIB) campaigns in multilingual social media environments without relying on fully labeled ground-truth training data.

---

## 🔬 Methodology Overview

This framework utilizes a hybrid pipeline combining multilingual text representations, density clustering, time-synchronization graphs, and consensus anomaly detection:

```
  +------------------+     +------------------------+     +------------------------+
  |  Raw Data Load   | --> | Preprocessing & LangID | --> | Sentence Transformers  |
  +------------------+     +------------------------+     +------------------------+
                                                                      |
                                                                      v
  +------------------+     +------------------------+     +------------------------+
  | Weak Supervision | <-- |  Consensus Anomalies   | <-- | UMAP & HDBSCAN Space   |
  |  (Pseudo-Label)  |     | (Isolation Forest/LOF) |     |  (Semantic Clusters)   |
  +------------------+     +------------------------+     +------------------------+
           |
           v
  +------------------+     +------------------------+     +------------------------+
  |  CTTIN Graph     | --> | Behavior Categorizer   | --> | Streamlit Dashboard    |
  |  (Louvain Comm)  |     |   (Explainable AI)     |     |   & PDF/CSV Export     |
  +------------------+     +------------------------+     +------------------------+
```

1. **Language-Specific Clean & Normalization:** Standardizes Unicode ranges, handles emojis, strips noise, and normalizes scripts (such as Arabic orthography variations).
2. **Dense Semantic Embeddings:** Encodes sentences via `all-mpnet-base-v2` (or LaBSE for high-fidelity cross-lingual alignment).
3. **Semantic Clusters (UMAP + HDBSCAN):** Reduces dimensionality and groups accounts into dense cluster spaces, calculating membership probability profiles.
4. **Graph-based Synchronization (CTTIN):** Maps user nodes onto a Coordinated Text-Time Interaction Network (CTTIN), where edges indicate users posting near-duplicate texts (cosine similarity > 85%) within a 60-second time lag.
5. **Behavioral Outlier Consensus:** Standardizes and runs a consensus unsupervised model (Isolation Forest + Local Outlier Factor) over temporal, network, and behavioral indicators.
6. **Weak Supervision Labeling:** Generates pseudo-labels using a weighted combination of cluster density, graph centrality, and posting synchronization.
7. **Explainability (SHAP):** Trains a surrogate classifier on the pseudo-labels to calculate global feature importances and local account attributions.

---

## 🛠️ Tech Stack & Dependencies

- **Runtime:** Python 3.11
- **Machine Learning & NLP:** `scikit-learn`, `PyTorch`, `transformers`, `sentence-transformers`, `umap-learn`, `hdbscan`, `langdetect`
- **Graph & Math:** `networkx`, `numpy`, `pandas`
- **Visualization:** `plotly`, `matplotlib`, `seaborn`, `streamlit`
- **Explainability & Optimization:** `shap`, `optuna`

---

## 🚀 Installation & Setup

1. **Clone & Open Project:**
   ```bash
   cd "f:/internships/AIIostolas/NLP Project"
   ```

2. **Create and Activate Virtual Environment:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```

3. **Install Requirements:**
   ```bash
   pip install -r requirements.txt
   ```
   *Note: If `hdbscan` fails to compile on Windows, the codebase will automatically fall back to the pre-compiled scikit-learn HDBSCAN or DBSCAN implementations.*

---

## 📈 Execution Guide

The master script `run.py` handles execution in two modes:

### A. Run Streamlit Dashboard (Recommended)
Launch the interactive web application to upload datasets, configure parameters, run pipelines, and explore plots:
```bash
python run.py --streamlit
```

### B. Run Command-Line Interface (CLI)
Execute batch processing directly through the terminal and export CSV result tables:
```bash
python run.py --cli --dataset data/samples/sample_twitter.csv --output reports/results/cib_output.csv
```

---

## 📂 Project Architecture

```
project_root/
│
├── app/
│   ├── streamlit_app.py        # Dashboard main entrypoint
│   ├── dashboard.py            # Overview dashboard charts
│   └── pages/                  # Streamlit Page modules
│       ├── 1_Data_Explorer.py
│       ├── 2_Preprocessing.py
│       ├── 3_Feature_Analysis.py
│       ├── 4_CIB_Detection.py
│       ├── 5_Network_Graph.py
│       ├── 6_Results.py
│       └── 7_Explainability.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── embeddings/
│   ├── networks/
│   └── samples/                # Benchmark multilingual datasets
│
├── src/
│   ├── data/                   # Data loader, validation, schema validator
│   ├── preprocessing/          # Language ID, cleaning, normalization pipelines
│   ├── features/               # Textual, temporal, network, behavioral fusions
│   ├── models/                 # UMAP, HDBSCAN, CTTIN graphs, IFOREST/LOF, SHAP
│   ├── evaluation/             # Metrics, Optuna trials, cross-language evaluations
│   ├── visualization/          # NetworkX layout plots, Plotly widgets
│   └── utils/                  # System configurations, loggers, helpers
│
├── reports/                    # CSVs, markdown tables, and figure graphs
├── requirements.txt
└── run.py                      # Main entrypoint runner
```
