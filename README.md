# 🛍️ Shopper Spectrum

> **AI-powered customer segmentation & product recommendation for modern e-commerce**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-≥1.4-orange.svg)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

Shopper Spectrum is a production-ready, end-to-end Machine Learning and Data Analytics application that analyses customer transaction data from an online retail business.

**What it does:**
- 🎯 **Customer Segmentation** — RFM Analysis + K-Means Clustering
- 🛍️ **Product Recommendations** — Item-Based Collaborative Filtering (Cosine Similarity)
- 📊 **Business Analytics** — Interactive KPI dashboard, EDA, and trend charts
- 🤖 **AI Business Assistant** — Gemini-powered chatbot with automatic tool calling
- 🔒 **Role-Based Access Control** — Sales Staff, Analyst, Administrator roles
- 🔗 **MCP Server** — Standalone Model Context Protocol server for agent integrations

---

## Live Demo

Deploy this app yourself on [Streamlit Community Cloud](https://share.streamlit.io/) — see deployment instructions below.

---

## Features

### Customer Segments
| Segment | Description |
|---------|-------------|
| 🏆 High-Value | Best customers — frequent, recent, high spend |
| ✅ Regular | Reliable repeat customers |
| 🛒 Occasional | Infrequent buyers with potential |
| ⚠️ At-Risk | Previously active — now going quiet |
| 👁️ Needs Attention | Low engagement across all RFM dimensions |
| 🆕 New Buyer | Recently acquired customers |

### AI Business Assistant Capabilities
- Ask natural language questions about customers, products, and business metrics
- **Automatic Function Calling**: directly queries live pandas DataFrames via 4 tools:
  - `get_store_overview_metrics` — revenue, customers, transactions
  - `get_customer_profile` — per-customer RFM + segment info
  - `get_segment_info` — aggregate segment statistics
  - `get_similar_products` — collaborative-filtering recommendations
- **RBAC Security**: different data access per role — no raw credential exposure

---

## Project Structure

```
shopper-spectrum/
├── app.py                   # Main Streamlit application
├── mcp_server.py            # MCP server for agent integrations
├── requirements.txt         # Python dependencies
├── runtime.txt              # Python version pin (deployment)
├── packages.txt             # System packages (deployment platforms)
├── LICENSE                  # MIT License
├── README.md                # This file
├── .gitignore               # Git ignore rules
├── .streamlit/
│   ├── config.toml          # Streamlit theme & server config
│   └── secrets.toml         # API keys (git-ignored — DO NOT COMMIT)
├── models/                  # Saved ML models (auto-generated, git-ignored)
│   ├── kmeans_model.pkl     # git-ignored — regenerated on startup
│   ├── rfm_scaler.pkl       # git-ignored — regenerated on startup
│   └── product_similarity.pkl  # git-ignored — 35 MB, rebuilt automatically
├── outputs/                 # Generated segment exports (git-ignored)
│   └── customer_segments.csv
└── data/                    # Place your dataset here (git-ignored)
    └── online_retail.csv
```

> **Note:** All `.pkl` model files are listed in `.gitignore`. They are regenerated
> automatically the first time the app runs. The `product_similarity.pkl` file is
> ~35 MB and would exceed GitHub’s per-file limit — it must not be committed.

---

## Quick Start (Local)

```bash
# 1. Clone the repository
git clone https://github.com/sumitswami25372-alt/shoppersetup.git
cd shoppersetup

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Add your Gemini API key
echo 'GEMINI_API_KEY = "your_key_here"' > .streamlit/secrets.toml

# 5. Run the app
streamlit run app.py
```

---

## Deployment

### Streamlit Community Cloud (Recommended — Free)

1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io/) → **New App**.
3. Select repository, branch `main`, file `app.py`.
4. Click **Advanced settings** → **Secrets**, and add:
   ```toml
   GEMINI_API_KEY = "your_gemini_api_key_here"
   ```
5. Click **Deploy**. Done!

> The app includes built-in demo data so it works immediately — no dataset upload needed.

### Render / Railway

1. Set the build command: `pip install -r requirements.txt`
2. Set the start command: `streamlit run app.py --server.port $PORT --server.headless true`
3. Add `GEMINI_API_KEY` as an environment variable in the dashboard.

### Hugging Face Spaces

1. Create a new Space → **Streamlit** template.
2. Upload all project files.
3. Add `GEMINI_API_KEY` to the Space Secrets panel.

---

## API Key Security

The app resolves the Gemini API key in priority order:

1. `GEMINI_API_KEY` **environment variable** (production/CI)
2. `GEMINI_API_KEY` in `.streamlit/secrets.toml` (local dev — git-ignored)
3. Masked **sidebar text input** as a user-provided fallback

The key is **never logged, stored, or transmitted** beyond Gemini's API.

---

## MCP Server

A standalone [Model Context Protocol](https://modelcontextprotocol.io/) server is included for connecting Shopper Spectrum data to AI agent environments (e.g., Claude Desktop, Cursor).

```bash
# Run the MCP server
python mcp_server.py

# Or using fastmcp
fastmcp run mcp_server.py
```

Exposed tools:
- `get_customer_details(customer_id)` — segment + RFM values
- `get_product_recommendations(product_name, limit)` — similar product suggestions
- `get_segment_metrics()` — full segment distribution summary

---

## Dataset

The app works with the [Online Retail dataset](https://archive.ics.uci.edu/dataset/352/online+retail) (UCI Machine Learning Repository).

Place the CSV at `data/online_retail.csv` or upload it via the sidebar. If no file is provided, the app uses built-in synthetic demo data.

**Required columns:**
`InvoiceNo`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `UnitPrice`, `CustomerID`, `Country`

---

## Machine Learning

| Component | Algorithm |
|-----------|-----------|
| Feature Engineering | RFM (Recency, Frequency, Monetary) |
| Preprocessing | Log-transform + StandardScaler |
| Clustering | K-Means (configurable k=2–6) |
| Evaluation | Elbow method + Silhouette score |
| Recommendations | Item-based Cosine Similarity |

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| App Framework | Streamlit ≥ 1.35 |
| Data Processing | Pandas ≥ 2.0, NumPy ≥ 1.26 |
| Machine Learning | Scikit-learn ≥ 1.4 |
| AI / LLM | Google Gemini (`google-genai` SDK) |
| Agent Protocol | MCP Python SDK |
| Persistence | Joblib |
| Language | Python 3.11 |

---

## Author

**Sumit Swami**  
Data Analytics & Machine Learning Project  
[GitHub](https://github.com/sumitswami25372-alt) | [Streamlit Community](https://share.streamlit.io/)

---

## License

This project is licensed under the [MIT License](LICENSE).