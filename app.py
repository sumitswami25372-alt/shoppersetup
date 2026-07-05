"""Shopper Spectrum — Production-ready e-commerce analytics dashboard.

Architecture
------------
All business logic lives in pure, module-level functions so that static
analysis tools (Pyrefly, Pyright, mypy) can fully resolve every name
without encountering orphaned code fragments.

Pages
-----
Home | Customer Segmentation | Product Recommendation |
Project Insights | Dataset | AI Assistant
"""
from __future__ import annotations

# ── Standard library ──────────────────────────────────────────────────────────
import io
import os
import warnings
from pathlib import Path
from typing import Optional

# ── Third-party ───────────────────────────────────────────────────────────────
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from google import genai
from google.genai import types
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Page config must be first Streamlit call ──────────────────────────────────
st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/sumitswami25372-alt/shoppersetup",
        "About": "Shopper Spectrum — AI-powered e-commerce analytics by Sumit Swami",
    },
)

# ── Constants ─────────────────────────────────────────────────────────────────
REQUIRED_COLUMNS: list[str] = [
    "InvoiceNo", "StockCode", "Description", "Quantity",
    "InvoiceDate", "UnitPrice", "CustomerID", "Country",
]

MODEL_PATHS: dict[str, Path] = {
    "kmeans": Path("models/kmeans_model.pkl"),
    "scaler": Path("models/rfm_scaler.pkl"),
    "similarity": Path("models/product_similarity.pkl"),
}

SEGMENT_LABELS: list[str] = [
    "High-Value", "Regular", "Occasional", "At-Risk", "Needs Attention", "New Buyer",
]

PRODUCT_CATALOG: list[tuple[str, str, str, float]] = [
    ("SKU1001", "WHITE HANGING HEART T-LIGHT HOLDER", "Home Decor", 2.95),
    ("SKU1002", "REGENCY CAKESTAND 3 TIER", "Kitchen", 12.75),
    ("SKU1003", "JUMBO BAG RED RETROSPOT", "Bags", 2.08),
    ("SKU1004", "ASSORTED COLOUR BIRD ORNAMENT", "Home Decor", 1.69),
    ("SKU1005", "LUNCH BAG RED RETROSPOT", "Bags", 1.65),
    ("SKU1006", "PACK OF 72 RETROSPOT CAKE CASES", "Kitchen", 0.55),
    ("SKU1007", "SET OF 3 CAKE TINS PANTRY DESIGN", "Kitchen", 4.95),
    ("SKU1008", "PAPER CHAIN KIT 50S CHRISTMAS", "Seasonal", 2.95),
    ("SKU1009", "NATURAL SLATE HEART CHALKBOARD", "Home Decor", 2.95),
    ("SKU1010", "HEART OF WICKER SMALL", "Home Decor", 1.65),
    ("SKU1011", "VICTORIAN GLASS HANGING T-LIGHT", "Home Decor", 1.25),
    ("SKU1012", "RABBIT NIGHT LIGHT", "Kids", 2.08),
    ("SKU1013", "WOODEN PICTURE FRAME WHITE FINISH", "Home Decor", 2.55),
    ("SKU1014", "SET OF 6 SPICE TINS PANTRY DESIGN", "Kitchen", 3.95),
    ("SKU1015", "RED RETROSPOT CHARLOTTE BAG", "Bags", 0.85),
    ("SKU1016", "ALARM CLOCK BAKELIKE RED", "Gifts", 3.75),
    ("SKU1017", "HOT WATER BOTTLE KEEP CALM", "Wellness", 4.65),
    ("SKU1018", "SET OF 4 PANTRY JELLY MOULDS", "Kitchen", 1.25),
    ("SKU1019", "GARDENERS KNEELING PAD KEEP CALM", "Garden", 1.65),
    ("SKU1020", "WOOD BLACK BOARD ANT WHITE FINISH", "Home Decor", 6.45),
    ("SKU1021", "VINTAGE SNAP CARDS", "Kids", 0.85),
    ("SKU1022", "BAKING SET 9 PIECE RETROSPOT", "Kitchen", 4.95),
    ("SKU1023", "CHARLOTTE BAG SUKI DESIGN", "Bags", 0.85),
    ("SKU1024", "PINK BLUE FELT CRAFT TRINKET BOX", "Kids", 1.25),
    ("SKU1025", "CERAMIC STRAWBERRY CAKE MONEY BANK", "Gifts", 1.45),
    ("SKU1026", "HAND WARMER UNION JACK", "Wellness", 2.10),
    ("SKU1027", "VINTAGE DOILY TRAVEL SEWING KIT", "Gifts", 1.95),
    ("SKU1028", "SET OF 12 MINI LOAF BAKING CASES", "Kitchen", 0.83),
    ("SKU1029", "FELTCRAFT CUSHION RABBIT", "Kids", 3.75),
    ("SKU1030", "RETROSPOT TEA SET CERAMIC 11 PC", "Kitchen", 4.95),
]

CATEGORY_WEIGHTS: dict[str, float] = {
    "Home Decor": 1.25, "Kitchen": 1.2, "Bags": 1.05, "Kids": 0.9,
    "Gifts": 0.82, "Wellness": 0.7, "Garden": 0.55, "Seasonal": 0.65,
}

SEGMENT_STRATEGIES: dict[str, str] = {
    "High-Value": "Offer VIP benefits, early access, premium bundles, and loyalty rewards.",
    "Regular": "Send personalized offers and cross-sell products related to previous purchases.",
    "Occasional": "Use seasonal discounts and reminder campaigns to increase repeat purchases.",
    "At-Risk": "Trigger win-back campaigns, limited-time coupons, and feedback collection.",
    "Needs Attention": "Monitor purchase behavior and test targeted product offers.",
    "New Buyer": "Send onboarding recommendations and first-repeat-purchase incentives.",
}

ROLES: list[str] = ["Sales Staff", "Business Analyst", "Administrator"]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CSS — Premium dark-gradient glassmorphism theme
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def add_css() -> None:
    """Inject premium custom CSS with Inter font, gradient hero, and card styles."""
    st.markdown(
        """
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
        <style>
        /* ── Global typography ── */
        html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

        /* ── Layout ── */
        .main .block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1200px; }

        /* ── Hero banner ── */
        .hero {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 45%, #0f2460 100%);
            border: 1px solid rgba(99, 102, 241, 0.3);
            border-radius: 16px;
            padding: 2rem 2.4rem;
            margin-bottom: 1.5rem;
            position: relative;
            overflow: hidden;
        }
        .hero::before {
            content: '';
            position: absolute; inset: 0;
            background: radial-gradient(ellipse at 70% 50%, rgba(99, 102, 241, 0.15) 0%, transparent 60%);
            pointer-events: none;
        }
        .hero h1 {
            font-size: 2.4rem; font-weight: 800; margin: 0 0 .4rem 0;
            background: linear-gradient(90deg, #e0e7ff, #a5b4fc, #818cf8);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .hero p { margin: 0; color: #94a3b8; font-size: 1.05rem; font-weight: 400; }
        .hero .badge {
            display: inline-block; margin-top: .75rem;
            background: rgba(99,102,241,.2); border: 1px solid rgba(99,102,241,.4);
            color: #a5b4fc; padding: .2rem .75rem; border-radius: 20px; font-size: .8rem; font-weight: 500;
        }

        /* ── KPI metric cards ── */
        .kpi-grid { display: flex; gap: 1rem; margin-bottom: 1.5rem; flex-wrap: wrap; }
        .kpi-card {
            flex: 1; min-width: 140px;
            background: linear-gradient(135deg, #1e293b, #0f172a);
            border: 1px solid rgba(99,102,241,.25); border-radius: 12px;
            padding: 1rem 1.2rem; text-align: center;
            transition: transform .2s, border-color .2s;
        }
        .kpi-card:hover { transform: translateY(-2px); border-color: rgba(99,102,241,.6); }
        .kpi-label { font-size: .75rem; font-weight: 500; color: #64748b; text-transform: uppercase; letter-spacing: .08em; margin-bottom: .25rem; }
        .kpi-value { font-size: 1.6rem; font-weight: 700; color: #e0e7ff; }
        .kpi-icon { font-size: 1.1rem; margin-bottom: .3rem; }

        /* ── Section headers ── */
        .section-title {
            font-size: 1.2rem; font-weight: 700; color: #e2e8f0;
            border-left: 4px solid #6366f1; padding-left: .75rem; margin: 1.5rem 0 1rem 0;
        }

        /* ── Segment prediction card ── */
        .segment-card {
            border: 1px solid rgba(99,102,241,.3); border-radius: 12px;
            padding: 1.2rem; background: linear-gradient(135deg, #1e293b, #172033);
            margin-top: .75rem;
        }
        .segment-card h3 { margin: 0 0 .4rem 0; color: #a5b4fc; font-size: 1.3rem; font-weight: 700; }
        .segment-card .cluster-badge {
            display: inline-block; background: rgba(99,102,241,.2);
            color: #818cf8; border: 1px solid rgba(99,102,241,.3);
            padding: .15rem .6rem; border-radius: 20px; font-size: .8rem; margin-bottom: .5rem;
        }
        .segment-card p { margin: .25rem 0; color: #94a3b8; font-size: .95rem; }
        .strategy-tip { color: #6ee7b7 !important; font-style: italic; }

        /* ── Recommendation cards ── */
        .recommend-card {
            border: 1px solid rgba(99,102,241,.2); border-left: 4px solid #6366f1;
            border-radius: 10px; padding: .9rem 1.1rem; margin-bottom: .65rem;
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            transition: transform .15s, border-left-color .15s;
        }
        .recommend-card:hover { transform: translateX(4px); border-left-color: #818cf8; }
        .rec-rank { font-size: .8rem; font-weight: 600; color: #818cf8; }
        .rec-name { font-weight: 600; color: #e2e8f0; font-size: .95rem; }
        .rec-score { font-size: .8rem; color: #64748b; margin-top: .2rem; }
        .rec-bar { height: 4px; border-radius: 2px; margin-top: .4rem;
                   background: linear-gradient(90deg, #6366f1, #a78bfa); }

        /* ── Insight / benefit cards ── */
        .benefit-card {
            background: linear-gradient(135deg, #1e293b, #0f2460);
            border: 1px solid rgba(99,102,241,.2); border-radius: 12px;
            padding: 1.2rem; height: 100%; transition: transform .2s;
        }
        .benefit-card:hover { transform: translateY(-3px); }
        .benefit-card h4 { color: #a5b4fc; font-weight: 700; margin: 0 0 .5rem 0; }
        .benefit-card p { color: #94a3b8; font-size: .9rem; margin: 0; }
        .benefit-icon { font-size: 1.8rem; margin-bottom: .5rem; }

        /* ── AI chat area ── */
        .chat-header {
            background: linear-gradient(135deg, #1e293b, #0f172a);
            border: 1px solid rgba(99,102,241,.3); border-radius: 12px;
            padding: 1rem 1.4rem; margin-bottom: 1rem;
        }
        .chat-header h3 { color: #a5b4fc; margin: 0; font-size: 1.1rem; font-weight: 700; }
        .chat-header p { color: #64748b; margin: .25rem 0 0 0; font-size: .85rem; }

        /* ── Status pills ── */
        .role-pill {
            display: inline-block; padding: .2rem .8rem; border-radius: 20px;
            font-size: .8rem; font-weight: 600; margin-top: .5rem;
        }
        .role-admin { background: rgba(16,185,129,.15); color: #6ee7b7; border: 1px solid rgba(16,185,129,.3); }
        .role-analyst { background: rgba(99,102,241,.15); color: #a5b4fc; border: 1px solid rgba(99,102,241,.3); }
        .role-staff { background: rgba(245,158,11,.15); color: #fcd34d; border: 1px solid rgba(245,158,11,.3); }

        /* ── Sidebar tweaks ── */
        section[data-testid="stSidebar"] { background: #0f172a; }
        section[data-testid="stSidebar"] .stRadio label { color: #cbd5e1 !important; }
        section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span { color: #94a3b8; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Data generation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate_sample_transactions(rows: int = 4500, seed: int = 42) -> pd.DataFrame:
    """Generate realistic synthetic e-commerce transaction data for demo purposes."""
    rng = np.random.default_rng(seed)
    customers = np.arange(10001, 10321)
    countries = np.array([
        "United Kingdom", "Germany", "France", "Spain",
        "Netherlands", "India", "Australia",
    ])
    country_prob = np.array([0.58, 0.11, 0.10, 0.07, 0.05, 0.05, 0.04])

    catalog = pd.DataFrame(
        PRODUCT_CATALOG, columns=["StockCode", "Description", "Category", "BasePrice"]
    )
    weights = catalog["Category"].map(CATEGORY_WEIGHTS).to_numpy()
    weights = weights / weights.sum()

    start_date = np.datetime64("2022-01-01")
    dates = start_date + rng.integers(0, 730, size=rows).astype("timedelta64[D]")
    times = rng.integers(8 * 60, 22 * 60, size=rows).astype("timedelta64[m]")
    picked = catalog.iloc[
        rng.choice(len(catalog), size=rows, p=weights)
    ].reset_index(drop=True)

    quantities: np.ndarray = rng.poisson(lam=3.2, size=rows) + 1
    bulk_mask: np.ndarray = rng.random(rows) < 0.08
    # Cast to int to avoid NumPy >=1.25 type-mismatch in rng.integers
    bulk_count = int(bulk_mask.sum())
    if bulk_count > 0:
        quantities[bulk_mask] += rng.integers(8, 35, size=bulk_count)

    prices = picked["BasePrice"].to_numpy() * rng.normal(1.0, 0.08, size=rows)
    prices = np.clip(np.round(prices, 2), 0.2, None)

    df = pd.DataFrame({
        "InvoiceNo": (536000 + rng.integers(0, rows // 2, size=rows)).astype(str),
        "StockCode": picked["StockCode"].values,
        "Description": picked["Description"].values,
        "Quantity": quantities,
        "InvoiceDate": pd.to_datetime((dates + times).astype("datetime64[ns]")),
        "UnitPrice": prices,
        "CustomerID": rng.choice(customers, size=rows),
        "Country": rng.choice(countries, size=rows, p=country_prob),
    })

    # Simulate cancellations (~1.5%) and missing customer IDs (~2%)
    cancelled_idx = df.sample(frac=0.015, random_state=seed).index
    df.loc[cancelled_idx, "InvoiceNo"] = "C" + df.loc[cancelled_idx, "InvoiceNo"].astype(str)
    missing_idx = df.sample(frac=0.02, random_state=seed + 1).index
    df.loc[missing_idx, "CustomerID"] = np.nan
    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Data loading helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _read_uploaded_csv(csv_bytes: bytes) -> pd.DataFrame:
    """Try common encodings when reading an uploaded CSV file."""
    for encoding in ("utf-8", "latin1", "cp1252"):
        try:
            return pd.read_csv(io.BytesIO(csv_bytes), encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(io.BytesIO(csv_bytes))  # last-resort default


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to match required schema (case- and whitespace-insensitive)."""
    normalized: dict[str, object] = {
        str(col).strip().lower(): col for col in df.columns
    }
    rename_map: dict[object, str] = {}
    for required in REQUIRED_COLUMNS:
        found = normalized.get(required.lower())
        if found is not None:
            rename_map[found] = required
    return df.rename(columns=rename_map)


def load_transactions(csv_bytes: Optional[bytes]) -> pd.DataFrame:
    """Load transaction data from upload, local file, or synthetic fallback."""
    if csv_bytes is not None:
        df = _normalize_columns(_read_uploaded_csv(csv_bytes))
        missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            st.error(f"CSV is missing required columns: {', '.join(missing)}")
            st.stop()
        return df

    # Search for a local dataset (common path patterns)
    for path in [
        Path("data/data/online_retail.csv"),
        Path("data/OnlineRetail.csv"),
        Path("data/online_retail.csv"),
        Path("online_retail.csv"),
        Path("OnlineRetail.csv"),
    ]:
        if path.exists():
            try:
                return _normalize_columns(pd.read_csv(path, encoding="latin1"))
            except Exception:
                pass  # fall through to next candidate

    # Use demo data as a last resort
    return generate_sample_transactions()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Cleaning & feature engineering
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Apply standard cleaning steps: dedup, remove cancellations, cast types."""
    cleaned = df.copy().drop_duplicates()
    cleaned = cleaned.dropna(subset=["CustomerID"])
    cleaned["InvoiceNo"] = cleaned["InvoiceNo"].astype(str)
    cleaned = cleaned[~cleaned["InvoiceNo"].str.startswith("C", na=False)]
    cleaned["Quantity"] = pd.to_numeric(cleaned["Quantity"], errors="coerce")
    cleaned["UnitPrice"] = pd.to_numeric(cleaned["UnitPrice"], errors="coerce")
    cleaned = cleaned[(cleaned["Quantity"] > 0) & (cleaned["UnitPrice"] > 0)]
    cleaned["InvoiceDate"] = pd.to_datetime(cleaned["InvoiceDate"], errors="coerce")
    cleaned = cleaned.dropna(subset=["InvoiceDate", "Description", "StockCode"])
    cleaned["CustomerID"] = cleaned["CustomerID"].astype(float).astype(int).astype(str)
    cleaned["Description"] = cleaned["Description"].astype(str).str.strip().str.upper()
    cleaned["TotalAmount"] = (cleaned["Quantity"] * cleaned["UnitPrice"]).round(2)
    return cleaned.reset_index(drop=True)


def build_rfm(cleaned: pd.DataFrame) -> pd.DataFrame:
    """Compute Recency, Frequency, and Monetary values per customer."""
    snapshot_date = cleaned["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = (
        cleaned.groupby("CustomerID")
        .agg(
            Recency=("InvoiceDate", lambda v: (snapshot_date - v.max()).days),
            Frequency=("InvoiceNo", "nunique"),
            Monetary=("TotalAmount", "sum"),
        )
        .reset_index()
    )
    rfm["Monetary"] = rfm["Monetary"].round(2)
    return rfm


def transform_rfm(rfm: pd.DataFrame) -> pd.DataFrame:
    """Log-transform RFM columns to reduce skewness before scaling."""
    transformed = rfm[["Recency", "Frequency", "Monetary"]].copy()
    for col in ("Recency", "Frequency", "Monetary"):
        transformed[col] = np.log1p(transformed[col].clip(lower=0))
    return transformed


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Clustering
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def label_segments(clustered_rfm: pd.DataFrame) -> pd.Series:
    """Map cluster numbers to human-readable segment names by RFM score ranking."""
    profiles = (
        clustered_rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary"]]
        .mean()
        .assign(
            RecencyRank=lambda d: d["Recency"].rank(ascending=True),
            FrequencyRank=lambda d: d["Frequency"].rank(ascending=False),
            MonetaryRank=lambda d: d["Monetary"].rank(ascending=False),
        )
    )
    profiles["Score"] = (
        profiles["RecencyRank"] + profiles["FrequencyRank"] + profiles["MonetaryRank"]
    )
    ordered_clusters = profiles.sort_values("Score").index.tolist()
    label_map: dict[int, str] = {
        cluster: (SEGMENT_LABELS[i] if i < len(SEGMENT_LABELS) else f"Cluster {cluster}")
        for i, cluster in enumerate(ordered_clusters)
    }
    return clustered_rfm["Cluster"].map(label_map)


def fit_kmeans(
    rfm: pd.DataFrame, cluster_count: int
) -> tuple[pd.DataFrame, StandardScaler, KMeans, np.ndarray]:
    """Fit K-Means on scaled RFM data; returns clustered DataFrame, scaler, model, scaled X."""
    cluster_count = max(2, min(cluster_count, len(rfm)))
    scaler = StandardScaler()
    x_scaled: np.ndarray = scaler.fit_transform(transform_rfm(rfm))
    model = KMeans(n_clusters=cluster_count, n_init=30, random_state=42)
    clustered = rfm.copy()
    clustered["Cluster"] = model.fit_predict(x_scaled)
    clustered["Segment"] = label_segments(clustered)
    return clustered, scaler, model, x_scaled


def evaluate_clusters(rfm: pd.DataFrame, max_k: int = 8) -> pd.DataFrame:
    """Compute elbow (inertia) and silhouette scores for k=2…max_k."""
    if len(rfm) < 3:
        return pd.DataFrame(columns=["k", "inertia", "silhouette"])
    scaler = StandardScaler()
    x_scaled: np.ndarray = scaler.fit_transform(transform_rfm(rfm))
    rows: list[dict[str, object]] = []
    for k in range(2, min(max_k, len(rfm) - 1) + 1):
        mdl = KMeans(n_clusters=k, n_init=20, random_state=42)
        labels = mdl.fit_predict(x_scaled)
        rows.append({
            "k": k,
            "inertia": round(float(mdl.inertia_), 2),
            "silhouette": round(float(silhouette_score(x_scaled, labels)), 4),
        })
    return pd.DataFrame(rows)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Product similarity
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_similarity(cleaned: pd.DataFrame) -> pd.DataFrame:
    """Build item-based cosine similarity matrix from customer-product purchase counts."""
    top_products = cleaned["Description"].value_counts().head(2000).index
    filtered = cleaned[cleaned["Description"].isin(top_products)]
    product_matrix = filtered.pivot_table(
        index="CustomerID", columns="Description",
        values="Quantity", aggfunc="sum", fill_value=0,
    ).astype("float32")
    sim = cosine_similarity(product_matrix.T)
    return pd.DataFrame(
        sim, index=product_matrix.columns, columns=product_matrix.columns, dtype="float32"
    )


def recommend_products(
    product_name: str, similarity_df: pd.DataFrame, top_n: int
) -> tuple[Optional[str], pd.Series]:
    """Return (matched_name, top_n similarity scores) for the given product keyword."""
    product_name = product_name.strip().upper()
    if not product_name:
        return None, pd.Series(dtype=float)

    matched: str = product_name
    if matched not in similarity_df.index:
        products = similarity_df.index.to_series()
        matches = products[products.str.contains(product_name, case=False, regex=False)]
        if matches.empty:
            return None, pd.Series(dtype=float)
        matched = str(matches.iloc[0])

    recs = (
        similarity_df[matched]
        .drop(index=matched)
        .sort_values(ascending=False)
        .head(top_n)
    )
    return matched, recs


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Model persistence
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def save_models(
    model: KMeans, scaler: StandardScaler, similarity_df: pd.DataFrame
) -> None:
    """Persist trained models to disk under models/."""
    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATHS["kmeans"])
    joblib.dump(scaler, MODEL_PATHS["scaler"])
    joblib.dump(similarity_df, MODEL_PATHS["similarity"])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Business logic helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def segment_strategy(segment: str) -> str:
    """Return a one-sentence marketing action for the given segment."""
    return SEGMENT_STRATEGIES.get(
        segment, "Review this customer group and design a targeted campaign."
    )


def predict_segment(
    recency: float,
    frequency: float,
    monetary: float,
    scaler: StandardScaler,
    model: KMeans,
    clustered_rfm: pd.DataFrame,
) -> tuple[int, str]:
    """Predict the cluster and segment label for a single customer's RFM values."""
    input_df = pd.DataFrame([{
        "Recency": recency, "Frequency": frequency, "Monetary": monetary
    }])
    scaled = scaler.transform(transform_rfm(input_df))
    cluster = int(model.predict(scaled)[0])
    modes = clustered_rfm.loc[clustered_rfm["Cluster"] == cluster, "Segment"].mode()
    return cluster, str(modes.iloc[0]) if not modes.empty else "Regular"


def summarize_segments(clustered_rfm: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-segment customer counts and average RFM metrics."""
    return (
        clustered_rfm.groupby("Segment")
        .agg(
            Customers=("CustomerID", "count"),
            AvgRecency=("Recency", "mean"),
            AvgFrequency=("Frequency", "mean"),
            AvgMonetary=("Monetary", "mean"),
        )
        .round(2)
        .sort_values("AvgMonetary", ascending=False)
        .reset_index()
    )


def build_customer_view(
    cleaned: pd.DataFrame, clustered_rfm: pd.DataFrame
) -> pd.DataFrame:
    """Enrich RFM clusters with last-purchase date and country columns."""
    last_purchase = (
        cleaned.groupby("CustomerID")["InvoiceDate"].max().dt.date.rename("LastPurchase")
    )
    country = (
        cleaned.groupby("CustomerID")["Country"]
        .agg(lambda v: v.mode().iloc[0])
        .rename("Country")
    )
    return clustered_rfm.join(last_purchase, on="CustomerID").join(country, on="CustomerID")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Cached data loaders (Streamlit cache decorators)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@st.cache_data(show_spinner="⏳ Loading and cleaning transactions…")
def load_clean_data(
    csv_bytes: Optional[bytes],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load → clean → validate. Cached on the hash of csv_bytes."""
    raw = load_transactions(csv_bytes)
    cleaned = clean_transactions(raw)
    if cleaned.empty or cleaned["CustomerID"].nunique() < 2:
        st.error("Not enough valid customer transactions after cleaning.")
        st.stop()
    return raw, cleaned


@st.cache_data(show_spinner="⏳ Building product recommendation matrix…")
def get_similarity_matrix(
    cleaned_df: pd.DataFrame, use_saved: bool = True
) -> pd.DataFrame:
    """Load similarity matrix from disk if available; otherwise recompute."""
    if use_saved and MODEL_PATHS["similarity"].exists():
        try:
            return joblib.load(MODEL_PATHS["similarity"])  # type: ignore[return-value]
        except Exception:
            pass
    return build_similarity(cleaned_df)


@st.cache_resource(show_spinner="⏳ Clustering customers with K-Means…")
def get_rfm_clustering(
    cleaned_df: pd.DataFrame, cluster_count: int, use_saved: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler, KMeans, pd.DataFrame]:
    """Load or train K-Means clustering; return (rfm, clustered, customer_view, scaler, model, eval)."""
    rfm = build_rfm(cleaned_df)

    # Try loading pre-trained models
    if use_saved and MODEL_PATHS["kmeans"].exists() and MODEL_PATHS["scaler"].exists():
        try:
            model: KMeans = joblib.load(MODEL_PATHS["kmeans"])
            scaler: StandardScaler = joblib.load(MODEL_PATHS["scaler"])
            if hasattr(model, "n_clusters") and model.n_clusters == cluster_count:
                x_scaled = scaler.transform(transform_rfm(rfm))
                clustered = rfm.copy()
                clustered["Cluster"] = model.predict(x_scaled)
                clustered["Segment"] = label_segments(clustered)
                evaluation = evaluate_clusters(rfm)
                customer_view = build_customer_view(cleaned_df, clustered)
                return rfm, clustered, customer_view, scaler, model, evaluation
        except Exception:
            pass  # Fall through to fresh training

    clustered, scaler, model, _ = fit_kmeans(rfm, cluster_count)
    evaluation = evaluate_clusters(rfm)
    customer_view = build_customer_view(cleaned_df, clustered)
    return rfm, clustered, customer_view, scaler, model, evaluation


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Gemini tool functions (called automatically by the AI agent)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_store_overview_metrics() -> str:
    """Return high-level business metrics: revenue, customers, products, transactions."""
    cleaned: Optional[pd.DataFrame] = st.session_state.get("cleaned_df")
    if cleaned is None:
        return "Error: Cleaned dataset is not available."
    current_role: str = st.session_state.get("user_role", "Sales Staff")
    if current_role == "Sales Staff":
        return "Access Denied: The 'Sales Staff' role is not authorised to view store financial totals."
    total_rev = float(cleaned["TotalAmount"].sum())
    customers = int(cleaned["CustomerID"].nunique())
    products = int(cleaned["Description"].nunique())
    transactions = int(cleaned["InvoiceNo"].nunique())
    top_countries: dict[str, int] = cleaned["Country"].value_counts().head(3).to_dict()
    return (
        f"Store Metrics Overview:\n"
        f"- Total Revenue: ${total_rev:,.2f}\n"
        f"- Unique Customers: {customers:,}\n"
        f"- Unique Products: {products:,}\n"
        f"- Total Transactions: {transactions:,}\n"
        f"- Top Countries: {', '.join(f'{k} ({v} orders)' for k, v in top_countries.items())}"
    )


def get_customer_profile(customer_id: str) -> str:
    """Retrieve segment, RFM values and marketing strategy for a customer ID.

    Args:
        customer_id: The numeric customer ID (e.g. '17850').
    """
    customer_view: Optional[pd.DataFrame] = st.session_state.get("customer_view")
    if customer_view is None:
        return "Error: Customer segments data is not available."
    customer_id = str(customer_id).strip()
    if not customer_id.isdigit():
        return "Error: Customer ID must be a numeric string."
    cust_data = customer_view[customer_view["CustomerID"] == customer_id]
    if cust_data.empty:
        return f"No customer found with ID: {customer_id}"
    row = cust_data.iloc[0]
    segment = str(row["Segment"])
    rec = float(row["Recency"])
    freq = float(row["Frequency"])
    mon = float(row["Monetary"])
    country = str(row["Country"])
    current_role: str = st.session_state.get("user_role", "Sales Staff")
    mon_display = (
        "[REDACTED — Requires Analyst/Admin role]"
        if current_role == "Sales Staff"
        else f"${mon:,.2f}"
    )
    cust_id_display = f"***{customer_id[-2:]}" if current_role == "Business Analyst" else customer_id
    return (
        f"Customer Profile (ID: {cust_id_display}):\n"
        f"- Segment: {segment}\n"
        f"- Country: {country}\n"
        f"- Recency (days since last purchase): {rec:.0f}\n"
        f"- Frequency (number of orders): {freq:.0f}\n"
        f"- Monetary Spent: {mon_display}\n"
        f"- Actionable Strategy: {segment_strategy(segment)}"
    )


def get_segment_info(segment_name: str) -> str:
    """Return average RFM values and marketing tips for a named customer segment.

    Args:
        segment_name: The segment name or keyword (e.g. 'High-Value').
    """
    clustered_rfm: Optional[pd.DataFrame] = st.session_state.get("clustered_rfm")
    if clustered_rfm is None:
        return "Error: Customer clustering data is not available."
    segment_name = segment_name.strip()
    summary = summarize_segments(clustered_rfm)
    seg_data = summary[summary["Segment"].str.lower() == segment_name.lower()]
    if seg_data.empty:
        valid = ", ".join(summary["Segment"].tolist())
        return f"Segment '{segment_name}' not found. Valid segments: {valid}"
    row = seg_data.iloc[0]
    name = str(row["Segment"])
    count = int(row["Customers"])
    avg_rec = float(row["AvgRecency"])
    avg_freq = float(row["AvgFrequency"])
    avg_mon = float(row["AvgMonetary"])
    current_role: str = st.session_state.get("user_role", "Sales Staff")
    mon_display = (
        "[REDACTED — Requires Analyst/Admin role]"
        if current_role == "Sales Staff"
        else f"${avg_mon:,.2f}"
    )
    return (
        f"Segment Info: {name}\n"
        f"- Customer Count: {count:,} ({count / len(clustered_rfm) * 100:.1f}% of total)\n"
        f"- Average Recency: {avg_rec:.1f} days\n"
        f"- Average Frequency: {avg_freq:.1f} orders\n"
        f"- Average Monetary Spend: {mon_display}\n"
        f"- Actionable Marketing Strategy: {segment_strategy(name)}"
    )


def get_similar_products(product_name: str) -> str:
    """Recommend products similar to a keyword using collaborative filtering.

    Args:
        product_name: The name or keyword of the product.
    """
    similarity_df: Optional[pd.DataFrame] = st.session_state.get("similarity_df")
    if similarity_df is None:
        return "Error: Product similarity database is not available."
    product_name = str(product_name).strip()
    if len(product_name) < 3:
        return "Error: Product search query must be at least 3 characters."
    matched, recs = recommend_products(product_name, similarity_df, top_n=5)
    if recs.empty:
        return f"Could not find any product matching '{product_name}'."
    result = f"Top recommendations for '{matched}':\n"
    for rank, (prod, score) in enumerate(recs.items(), start=1):
        result += f"{rank}. {prod} (similarity score: {score:.3f})\n"
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Secure API key resolution
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _resolve_api_key() -> str:
    """Resolve Gemini API key: env var → secrets.toml → masked sidebar input."""
    # Priority 1: server-side environment variable (production / CI)
    key_env = os.environ.get("GEMINI_API_KEY", "").strip()
    if key_env:
        st.success("🔒 API Key loaded from environment variable.", icon="✅")
        return key_env

    # Priority 2: .streamlit/secrets.toml (git-ignored; Streamlit Cloud secrets panel)
    key_secrets = ""
    try:
        key_secrets = str(st.secrets.get("GEMINI_API_KEY", "")).strip()
    except Exception:
        pass
    if key_secrets:
        st.success("🔒 API Key loaded from secrets.toml.", icon="✅")
        return key_secrets

    # Priority 3: masked text input (never echoed or stored)
    user_input = st.text_input(
        "Gemini API Key",
        type="password",
        placeholder="AIza…  (paste your key here)",
        help=(
            "Your key is never stored or logged. "
            "For permanent setup, add it to .streamlit/secrets.toml "
            "or set the GEMINI_API_KEY environment variable."
        ),
    )
    resolved = user_input.strip()
    if not resolved:
        st.caption("🔑 Enter your key above to activate the AI Assistant.")
    return resolved


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# UI helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def download_csv_button(label: str, df: pd.DataFrame, file_name: str) -> None:
    """Render a styled download button for a DataFrame as CSV."""
    st.download_button(
        label=label,
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=file_name,
        mime="text/csv",
        use_container_width=True,
    )


def _kpi_html(icon: str, label: str, value: str) -> str:
    return (
        f'<div class="kpi-card">'
        f'<div class="kpi-icon">{icon}</div>'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f"</div>"
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Sidebar
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_sidebar() -> tuple[str, Optional[bytes], int, int, str]:
    """Render sidebar controls; return (page, csv_bytes, cluster_count, top_n, api_key)."""
    with st.sidebar:
        st.markdown("## 🛒 Shopper Spectrum")
        st.caption("E-commerce analytics & AI insights")
        st.divider()

        page: str = st.radio(
            "📋 Navigation",
            options=[
                "🏠 Home",
                "👥 Customer Segmentation",
                "🛍️ Product Recommendation",
                "📊 Project Insights",
                "🗄️ Dataset",
                "🤖 AI Assistant",
            ],
        )
        st.divider()

        st.markdown("**📁 Data Source**")
        uploaded_file = st.file_uploader(
            "Upload Online Retail CSV",
            type=["csv"],
            help="Leave empty to use built-in demo data.",
        )
        csv_bytes: Optional[bytes] = (
            uploaded_file.getvalue() if uploaded_file is not None else None
        )

        st.markdown("**⚙️ Model Settings**")
        cluster_count: int = st.slider(
            "K-Means Clusters", min_value=2, max_value=6, value=4,
            help="Number of customer segments to create.",
        )
        top_n: int = st.slider(
            "Recommendations", min_value=3, max_value=10, value=5,
            help="How many similar products to return.",
        )
        st.caption("💡 Demo data is used when no CSV is uploaded.")
        st.divider()

        st.markdown("**🤖 AI Assistant Settings**")
        api_key: str = _resolve_api_key()

    # Strip emojis from page name for clean string comparison
    clean_page = page.split(" ", 1)[-1].strip()
    return clean_page, csv_bytes, cluster_count, top_n, api_key


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Page renderers — each is a self-contained function
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def render_home(cleaned: pd.DataFrame, clustered_rfm: pd.DataFrame) -> None:
    """Home page: segment summary table + revenue trend + segment bar chart."""
    st.markdown('<div class="section-title">📈 Overview</div>', unsafe_allow_html=True)
    left, right = st.columns([0.42, 0.58])

    with left:
        st.markdown("**🔬 Project Modules**")
        for module in [
            "✅ RFM Feature Engineering",
            "✅ K-Means Customer Clustering",
            "✅ Item-Based Collaborative Filtering",
            "✅ Interactive Streamlit Dashboard",
            "✅ Secure AI Business Assistant (Gemini)",
            "✅ MCP Server for Agent Integrations",
        ]:
            st.markdown(f"- {module}")

        st.markdown("**📋 Segment Summary**")
        seg_summary = summarize_segments(clustered_rfm)
        st.dataframe(seg_summary, use_container_width=True, hide_index=True)

    with right:
        st.markdown("**📅 Monthly Revenue Trend**")
        monthly_sales = (
            cleaned.assign(Month=cleaned["InvoiceDate"].dt.to_period("M").astype(str))
            .groupby("Month")["TotalAmount"]
            .sum()
            .reset_index()
            .rename(columns={"TotalAmount": "Revenue (£)"})
            .set_index("Month")
        )
        st.line_chart(monthly_sales, color="#6366f1")

        st.markdown("**👥 Customers by Segment**")
        st.bar_chart(
            clustered_rfm["Segment"].value_counts().rename("Customers"),
            color="#818cf8",
        )


def render_segmentation(
    cleaned: pd.DataFrame,
    clustered_rfm: pd.DataFrame,
    customer_view: pd.DataFrame,
    scaler: StandardScaler,
    model: KMeans,
    similarity_df: pd.DataFrame,
) -> None:
    """Customer Segmentation page: predictor + interactive RFM scatter map."""
    left, right = st.columns([0.35, 0.65])

    with left:
        st.markdown('<div class="section-title">🔮 Predict Segment</div>', unsafe_allow_html=True)
        recency: float = float(
            st.number_input("Recency (days since last purchase)", min_value=0, value=30, step=1,
                            help="How many days ago the customer last purchased.")
        )
        frequency: float = float(
            st.number_input("Frequency (number of orders)", min_value=1, value=5, step=1,
                            help="Total number of distinct orders.")
        )
        monetary: float = float(
            st.number_input("Monetary (total spend £)", min_value=1.0, value=500.0, step=50.0,
                            help="Total lifetime spend in GBP.")
        )

        if st.button("🔮 Predict Customer Segment", type="primary", use_container_width=True):
            cluster, segment = predict_segment(
                recency, frequency, monetary, scaler, model, clustered_rfm
            )
            strategy = segment_strategy(segment)
            st.markdown(
                f"""
                <div class="segment-card">
                    <h3>🏷️ {segment}</h3>
                    <span class="cluster-badge">Cluster #{cluster}</span>
                    <p class="strategy-tip">💡 {strategy}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.divider()
        if st.button("💾 Save Trained Models", use_container_width=True):
            save_models(model, scaler, similarity_df)
            st.success("✅ Models saved to models/ folder.")

    with right:
        st.markdown('<div class="section-title">🗺️ RFM Customer Map</div>', unsafe_allow_html=True)
        all_segs = sorted(customer_view["Segment"].dropna().unique())
        selected = st.multiselect(
            "Filter segments", all_segs, default=all_segs,
            help="Select one or more segments to display.",
        )
        filtered = customer_view[customer_view["Segment"].isin(selected)]
        if filtered.empty:
            st.warning("Please select at least one segment to display the map.")
        else:
            st.scatter_chart(
                filtered, x="Frequency", y="Monetary", color="Segment", size="Recency"
            )
            with st.expander("📄 View customer data table"):
                st.dataframe(filtered.head(300), use_container_width=True, hide_index=True)


def render_recommendations(similarity_df: pd.DataFrame, top_n: int) -> None:
    """Product Recommendation page: search, results cards, and similarity matrix."""
    st.markdown(
        '<div class="section-title">🛍️ Product Recommendation Engine</div>',
        unsafe_allow_html=True,
    )

    col_search, col_btn = st.columns([0.8, 0.2])
    with col_search:
        products = sorted(similarity_df.index.tolist())
        selected_product: str = str(
            st.selectbox("Select a product", products, help="Pick from the catalog.")
        )
        search_product: str = st.text_input(
            "Or type a keyword", value=selected_product,
            help="Partial match is supported.",
        )
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        search_clicked = st.button("🔍 Get Recommendations", type="primary", use_container_width=True)

    if search_clicked:
        matched, recs = recommend_products(search_product, similarity_df, top_n)
        if recs.empty:
            st.error("❌ Product not found. Try a different keyword.")
        else:
            st.success(f"✅ Showing top {top_n} recommendations for: **{matched}**")
            for rank, (product, score) in enumerate(recs.items(), start=1):
                bar_width = int(score * 100)
                st.markdown(
                    f"""
                    <div class="recommend-card">
                        <span class="rec-rank">#{rank}</span>
                        <div class="rec-name">{product}</div>
                        <div class="rec-score">Similarity Score: {score:.4f}</div>
                        <div class="rec-bar" style="width:{bar_width}%"></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    with st.expander("🔬 View product similarity matrix (top 15×15)"):
        st.dataframe(similarity_df.iloc[:15, :15].round(3), use_container_width=True)


def render_insights(cleaned: pd.DataFrame, evaluation: pd.DataFrame) -> None:
    """Project Insights page: cluster evaluation, business benefits, top products."""
    st.markdown(
        '<div class="section-title">📊 Model Evaluation</div>', unsafe_allow_html=True
    )
    if evaluation.empty:
        st.warning("⚠️ More customer records are needed for silhouette evaluation.")
    else:
        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("**Cluster Evaluation Table**")
            st.dataframe(evaluation, use_container_width=True, hide_index=True)
        with col_right:
            st.markdown("**Elbow & Silhouette Curves**")
            st.line_chart(evaluation.set_index("k")[["inertia", "silhouette"]])

    st.markdown(
        '<div class="section-title">💼 Business Benefits</div>', unsafe_allow_html=True
    )
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown(
            """<div class="benefit-card">
                <div class="benefit-icon">🎯</div>
                <h4>Customer Segmentation</h4>
                <p>Identify High-Value, Regular, Occasional, and At-Risk customers
                using RFM clustering for precision marketing.</p>
            </div>""",
            unsafe_allow_html=True,
        )
    with col_b:
        st.markdown(
            """<div class="benefit-card">
                <div class="benefit-icon">🛍️</div>
                <h4>Product Recommendation</h4>
                <p>Suggest complementary products using item-based collaborative
                filtering powered by cosine similarity.</p>
            </div>""",
            unsafe_allow_html=True,
        )
    with col_c:
        st.markdown(
            """<div class="benefit-card">
                <div class="benefit-icon">📈</div>
                <h4>Marketing Strategy</h4>
                <p>Leverage segment behaviour to design retention, upselling,
                and win-back campaigns for measurable ROI.</p>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="section-title">🏆 Top Selling Products</div>', unsafe_allow_html=True
    )
    top_products = (
        cleaned.groupby("Description")["Quantity"]
        .sum()
        .sort_values(ascending=False)
        .head(12)
        .rename("Units Sold")
    )
    st.bar_chart(top_products, color="#6366f1")


def render_dataset(cleaned: pd.DataFrame, customer_view: pd.DataFrame) -> None:
    """Dataset page: preview, download buttons, and EDA charts."""
    st.markdown(
        '<div class="section-title">🗄️ Dataset Preview</div>', unsafe_allow_html=True
    )
    st.dataframe(cleaned.head(500), use_container_width=True, hide_index=True)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        download_csv_button("⬇️ Download Cleaned Data", cleaned, "cleaned_transactions.csv")
    with col_b:
        download_csv_button(
            "⬇️ Download Customer Segments", customer_view, "customer_segments.csv"
        )
    with col_c:
        download_csv_button(
            "⬇️ Download Sample Dataset", generate_sample_transactions(), "sample_online_retail.csv"
        )

    st.markdown('<div class="section-title">📊 Exploratory Data Analysis</div>', unsafe_allow_html=True)
    eda_l, eda_r = st.columns(2)
    with eda_l:
        st.markdown("**Transaction Volume by Country**")
        st.bar_chart(
            cleaned["Country"].value_counts().head(12).rename("Transactions"),
            color="#6366f1",
        )
    with eda_r:
        st.markdown("**Revenue by Country (£)**")
        st.bar_chart(
            cleaned.groupby("Country")["TotalAmount"]
            .sum()
            .sort_values(ascending=False)
            .head(12)
            .rename("Revenue (£)"),
            color="#818cf8",
        )


def render_ai_assistant(
    cleaned: pd.DataFrame,
    clustered_rfm: pd.DataFrame,
    customer_view: pd.DataFrame,
    similarity_df: pd.DataFrame,
    api_key: str,
) -> None:
    """AI Assistant page: Gemini-powered chatbot with RBAC and automatic tool calling."""
    st.markdown(
        """
        <div class="chat-header">
            <h3>🤖 AI Business Assistant</h3>
            <p>Ask questions about customer segments, products, store metrics, and marketing strategies.
            Powered by Gemini with automatic tool calling.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not api_key:
        st.info(
            "🔑 Please provide a **Gemini API Key** in the sidebar to activate "
            "the AI Business Assistant.",
            icon="ℹ️",
        )
        return

    # Role-based access control (RBAC) selector
    role: str = st.sidebar.selectbox(
        "🔐 Your Role (Security Demo)", ROLES,
        help="Different roles grant different levels of data access.",
    )
    st.session_state["user_role"] = role

    if role == "Sales Staff":
        st.markdown(
            '<span class="role-pill role-staff">🔒 Sales Staff — Financial data redacted</span>',
            unsafe_allow_html=True,
        )
    elif role == "Business Analyst":
        st.markdown(
            '<span class="role-pill role-analyst">🔑 Business Analyst — Customer IDs masked</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="role-pill role-admin">🔓 Administrator — Full access</span>',
            unsafe_allow_html=True,
        )

    # Chat history management
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state["chat_history"] = []
        st.rerun()

    # Render existing messages
    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Handle new user input
    prompt: Optional[str] = st.chat_input(
        "Ask about customers, products, segments, or business metrics…"
    )
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state["chat_history"].append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("⏳ Thinking…")
            try:
                client = genai.Client(api_key=api_key)
                contents: list[types.Content] = []
                for msg in st.session_state["chat_history"][:-1]:
                    role_map = "user" if msg["role"] == "user" else "model"
                    contents.append(
                        types.Content(
                            role=role_map,
                            parts=[types.Part.from_text(text=msg["content"])],
                        )
                    )
                contents.append(
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=prompt)],
                    )
                )
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=(
                            "You are the Shopper Spectrum AI Concierge — a highly secure, "
                            "knowledgeable business intelligence agent. You help e-commerce teams "
                            "query business metrics, analyse customer segments, and retrieve product "
                            "recommendations. You have access to live tools that query real data. "
                            "Security rules: always enforce the active role constraints. If a tool "
                            "returns 'Access Denied', explain politely without revealing internal "
                            "system details or credentials. "
                            "Format all responses using clean markdown: use tables, bullet points, "
                            "and bold headings to improve readability."
                        ),
                        tools=[
                            get_store_overview_metrics,
                            get_customer_profile,
                            get_segment_info,
                            get_similar_products,
                        ],
                    ),
                )
                ai_response = response.text
                placeholder.markdown(ai_response)
                st.session_state["chat_history"].append(
                    {"role": "assistant", "content": ai_response}
                )
            except Exception as exc:
                err_msg = f"⚠️ Error: {exc}"
                placeholder.markdown(err_msg)
                st.session_state["chat_history"].append(
                    {"role": "assistant", "content": err_msg}
                )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Application entry point
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main() -> None:
    """Orchestrates sidebar, data loading, caching, and page routing."""
    add_css()

    # ── Sidebar: returns all user settings ─────────────────────────────────
    page, csv_bytes, cluster_count, top_n, api_key = build_sidebar()

    # ── Data loading (all cached) ──────────────────────────────────────────
    _raw, cleaned = load_clean_data(csv_bytes)
    use_saved: bool = csv_bytes is None
    similarity_df: pd.DataFrame = get_similarity_matrix(cleaned, use_saved=use_saved)
    rfm, clustered_rfm, customer_view, scaler, model, evaluation = get_rfm_clustering(
        cleaned, cluster_count, use_saved=use_saved
    )

    # ── Persist DataFrames in session state for Gemini tool functions ──────
    st.session_state["cleaned_df"] = cleaned
    st.session_state["clustered_rfm"] = clustered_rfm
    st.session_state["customer_view"] = customer_view
    st.session_state["similarity_df"] = similarity_df

    # ── Auto-export customer segments (for MCP server and external tools) ──
    try:
        Path("outputs").mkdir(exist_ok=True)
        customer_view.to_csv("outputs/customer_segments.csv", index=False)
    except Exception:
        pass

    # ── Hero banner ────────────────────────────────────────────────────────
    total_rev = cleaned["TotalAmount"].sum()
    num_customers = cleaned["CustomerID"].nunique()
    num_products = cleaned["Description"].nunique()
    num_transactions = cleaned["InvoiceNo"].nunique()
    num_countries = cleaned["Country"].nunique()

    st.markdown(
        f"""
        <div class="hero">
            <h1>🛒 Shopper Spectrum</h1>
            <p>AI-powered customer segmentation &amp; product recommendation for modern e-commerce</p>
            <span class="badge">🚀 Production Ready</span>
            <span class="badge" style="margin-left:.5rem">🤖 Gemini AI</span>
            <span class="badge" style="margin-left:.5rem">📊 Real-time Analytics</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── KPI cards row ──────────────────────────────────────────────────────
    kpi_html = (
        '<div class="kpi-grid">'
        + _kpi_html("💰", "Total Revenue", f"£{total_rev:,.0f}")
        + _kpi_html("👥", "Customers", f"{num_customers:,}")
        + _kpi_html("📦", "Products", f"{num_products:,}")
        + _kpi_html("🧾", "Transactions", f"{num_transactions:,}")
        + _kpi_html("🌍", "Countries", f"{num_countries:,}")
        + "</div>"
    )
    st.markdown(kpi_html, unsafe_allow_html=True)

    # ── Page routing ───────────────────────────────────────────────────────
    if page == "Home":
        render_home(cleaned, clustered_rfm)
    elif page == "Customer Segmentation":
        render_segmentation(cleaned, clustered_rfm, customer_view, scaler, model, similarity_df)
    elif page == "Product Recommendation":
        render_recommendations(similarity_df, top_n)
    elif page == "Project Insights":
        render_insights(cleaned, evaluation)
    elif page == "Dataset":
        render_dataset(cleaned, customer_view)
    elif page == "AI Assistant":
        render_ai_assistant(
            cleaned, clustered_rfm, customer_view, similarity_df, api_key=api_key
        )


if __name__ == "__main__":
    main()