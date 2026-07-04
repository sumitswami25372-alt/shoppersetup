from __future__ import annotations

import io
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler


st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon=":shopping_trolley:",
    layout="wide",
    initial_sidebar_state="expanded",
)


REQUIRED_COLUMNS = [
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
    "Country",
]

MODEL_PATHS = {
    "kmeans": Path("models/kmeans_model.pkl"),
    "scaler": Path("models/rfm_scaler.pkl"),
    "similarity": Path("models/product_similarity.pkl"),
}

PRODUCT_CATALOG = [
    ("SKU1001", "WHITE HANGING HEART T-LIGHT HOLDER", "Home Decor", 2.95),
    ("SKU1002", "REGENCY CAKESTAND 3 TIER", "Kitchen", 12.75),
    ("SKU1003", "JUMBO BAG RED RETROSPOT", "Bags", 2.08),
    ("SKU1004", "ASSORTED COLOUR BIRD ORNAMENT", "Home Decor", 1.69),
    ("SKU1005", "LUNCH BAG RED RETROSPOT", "Bags", 1.65),
    ("SKU1006", "PACK OF 72 RETROSPOT CAKE CASES", "Kitchen", 0.55),
    ("SKU1007", "SET OF 3 CAKE TINS PANTRY DESIGN", "Kitchen", 4.95),
    ("SKU1008", "PAPER CHAIN KIT 50'S CHRISTMAS", "Seasonal", 2.95),
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


def add_css() -> None:
    st.markdown(
        """
        <style>
        .main .block-container {padding-top: 1.5rem;}
        .hero {
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 1.25rem 1.4rem;
            background: #ffffff;
            margin-bottom: 1rem;
        }
        .hero h1 {
            font-size: 2.1rem;
            margin: 0 0 .25rem 0;
            color: #111827;
        }
        .hero p {
            margin: 0;
            color: #475569;
            font-size: 1rem;
        }
        .segment-card {
            border: 1px solid #dbe3ea;
            border-radius: 8px;
            padding: 1rem;
            background: #f8fafc;
            margin-top: .75rem;
        }
        .segment-card h3 {
            margin: 0 0 .35rem 0;
            color: #111827;
        }
        .recommend-card {
            border: 1px solid #e2e8f0;
            border-left: 5px solid #2563eb;
            border-radius: 8px;
            padding: .85rem 1rem;
            margin-bottom: .65rem;
            background: #ffffff;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def generate_sample_transactions(rows: int = 4500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    customers = np.arange(10001, 10321)
    countries = np.array(["United Kingdom", "Germany", "France", "Spain", "Netherlands", "India", "Australia"])
    country_prob = np.array([0.58, 0.11, 0.10, 0.07, 0.05, 0.05, 0.04])

    catalog = pd.DataFrame(PRODUCT_CATALOG, columns=["StockCode", "Description", "Category", "BasePrice"])
    category_weights = {
        "Home Decor": 1.25,
        "Kitchen": 1.2,
        "Bags": 1.05,
        "Kids": 0.9,
        "Gifts": 0.82,
        "Wellness": 0.7,
        "Garden": 0.55,
        "Seasonal": 0.65,
    }
    weights = catalog["Category"].map(category_weights).to_numpy()
    weights = weights / weights.sum()

    start_date = np.datetime64("2022-01-01")
    dates = start_date + rng.integers(0, 730, size=rows).astype("timedelta64[D]")
    times = rng.integers(8 * 60, 22 * 60, size=rows).astype("timedelta64[m]")
    picked_products = catalog.iloc[rng.choice(catalog.index, size=rows, p=weights)].reset_index(drop=True)

    quantities = rng.poisson(lam=3.2, size=rows) + 1
    bulk_mask = rng.random(rows) < 0.08
    quantities[bulk_mask] += rng.integers(8, 35, size=bulk_mask.sum())
    prices = picked_products["BasePrice"].to_numpy() * rng.normal(1.0, 0.08, size=rows)
    prices = np.clip(np.round(prices, 2), 0.2, None)

    df = pd.DataFrame(
        {
            "InvoiceNo": (536000 + rng.integers(0, rows // 2, size=rows)).astype(str),
            "StockCode": picked_products["StockCode"],
            "Description": picked_products["Description"],
            "Quantity": quantities,
            "InvoiceDate": pd.to_datetime((dates + times).astype("datetime64[ns]")),
            "UnitPrice": prices,
            "CustomerID": rng.choice(customers, size=rows),
            "Country": rng.choice(countries, size=rows, p=country_prob),
        }
    )

    cancelled_index = df.sample(frac=0.015, random_state=seed).index
    df.loc[cancelled_index, "InvoiceNo"] = "C" + df.loc[cancelled_index, "InvoiceNo"].astype(str)
    missing_index = df.sample(frac=0.02, random_state=seed + 1).index
    df.loc[missing_index, "CustomerID"] = np.nan
    return df


def read_uploaded_csv(csv_bytes: bytes) -> pd.DataFrame:
    for encoding in ("utf-8", "latin1", "cp1252"):
        try:
            return pd.read_csv(io.BytesIO(csv_bytes), encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(io.BytesIO(csv_bytes))


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = {str(col).strip().lower(): col for col in df.columns}
    rename_map = {}
    for required in REQUIRED_COLUMNS:
        found = normalized.get(required.lower())
        if found is not None:
            rename_map[found] = required
    return df.rename(columns=rename_map)


def load_transactions(csv_bytes: bytes | None) -> pd.DataFrame:
    if csv_bytes is None:
        return generate_sample_transactions()

    df = normalize_columns(read_uploaded_csv(csv_bytes))
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        st.error("Your CSV is missing required columns: " + ", ".join(missing))
        st.stop()
    return df


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
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
    snapshot_date = cleaned["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = (
        cleaned.groupby("CustomerID")
        .agg(
            Recency=("InvoiceDate", lambda value: (snapshot_date - value.max()).days),
            Frequency=("InvoiceNo", "nunique"),
            Monetary=("TotalAmount", "sum"),
        )
        .reset_index()
    )
    rfm["Monetary"] = rfm["Monetary"].round(2)
    return rfm


def transform_rfm(rfm: pd.DataFrame) -> pd.DataFrame:
    transformed = rfm[["Recency", "Frequency", "Monetary"]].copy()
    transformed["Recency"] = np.log1p(transformed["Recency"].clip(lower=0))
    transformed["Frequency"] = np.log1p(transformed["Frequency"].clip(lower=0))
    transformed["Monetary"] = np.log1p(transformed["Monetary"].clip(lower=0))
    return transformed


def label_segments(clustered_rfm: pd.DataFrame) -> pd.Series:
    profiles = (
        clustered_rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary"]]
        .mean()
        .assign(
            RecencyRank=lambda df: df["Recency"].rank(ascending=True),
            FrequencyRank=lambda df: df["Frequency"].rank(ascending=False),
            MonetaryRank=lambda df: df["Monetary"].rank(ascending=False),
        )
    )
    profiles["Score"] = profiles["RecencyRank"] + profiles["FrequencyRank"] + profiles["MonetaryRank"]
    label_order = ["High-Value", "Regular", "Occasional", "At-Risk", "Needs Attention", "New Buyer"]
    ordered_clusters = profiles.sort_values("Score").index.tolist()
    label_map = {cluster: label_order[index] for index, cluster in enumerate(ordered_clusters)}
    return clustered_rfm["Cluster"].map(label_map)


def fit_kmeans(rfm: pd.DataFrame, cluster_count: int):
    cluster_count = max(2, min(cluster_count, len(rfm)))
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(transform_rfm(rfm))
    model = KMeans(n_clusters=cluster_count, n_init=30, random_state=42)
    clustered = rfm.copy()
    clustered["Cluster"] = model.fit_predict(x_scaled)
    clustered["Segment"] = label_segments(clustered)
    return clustered, scaler, model, x_scaled


def evaluate_clusters(rfm: pd.DataFrame, max_k: int = 8) -> pd.DataFrame:
    if len(rfm) < 3:
        return pd.DataFrame(columns=["k", "inertia", "silhouette"])

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(transform_rfm(rfm))
    rows = []
    for k in range(2, min(max_k, len(rfm) - 1) + 1):
        model = KMeans(n_clusters=k, n_init=20, random_state=42)
        labels = model.fit_predict(x_scaled)
        rows.append(
            {
                "k": k,
                "inertia": round(model.inertia_, 2),
                "silhouette": round(silhouette_score(x_scaled, labels), 4),
            }
        )
    return pd.DataFrame(rows)


def build_similarity(cleaned: pd.DataFrame) -> pd.DataFrame:
    product_matrix = cleaned.pivot_table(
        index="CustomerID",
        columns="Description",
        values="Quantity",
        aggfunc="sum",
        fill_value=0,
    )
    similarity = cosine_similarity(product_matrix.T)
    return pd.DataFrame(similarity, index=product_matrix.columns, columns=product_matrix.columns)


def load_saved_models():
    if all(path.exists() for path in MODEL_PATHS.values()):
        return (
            joblib.load(MODEL_PATHS["kmeans"]),
            joblib.load(MODEL_PATHS["scaler"]),
            joblib.load(MODEL_PATHS["similarity"]),
        )
    return None


def save_models(model, scaler, similarity_df: pd.DataFrame) -> None:
    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATHS["kmeans"])
    joblib.dump(scaler, MODEL_PATHS["scaler"])
    joblib.dump(similarity_df, MODEL_PATHS["similarity"])


def segment_strategy(segment: str) -> str:
    strategies = {
        "High-Value": "Offer VIP benefits, early access, premium bundles, and loyalty rewards.",
        "Regular": "Send personalized offers and cross-sell products related to previous purchases.",
        "Occasional": "Use seasonal discounts and reminder campaigns to increase repeat purchases.",
        "At-Risk": "Trigger win-back campaigns, limited-time coupons, and feedback collection.",
        "Needs Attention": "Monitor purchase behavior and test targeted product offers.",
        "New Buyer": "Send onboarding recommendations and first-repeat-purchase incentives.",
    }
    return strategies.get(segment, "Review this customer group and design a targeted campaign.")


def predict_segment(recency, frequency, monetary, scaler, model, clustered_rfm):
    input_df = pd.DataFrame([{"Recency": recency, "Frequency": frequency, "Monetary": monetary}])
    scaled_input = scaler.transform(transform_rfm(input_df))
    cluster = int(model.predict(scaled_input)[0])
    segment = clustered_rfm.loc[clustered_rfm["Cluster"] == cluster, "Segment"].mode().iloc[0]
    return cluster, segment


def recommend_products(product_name: str, similarity_df: pd.DataFrame, top_n: int):
    product_name = product_name.strip().upper()
    if not product_name:
        return None, pd.Series(dtype=float)

    products = similarity_df.index.to_series()
    matched_product = product_name
    if matched_product not in similarity_df.index:
        matches = products[products.str.contains(product_name, case=False, regex=False)]
        if matches.empty:
            return None, pd.Series(dtype=float)
        matched_product = matches.iloc[0]

    recommendations = (
        similarity_df[matched_product]
        .drop(index=matched_product)
        .sort_values(ascending=False)
        .head(top_n)
    )
    return matched_product, recommendations


def summarize_segments(clustered_rfm: pd.DataFrame) -> pd.DataFrame:
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


def build_customer_view(cleaned: pd.DataFrame, clustered_rfm: pd.DataFrame) -> pd.DataFrame:
    last_purchase = cleaned.groupby("CustomerID")["InvoiceDate"].max().dt.date.rename("LastPurchase")
    country = cleaned.groupby("CustomerID")["Country"].agg(lambda value: value.mode().iloc[0]).rename("Country")
    return clustered_rfm.join(last_purchase, on="CustomerID").join(country, on="CustomerID")


@st.cache_data(show_spinner="Preparing data and ML models...")
def prepare_data(csv_bytes: bytes | None, cluster_count: int):
    raw = load_transactions(csv_bytes)
    cleaned = clean_transactions(raw)
    if cleaned.empty or cleaned["CustomerID"].nunique() < 2:
        st.error("Not enough valid customer transactions after cleaning.")
        st.stop()

    rfm = build_rfm(cleaned)
    clustered_rfm, scaler, model, _ = fit_kmeans(rfm, cluster_count)
    similarity_df = build_similarity(cleaned)
    evaluation = evaluate_clusters(rfm)
    customer_view = build_customer_view(cleaned, clustered_rfm)
    return raw, cleaned, rfm, clustered_rfm, customer_view, scaler, model, similarity_df, evaluation


def download_csv_button(label: str, df: pd.DataFrame, file_name: str) -> None:
    st.download_button(
        label=label,
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=file_name,
        mime="text/csv",
        use_container_width=True,
    )


def main() -> None:
    add_css()

    with st.sidebar:
        st.title("Shopper Spectrum")
        page = st.radio(
            "Navigation",
            ["Home", "Customer Segmentation", "Product Recommendation", "Project Insights", "Dataset"],
        )
        st.divider()
        uploaded_file = st.file_uploader("Upload Online Retail CSV", type=["csv"])
        csv_bytes = uploaded_file.getvalue() if uploaded_file is not None else None
        cluster_count = st.slider("K-Means clusters", 2, 6, 4)
        top_n = st.slider("Recommendations", 3, 10, 5)
        st.caption("The app uses sample data when no CSV is uploaded.")

    raw, cleaned, rfm, clustered_rfm, customer_view, scaler, model, similarity_df, evaluation = prepare_data(
        csv_bytes,
        cluster_count,
    )

    st.markdown(
        """
        <div class="hero">
            <h1>Shopper Spectrum</h1>
            <p>Interactive customer segmentation and product recommendation system for e-commerce analytics.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_cols = st.columns(5)
    metric_cols[0].metric("Revenue", f"${cleaned['TotalAmount'].sum():,.2f}")
    metric_cols[1].metric("Customers", f"{cleaned['CustomerID'].nunique():,}")
    metric_cols[2].metric("Products", f"{cleaned['Description'].nunique():,}")
    metric_cols[3].metric("Transactions", f"{cleaned['InvoiceNo'].nunique():,}")
    metric_cols[4].metric("Countries", f"{cleaned['Country'].nunique():,}")

    if page == "Home":
        left, right = st.columns([0.45, 0.55])
        with left:
            st.subheader("Project Modules")
            st.write("RFM feature engineering")
            st.write("K-Means customer clustering")
            st.write("Item-based collaborative filtering")
            st.write("Interactive Streamlit dashboard")

            st.subheader("Segment Summary")
            st.dataframe(summarize_segments(clustered_rfm), use_container_width=True, hide_index=True)

        with right:
            st.subheader("Monthly Revenue Trend")
            monthly_sales = (
                cleaned.assign(Month=cleaned["InvoiceDate"].dt.to_period("M").astype(str))
                .groupby("Month")["TotalAmount"]
                .sum()
            )
            st.line_chart(monthly_sales)

            st.subheader("Customer Count by Segment")
            st.bar_chart(clustered_rfm["Segment"].value_counts())

    elif page == "Customer Segmentation":
        left, right = st.columns([0.35, 0.65])
        with left:
            st.subheader("Predict Customer Segment")
            recency = st.number_input("Recency in days", min_value=0, value=30, step=1)
            frequency = st.number_input("Frequency", min_value=1, value=5, step=1)
            monetary = st.number_input("Monetary spend", min_value=1.0, value=500.0, step=50.0)

            if st.button("Predict Customer Segment", type="primary", use_container_width=True):
                cluster, segment = predict_segment(recency, frequency, monetary, scaler, model, clustered_rfm)
                st.markdown(
                    f"""
                    <div class="segment-card">
                        <h3>{segment}</h3>
                        <p>Cluster Number: <b>{cluster}</b></p>
                        <p>{segment_strategy(segment)}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            if st.button("Save trained models", use_container_width=True):
                save_models(model, scaler, similarity_df)
                st.success("Models saved in the models folder.")

        with right:
            st.subheader("Interactive RFM Customer Map")
            segment_options = sorted(customer_view["Segment"].dropna().unique())
            selected_segments = st.multiselect("Filter segments", segment_options, default=segment_options)
            filtered = customer_view[customer_view["Segment"].isin(selected_segments)]
            st.scatter_chart(filtered, x="Frequency", y="Monetary", color="Segment", size="Recency")
            st.dataframe(filtered.head(300), use_container_width=True, hide_index=True)

    elif page == "Product Recommendation":
        st.subheader("Product Recommendation Engine")
        products = sorted(similarity_df.index.tolist())
        selected_product = st.selectbox("Select product", products)
        search_product = st.text_input("Or type product keyword", value=selected_product)

        if st.button("Get Recommendations", type="primary"):
            matched_product, recommendations = recommend_products(search_product, similarity_df, top_n)
            if recommendations.empty:
                st.error("Product not found. Try another product keyword.")
            else:
                st.success(f"Recommendations for: {matched_product}")
                for rank, (product, score) in enumerate(recommendations.items(), start=1):
                    st.markdown(
                        f"""
                        <div class="recommend-card">
                            <b>{rank}. {product}</b><br>
                            Similarity Score: {score:.3f}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        with st.expander("View product similarity matrix"):
            st.dataframe(similarity_df.iloc[:15, :15].round(2), use_container_width=True)

    elif page == "Project Insights":
        st.subheader("Model Evaluation")
        if evaluation.empty:
            st.warning("More customer records are needed for silhouette evaluation.")
        else:
            st.dataframe(evaluation, use_container_width=True, hide_index=True)
            st.line_chart(evaluation.set_index("k")[["inertia", "silhouette"]])

        st.subheader("Business Benefits")
        insight_cols = st.columns(3)
        with insight_cols[0]:
            st.write("Customer Segmentation")
            st.write("Identify high-value, regular, occasional, and at-risk customers.")
        with insight_cols[1]:
            st.write("Product Recommendation")
            st.write("Recommend similar products using customer-product purchase patterns.")
        with insight_cols[2]:
            st.write("Marketing Strategy")
            st.write("Use RFM behavior to plan retention, upselling, and win-back campaigns.")

        st.subheader("Top Selling Products")
        top_products = cleaned.groupby("Description")["Quantity"].sum().sort_values(ascending=False).head(12)
        st.bar_chart(top_products)

    elif page == "Dataset":
        st.subheader("Dataset Preview")
        st.dataframe(cleaned.head(500), use_container_width=True, hide_index=True)
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            download_csv_button("Download cleaned data", cleaned, "cleaned_transactions.csv")
        with col_b:
            download_csv_button("Download customer segments", customer_view, "customer_segments.csv")
        with col_c:
            sample_data = generate_sample_transactions()
            download_csv_button("Download sample dataset", sample_data, "sample_online_retail.csv")

        st.subheader("EDA")
        eda_left, eda_right = st.columns(2)
        with eda_left:
            st.write("Transaction volume by country")
            st.bar_chart(cleaned["Country"].value_counts().head(12))
        with eda_right:
            st.write("Revenue by country")
            st.bar_chart(cleaned.groupby("Country")["TotalAmount"].sum().sort_values(ascending=False).head(12))


if __name__ == "__main__":
    main()