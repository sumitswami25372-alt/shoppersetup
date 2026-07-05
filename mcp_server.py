"""Shopper Spectrum MCP Server — exposes customer and product data tools via the
Model Context Protocol for use with Claude Desktop, Cursor, and other AI agents.
"""
import warnings
import pandas as pd
import joblib
from pathlib import Path
from mcp.server.fastmcp import FastMCP

# Suppress sklearn unpickling warnings
warnings.filterwarnings("ignore")

# Initialize FastMCP Server
mcp = FastMCP("Shopper Spectrum MCP Server")

# Paths to models & processed files
SEGMENTS_PATH = Path("outputs/customer_segments.csv")
SIMILARITY_PATH = Path("models/product_similarity.pkl")


def load_data() -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Load customer segments CSV and product similarity matrix if they exist."""
    segments_df: pd.DataFrame | None = None
    similarity_df: pd.DataFrame | None = None

    if SEGMENTS_PATH.exists():
        try:
            segments_df = pd.read_csv(SEGMENTS_PATH)
        except Exception:
            pass

    if SIMILARITY_PATH.exists():
        try:
            similarity_df = joblib.load(SIMILARITY_PATH)
        except Exception:
            pass

    return segments_df, similarity_df


@mcp.tool()
def get_customer_details(customer_id: str) -> str:
    """Gets segment labels, recency, frequency, and monetary metrics for a specific customer ID from the Shopper Spectrum dataset.

    Args:
        customer_id: The ID of the customer (e.g., '17850').
    """
    segments_df, _ = load_data()
    if segments_df is None:
        return "Error: Customer segments data is not available. Please run the Streamlit dashboard first to generate outputs/customer_segments.csv."

    customer_id = customer_id.strip()
    cust_row = segments_df[segments_df["CustomerID"].astype(str) == customer_id]
    if cust_row.empty:
        # Try casting to float representation if ID contains decimal
        try:
            float_id = float(customer_id)
            cust_row = segments_df[segments_df["CustomerID"].astype(float) == float_id]
        except ValueError:
            pass

    if cust_row.empty:
        return f"Customer ID '{customer_id}' not found in the processed segments database."

    row = cust_row.iloc[0]
    return (
        f"Customer {customer_id} Details:\n"
        f"- Segment: {row['Segment']}\n"
        f"- Country: {row['Country']}\n"
        f"- Recency (Days since last purchase): {row['Recency']}\n"
        f"- Frequency (Number of orders): {row['Frequency']}\n"
        f"- Monetary Spent (Total value): ${row['Monetary']:,.2f}\n"
        f"- Last Purchase Date: {row['LastPurchase']}"
    )


@mcp.tool()
def get_product_recommendations(product_name: str, limit: int = 5) -> str:
    """Retrieves top product recommendations based on item similarity.

    Args:
        product_name: The name or keyword of the product (e.g., 'WHITE HANGING HEART').
        limit: Max number of recommendations to return (default 5, max 10).
    """
    _, similarity_df = load_data()
    if similarity_df is None:
        return "Error: Product similarity model is not available. Please save models from the Streamlit app."

    product_name = product_name.strip().upper()
    if not product_name:
        return "Error: Please provide a valid product name query."

    products = similarity_df.index.to_series()
    matched_product = product_name

    if matched_product not in similarity_df.index:
        matches = products[products.str.contains(product_name, case=False, regex=False)]
        if matches.empty:
            return f"Product keyword '{product_name}' not found."
        matched_product = str(matches.iloc[0])

    limit = min(max(1, limit), 10)
    recommendations = (
        similarity_df[matched_product]
        .drop(index=matched_product)
        .sort_values(ascending=False)
        .head(limit)
    )

    result = f"Top {limit} recommendations for '{matched_product}':\n"
    for rank, (prod, score) in enumerate(recommendations.items(), start=1):
        result += f"{rank}. {prod} (Similarity: {score:.3f})\n"
    return result


@mcp.tool()
def get_segment_metrics() -> str:
    """Returns a summary of all customer segments, customer counts, and averages."""
    segments_df, _ = load_data()
    if segments_df is None:
        return "Error: Customer segments data is not available. Please run the Streamlit dashboard first."

    summary = (
        segments_df.groupby("Segment")
        .agg(
            Customers=("CustomerID", "count"),
            AvgRecency=("Recency", "mean"),
            AvgFrequency=("Frequency", "mean"),
            AvgMonetary=("Monetary", "mean"),
        )
        .round(2)
        .sort_values("AvgMonetary", ascending=False)
    )

    result = "Shopper Spectrum Customer Segments Summary:\n"
    total_customers = len(segments_df)
    for idx, (seg, row) in enumerate(summary.iterrows(), 1):
        pct = (row['Customers'] / total_customers) * 100
        result += (
            f"\n{idx}. Segment: {seg}\n"
            f"   - Customers: {row['Customers']:,} ({pct:.1f}%)\n"
            f"   - Avg Recency: {row['AvgRecency']:.1f} days\n"
            f"   - Avg Frequency: {row['AvgFrequency']:.1f} orders\n"
            f"   - Avg Monetary Spent: ${row['AvgMonetary']:,.2f}\n"
        )
    return result


if __name__ == "__main__":
    # FastMCP uses stdio transport by default which Claude Desktop and other clients read from stdin/stdout.
    mcp.run()
