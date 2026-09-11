"""
Phase 14 - Streamlit dashboard.

DATA WINDOW LIMITATION (shown in-app, not hidden): all data is bounded to
the validated modeling window (2013-07-01 to 2023-06-30, per Phase 3/9).
"Current" price and "latest" prediction reflect the LAST date in this
dataset - there is no live data feed in this project.

Wheat has NO viable spike model - Phase 11: all three baselines predicted
zero spikes on only 11 historical events. Shown transparently (Option 1
from project discussion) rather than hidden from the commodity selector.

Uses Random Forest specifically - the model that won the Phase 10/11
comparison for Onion and Soyabean.
"""

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(r"C:\Users\siddh\Desktop\PBL")
DATA_PATH = BASE_DIR / "data" / "interim" / "price_sentiment_with_target.csv"
NEWS_PATH = BASE_DIR / "data" / "interim" / "news_with_sentiment.csv"
MODELS_DIR = BASE_DIR / "models"
SHAP_DIR = BASE_DIR / "reports" / "shap"
METADATA_PATH = MODELS_DIR / "training_metadata.json"

NON_FEATURE_COLS = {
    "commodity", "date", "dominant_sentiment",
    "next_price", "next_date", "pct_change_to_next", "days_to_next_report",
    "spike",
}
COMMODITIES_WITH_MODEL = ["Onion", "Soyabean"]
ALL_COMMODITIES = ["Onion", "Soyabean", "Wheat"]
SPIKE_THRESHOLD_PCT = 5.0


@st.cache_data
def load_price_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    return df.sort_values(["commodity", "date"]).reset_index(drop=True)


@st.cache_data
def load_news_data():
    df = pd.read_csv(NEWS_PATH, parse_dates=["publish_date"])
    return df.sort_values("publish_date", ascending=False).reset_index(drop=True)


@st.cache_data
def load_metadata():
    with open(METADATA_PATH) as f:
        return json.load(f)


@st.cache_resource
def load_model(commodity: str):
    return joblib.load(MODELS_DIR / f"{commodity.lower()}_random_forest.pkl")


def get_feature_cols(df):
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


st.set_page_config(page_title="Agricultural Commodity Price Spike Predictor", layout="wide")
st.title("Agricultural Commodity Price Spike Predictor")
st.caption("FinBERT news sentiment + historical mandi prices -> next-trading-day spike prediction. "
           "Semester project | ProsusAI/finbert + Random Forest.")

st.warning(
    "**Data window notice:** all data here is bounded to the validated modeling period "
    "(2013-07-01 to 2023-06-30). There is no live feed - 'current' means the last available "
    "date in this dataset, not today's actual market. Academic project, not a trading tool."
)

df = load_price_data()
news_df = load_news_data()
metadata = load_metadata()
feature_cols = get_feature_cols(df)

commodity = st.selectbox("Select a commodity", ALL_COMMODITIES)
sub = df[df["commodity"] == commodity].sort_values("date").reset_index(drop=True)
latest_row = sub.iloc[-1]

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"{commodity} - Price History")
    st.line_chart(sub.set_index("date")["price"])
    st.metric(f"Last available price ({latest_row['date'].date()})", f"Rs {latest_row['price']:.2f} / quintal")

with col2:
    st.subheader("Recent News & Sentiment")
    recent_news = news_df[news_df["commodity"] == commodity].head(5)
    if len(recent_news) == 0:
        st.info("No matching headlines found for this commodity.")
    else:
        for _, row in recent_news.iterrows():
            emoji = {"positive": "🟢", "negative": "🔴", "neutral": "⚪"}.get(row["sentiment_label"], "")
            st.write(f"{emoji} **{row['publish_date'].date()}** — {row['headline_text']} "
                     f"_(model: {row['sentiment_label']})_")

st.divider()
st.subheader("Spike Prediction")

if commodity not in COMMODITIES_WITH_MODEL:
    st.error(
        f"**No viable prediction model for {commodity}.** Only 11 spike events (5% next-day threshold) "
        f"occurred across {commodity}'s entire 10-year validated price history - too few to train or "
        f"evaluate a classifier meaningfully (the held-out test set contained just 3 positive examples). "
        f"All three baseline models tested (Logistic Regression, Random Forest, XGBoost) predicted zero "
        f"spikes. Reported here as a genuine finding about {commodity}'s volatility at this market and "
        f"threshold, not a missing feature."
    )
else:
    model = load_model(commodity)
    X_latest = sub[feature_cols].iloc[[-1]]

    if X_latest.isnull().any(axis=1).iloc[0]:
        st.error("Latest row has missing feature values - cannot generate a prediction. Check the pipeline output.")
    else:
        proba = model.predict_proba(X_latest)[0][1]
        prediction = "SPIKE" if proba >= 0.5 else "NO SPIKE"

        # Confidence band, not just the binary call - a 5% and a 45% probability
        # both round to "NO SPIKE" above, but they mean very different things
        # given this model's known precision (see metrics below). Bands are
        # somewhat arbitrary but deliberately wide in the middle, since a
        # ~33-38% precision model has limited ability to distinguish a
        # genuinely low-risk day from a genuinely uncertain one near 50%.
        if proba < 0.20:
            confidence = "Confident — low risk"
        elif proba < 0.40:
            confidence = "Leaning no spike"
        elif proba < 0.60:
            confidence = "Borderline — treat as uncertain"
        elif proba < 0.80:
            confidence = "Leaning spike"
        else:
            confidence = "Confident — high risk"

        pcol1, pcol2, pcol3 = st.columns(3)
        pcol1.metric("Predicted spike probability", f"{proba*100:.1f}%")
        pcol2.metric("Prediction", prediction)
        pcol3.metric("Confidence", confidence)
        st.caption(f"Based on features from {latest_row['date'].date()} (last available date), predicting "
                   f"whether {commodity}'s price rises {SPIKE_THRESHOLD_PCT:.0f}%+ on the next trading day. "
                   f"Confidence band is descriptive only, not a separate model output — see Model Performance "
                   f"below for this model's actual precision/recall before trusting any single prediction.")

        st.subheader("Model Performance (held-out test set, Phase 10/11)")
        mm = metadata[commodity]["models"]["random_forest"]
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Accuracy", f"{mm['accuracy']*100:.1f}%")
        m2.metric("Precision", f"{mm['precision']*100:.1f}%")
        m3.metric("Recall", f"{mm['recall']*100:.1f}%")
        m4.metric("F1 Score", f"{mm['f1']*100:.1f}%")
        m5.metric("ROC-AUC", f"{mm['roc_auc']*100:.1f}%" if mm['roc_auc'] else "N/A")
        st.caption(
            f"Evaluated on {metadata[commodity]['test_rows']} test days "
            f"({metadata[commodity]['test_date_range'][0]} to {metadata[commodity]['test_date_range'][1]}), "
            f"{metadata[commodity]['test_positive']} of which were actual spike days. Precision/recall in "
            f"this range are modest, not high-confidence - see project report for full discussion."
        )

        cm = mm["confusion_matrix"]
        st.write("**Confusion Matrix** (rows = actual, columns = predicted)")
        st.dataframe(pd.DataFrame(
            cm, index=["Actual: No Spike", "Actual: Spike"],
            columns=["Predicted: No Spike", "Predicted: Spike"]
        ))

        st.subheader("Most Influential Features (SHAP)")
        shap_path = SHAP_DIR / f"{commodity.lower()}_shap_bar.png"
        if shap_path.exists():
            st.image(str(shap_path), caption=f"{commodity} - SHAP feature importance (Phase 13)")
        else:
            st.info("SHAP plot not found - run src/explain.py first.")