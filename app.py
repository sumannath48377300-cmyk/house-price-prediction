"""
House Price Prediction & Analytics — Production-Ready Streamlit App
Dataset: Housing_cleaned.csv (545 properties, 13 features)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error
)
from sklearn.inspection import permutation_importance

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="House Price Prediction & Analytics",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .main { background-color: #f8fafc; }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 100%);
        border-radius: 12px;
        padding: 16px 20px;
        color: white !important;
        box-shadow: 0 4px 15px rgba(37,99,235,0.25);
    }
    [data-testid="metric-container"] * { color: white !important; }

    /* Section headers */
    .section-header {
        font-size: 1.45rem;
        font-weight: 700;
        color: #1e3a5f;
        border-left: 4px solid #2563eb;
        padding-left: 12px;
        margin: 24px 0 16px 0;
    }

    /* Prediction result box */
    .pred-box {
        background: linear-gradient(135deg, #1e3a5f 0%, #1d4ed8 100%);
        border-radius: 16px;
        padding: 32px;
        text-align: center;
        color: white;
        box-shadow: 0 8px 32px rgba(37,99,235,0.35);
        margin: 16px 0;
    }
    .pred-box h1 { font-size: 2.6rem; margin: 0; }
    .pred-box p  { font-size: 1rem; opacity: 0.85; margin: 4px 0 0 0; }

    /* Info card */
    .info-card {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 10px;
        padding: 16px 20px;
        margin: 8px 0;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e3a5f 100%);
    }
    section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    section[data-testid="stSidebar"] .stSelectbox label { color: #94a3b8 !important; }

    /* Tab styling */
    .stTabs [data-baseweb="tab"] {
        font-size: 0.95rem;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        border-bottom: 3px solid #2563eb;
        color: #2563eb !important;
    }

    /* Hide Streamlit branding */
    #MainMenu, footer { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
DATA_PATH = "Housing_cleaned.csv"
MODEL_PATH = "best_model.pkl"
SCALER_PATH = "scaler.pkl"

BINARY_COLS = [
    "mainroad", "guestroom", "basement",
    "hotwaterheating", "airconditioning", "prefarea"
]
FURNISHING_MAP = {0: "Unfurnished", 1: "Semi-Furnished", 2: "Furnished"}
FURNISHING_MAP_INV = {v: k for k, v in FURNISHING_MAP.items()}

FEATURE_COLS = [
    "area", "bedrooms", "bathrooms", "stories", "parking",
    "mainroad", "guestroom", "basement", "hotwaterheating",
    "airconditioning", "prefarea", "furnishingstatus"
]

FEATURE_LABELS = {
    "area": "Area (sq ft)",
    "bedrooms": "Bedrooms",
    "bathrooms": "Bathrooms",
    "stories": "Stories",
    "parking": "Parking Spaces",
    "mainroad": "Main Road Access",
    "guestroom": "Guest Room",
    "basement": "Basement",
    "hotwaterheating": "Hot Water Heating",
    "airconditioning": "Air Conditioning",
    "prefarea": "Preferred Area",
    "furnishingstatus": "Furnishing Status",
}

# ─────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["furnishing_label"] = df["furnishingstatus"].map(FURNISHING_MAP)
    df["price_lakh"] = df["price"] / 1e5
    df["price_per_sqft"] = df["price"] / df["area"]
    return df

# ─────────────────────────────────────────────────────────────
# MODEL TRAINING
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def train_models(df):
    X = df[FEATURE_COLS].copy()
    y = df["price"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    models = {
        "Linear Regression":    LinearRegression(),
        "Ridge Regression":     Ridge(alpha=10),
        "Lasso Regression":     Lasso(alpha=1000, max_iter=10000),
        "Decision Tree":        DecisionTreeRegressor(max_depth=6, random_state=42),
        "Random Forest":        RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
        "Gradient Boosting":    GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42, verbosity=0)

    results = {}
    trained  = {}

    for name, model in models.items():
        # Linear models benefit from scaling
        use_scaled = name in ("Linear Regression", "Ridge Regression", "Lasso Regression")
        Xtr = X_train_sc if use_scaled else X_train.values
        Xte = X_test_sc  if use_scaled else X_test.values

        model.fit(Xtr, y_train)
        preds = model.predict(Xte)

        mae  = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2   = r2_score(y_test, preds)
        mape = mean_absolute_percentage_error(y_test, preds) * 100

        # 5-fold CV R²
        cv = cross_val_score(model, Xtr, y_train, cv=5, scoring="r2")

        results[name] = {
            "MAE":     mae,
            "RMSE":    rmse,
            "R²":      r2,
            "MAPE (%)": mape,
            "CV R² Mean": cv.mean(),
            "CV R² Std":  cv.std(),
        }
        trained[name] = (model, use_scaled)

    # Best model by test R²
    best_name = max(results, key=lambda n: results[n]["R²"])
    best_model, best_uses_scale = trained[best_name]

    return trained, results, best_name, scaler, X_test, y_test, X_train, y_train

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def fmt_price(val):
    if val >= 1e7:
        return f"₹ {val/1e7:.2f} Cr"
    elif val >= 1e5:
        return f"₹ {val/1e5:.2f} L"
    return f"₹ {val:,.0f}"

def predict_price(model, scaler, use_scaled, input_dict):
    row = pd.DataFrame([input_dict])[FEATURE_COLS]
    if use_scaled:
        row_t = scaler.transform(row)
    else:
        row_t = row.values
    return model.predict(row_t)[0]

PLOTLY_TEMPLATE = "plotly_white"
ACCENT = "#2563eb"
PALETTE = px.colors.qualitative.Bold

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
def render_sidebar(df, results, best_name, trained):
    with st.sidebar:
        st.markdown("## 🏠 House Price App")
        st.markdown("---")

        pages = [
            "🏠 Dashboard",
            "📊 Data Explorer",
            "🤖 Model Performance",
            "🔮 Price Predictor",
            "📈 Feature Insights",
            "📋 Data Table",
        ]
        page = st.radio("Navigate", pages, label_visibility="collapsed")

        st.markdown("---")
        st.markdown("### 🏆 Best Model")
        st.success(best_name)
        r2 = results[best_name]["R²"]
        st.metric("Test R²", f"{r2:.4f}")
        st.metric("MAPE", f"{results[best_name]['MAPE (%)']:.2f}%")

        st.markdown("---")
        st.markdown("### ⚙️ Active Model")
        chosen = st.selectbox(
            "Select model for prediction",
            list(trained.keys()),
            index=list(trained.keys()).index(best_name),
            key="model_select",
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown(
            "<small style='color:#94a3b8'>📂 Dataset: Housing_cleaned.csv<br>"
            f"🔢 Rows: {len(df)} | Features: {len(FEATURE_COLS)}</small>",
            unsafe_allow_html=True,
        )

    return page.split(" ", 1)[1].strip(), chosen

# ─────────────────────────────────────────────────────────────
# PAGE: DASHBOARD
# ─────────────────────────────────────────────────────────────
def page_dashboard(df, results, best_name):
    st.markdown('<p class="section-header">📊 Dataset Overview</p>', unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Properties", len(df))
    col2.metric("Avg Price", fmt_price(df["price"].mean()))
    col3.metric("Median Price", fmt_price(df["price"].median()))
    col4.metric("Avg Area", f"{df['area'].mean():,.0f} sq ft")
    col5.metric("Best R²", f"{results[best_name]['R²']:.4f}")

    st.markdown("---")

    # Row 1: Price distribution + Furnishing breakdown
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<p class="section-header">Price Distribution</p>', unsafe_allow_html=True)
        fig = px.histogram(
            df, x="price_lakh", nbins=40,
            labels={"price_lakh": "Price (₹ Lakh)"},
            color_discrete_sequence=[ACCENT],
            template=PLOTLY_TEMPLATE,
        )
        fig.update_traces(marker_line_color="white", marker_line_width=0.5)
        fig.update_layout(showlegend=False, height=320, margin=dict(t=10, b=40))
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown('<p class="section-header">Properties by Furnishing</p>', unsafe_allow_html=True)
        counts = df["furnishing_label"].value_counts().reset_index()
        counts.columns = ["Furnishing", "Count"]
        fig = px.pie(
            counts, names="Furnishing", values="Count",
            color_discrete_sequence=PALETTE,
            template=PLOTLY_TEMPLATE, hole=0.45,
        )
        fig.update_layout(height=320, margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

    # Row 2: Area vs Price + Avg price by bedrooms
    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown('<p class="section-header">Area vs Price</p>', unsafe_allow_html=True)
        fig = px.scatter(
            df, x="area", y="price_lakh",
            color="furnishing_label",
            labels={"area": "Area (sq ft)", "price_lakh": "Price (₹ Lakh)", "furnishing_label": "Furnishing"},
            color_discrete_sequence=PALETTE,
            template=PLOTLY_TEMPLATE,
            opacity=0.75,
        )
        fig.update_traces(marker=dict(size=6))
        fig.update_layout(height=320, margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

    with col_d:
        st.markdown('<p class="section-header">Avg Price by Bedrooms</p>', unsafe_allow_html=True)
        avg_bed = df.groupby("bedrooms")["price_lakh"].mean().reset_index()
        fig = px.bar(
            avg_bed, x="bedrooms", y="price_lakh",
            labels={"bedrooms": "Bedrooms", "price_lakh": "Avg Price (₹ Lakh)"},
            color="price_lakh",
            color_continuous_scale="Blues",
            template=PLOTLY_TEMPLATE,
        )
        fig.update_layout(height=320, margin=dict(t=10), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    # Row 3: Model comparison bar
    st.markdown('<p class="section-header">Model R² Comparison</p>', unsafe_allow_html=True)
    model_df = pd.DataFrame(results).T.reset_index().rename(columns={"index": "Model"})
    fig = px.bar(
        model_df.sort_values("R²", ascending=False),
        x="Model", y="R²",
        color="R²",
        color_continuous_scale="Blues",
        text_auto=".4f",
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(height=320, margin=dict(t=10), coloraxis_showscale=False)
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# PAGE: DATA EXPLORER
# ─────────────────────────────────────────────────────────────
def page_data_explorer(df):
    st.markdown('<p class="section-header">🔍 Explore the Dataset</p>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📈 Distributions", "🔗 Correlations", "📦 Box Plots", "🌡️ Heat Map"]
    )

    with tab1:
        col = st.selectbox("Select feature", FEATURE_COLS + ["price", "price_per_sqft"], key="dist_col")
        fig = px.histogram(
            df, x=col, nbins=35,
            color_discrete_sequence=[ACCENT],
            template=PLOTLY_TEMPLATE,
            marginal="box",
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        # Stats
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mean",   f"{df[col].mean():,.2f}")
        c2.metric("Median", f"{df[col].median():,.2f}")
        c3.metric("Std Dev",f"{df[col].std():,.2f}")
        c4.metric("Range",  f"{df[col].max() - df[col].min():,.2f}")

    with tab2:
        x_col = st.selectbox("X-axis", FEATURE_COLS, key="sc_x")
        y_col = st.selectbox("Y-axis", FEATURE_COLS + ["price"], index=len(FEATURE_COLS), key="sc_y")
        hue   = st.selectbox("Color by", ["furnishing_label", "bedrooms", "stories", "airconditioning"], key="sc_hue")
        fig = px.scatter(
            df, x=x_col, y=y_col, color=hue,
            trendline="ols",
            color_discrete_sequence=PALETTE,
            template=PLOTLY_TEMPLATE,
            opacity=0.8,
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        cat_col = st.selectbox(
            "Group by", ["bedrooms", "bathrooms", "stories", "parking", "furnishing_label", "airconditioning"],
            key="box_cat"
        )
        fig = px.box(
            df, x=cat_col, y="price_lakh",
            color=cat_col,
            color_discrete_sequence=PALETTE,
            template=PLOTLY_TEMPLATE,
            labels={"price_lakh": "Price (₹ Lakh)"},
        )
        fig.update_layout(height=420, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        num_cols = ["price", "area", "bedrooms", "bathrooms", "stories", "parking",
                    "mainroad", "guestroom", "basement", "hotwaterheating",
                    "airconditioning", "prefarea", "furnishingstatus"]
        corr = df[num_cols].corr()
        fig = px.imshow(
            corr,
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            text_auto=".2f",
            template=PLOTLY_TEMPLATE,
            aspect="auto",
        )
        fig.update_layout(height=550)
        st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# PAGE: MODEL PERFORMANCE
# ─────────────────────────────────────────────────────────────
def page_model_performance(trained, results, scaler, X_test, y_test, X_train, y_train):
    st.markdown('<p class="section-header">🤖 Model Performance</p>', unsafe_allow_html=True)

    # Metrics table
    metrics_df = pd.DataFrame(results).T
    metrics_df = metrics_df.round(4)
    metrics_df.index.name = "Model"
    metrics_df = metrics_df.reset_index()
    metrics_df = metrics_df.sort_values("R²", ascending=False)

    def highlight_best(s):
        if s.name in ("R²", "CV R² Mean"):
            best_val = s.max()
            return ["background-color: #dbeafe; font-weight:bold" if v == best_val else "" for v in s]
        elif s.name in ("MAE", "RMSE", "MAPE (%)", "CV R² Std"):
            best_val = s.min()
            return ["background-color: #dcfce7; font-weight:bold" if v == best_val else "" for v in s]
        return [""] * len(s)

    st.dataframe(
        metrics_df.style.apply(highlight_best),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="section-header">R² Comparison</p>', unsafe_allow_html=True)
        fig = px.bar(
            metrics_df, x="Model", y="R²",
            color="R²", color_continuous_scale="Blues",
            text_auto=".4f", template=PLOTLY_TEMPLATE,
        )
        fig.update_layout(height=340, coloraxis_showscale=False)
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<p class="section-header">MAPE (%) Comparison</p>', unsafe_allow_html=True)
        fig = px.bar(
            metrics_df, x="Model", y="MAPE (%)",
            color="MAPE (%)", color_continuous_scale="Reds_r",
            text_auto=".2f", template=PLOTLY_TEMPLATE,
        )
        fig.update_layout(height=340, coloraxis_showscale=False)
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    # Actual vs Predicted for each model
    st.markdown('<p class="section-header">Actual vs Predicted</p>', unsafe_allow_html=True)
    chosen = st.selectbox("Choose model", list(trained.keys()), key="avp_model")
    model, use_scaled = trained[chosen]
    X_te = scaler.transform(X_test) if use_scaled else X_test.values
    preds = model.predict(X_te)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_test.values / 1e5, y=preds / 1e5,
        mode="markers", marker=dict(color=ACCENT, size=6, opacity=0.7),
        name="Predictions",
    ))
    mn = min(y_test.min(), preds.min()) / 1e5
    mx = max(y_test.max(), preds.max()) / 1e5
    fig.add_trace(go.Scatter(
        x=[mn, mx], y=[mn, mx],
        mode="lines", line=dict(color="red", dash="dash", width=2),
        name="Perfect Fit",
    ))
    fig.update_layout(
        xaxis_title="Actual Price (₹ Lakh)",
        yaxis_title="Predicted Price (₹ Lakh)",
        template=PLOTLY_TEMPLATE,
        height=420,
        legend=dict(x=0.02, y=0.95),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Residuals
    st.markdown('<p class="section-header">Residuals Analysis</p>', unsafe_allow_html=True)
    residuals = y_test.values - preds
    col_r1, col_r2 = st.columns(2)

    with col_r1:
        fig_res = px.histogram(
            x=residuals / 1e5, nbins=30,
            labels={"x": "Residual (₹ Lakh)"},
            color_discrete_sequence=[ACCENT],
            template=PLOTLY_TEMPLATE,
        )
        fig_res.add_vline(x=0, line_dash="dash", line_color="red")
        fig_res.update_layout(height=320, title="Residual Distribution")
        st.plotly_chart(fig_res, use_container_width=True)

    with col_r2:
        fig_rv = px.scatter(
            x=preds / 1e5, y=residuals / 1e5,
            labels={"x": "Predicted (₹ Lakh)", "y": "Residual"},
            color_discrete_sequence=[ACCENT],
            template=PLOTLY_TEMPLATE,
            opacity=0.7,
        )
        fig_rv.add_hline(y=0, line_dash="dash", line_color="red")
        fig_rv.update_layout(height=320, title="Residuals vs Predicted")
        st.plotly_chart(fig_rv, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# PAGE: PRICE PREDICTOR
# ─────────────────────────────────────────────────────────────
def page_price_predictor(df, trained, scaler, chosen_model):
    st.markdown('<p class="section-header">🔮 Predict House Price</p>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="info-card">Using model: <strong>{chosen_model}</strong></div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 🏗️ Physical Features")
        area       = st.slider("Area (sq ft)", int(df.area.min()), int(df.area.max()), 6000, 100)
        bedrooms   = st.selectbox("Bedrooms", sorted(df.bedrooms.unique()), index=1)
        bathrooms  = st.selectbox("Bathrooms", sorted(df.bathrooms.unique()), index=0)
        stories    = st.selectbox("Stories", sorted(df.stories.unique()), index=0)
        parking    = st.selectbox("Parking Spaces", sorted(df.parking.unique()), index=1)

    with col2:
        st.markdown("#### 🏠 Amenities")
        mainroad        = st.radio("Main Road Access",    ["Yes", "No"], horizontal=True) == "Yes"
        guestroom       = st.radio("Guest Room",          ["Yes", "No"], horizontal=True) == "Yes"
        basement        = st.radio("Basement",            ["Yes", "No"], horizontal=True) == "Yes"
        hotwaterheating = st.radio("Hot Water Heating",   ["Yes", "No"], horizontal=True) == "Yes"

    with col3:
        st.markdown("#### ⭐ Preferences & Finish")
        airconditioning = st.radio("Air Conditioning",   ["Yes", "No"], horizontal=True) == "Yes"
        prefarea        = st.radio("Preferred Area",     ["Yes", "No"], horizontal=True) == "Yes"
        furnishing_sel  = st.selectbox(
            "Furnishing Status",
            ["Unfurnished", "Semi-Furnished", "Furnished"],
            index=1
        )
        furnishingstatus = FURNISHING_MAP_INV[furnishing_sel]

    st.markdown("---")

    if st.button("🏠 Predict Price", use_container_width=True, type="primary"):
        input_dict = {
            "area":            area,
            "bedrooms":        bedrooms,
            "bathrooms":       bathrooms,
            "stories":         stories,
            "parking":         parking,
            "mainroad":        int(mainroad),
            "guestroom":       int(guestroom),
            "basement":        int(basement),
            "hotwaterheating": int(hotwaterheating),
            "airconditioning": int(airconditioning),
            "prefarea":        int(prefarea),
            "furnishingstatus": furnishingstatus,
        }

        model, use_scaled = trained[chosen_model]
        predicted = predict_price(model, scaler, use_scaled, input_dict)
        predicted = max(predicted, 0)

        st.markdown(
            f"""
            <div class="pred-box">
                <p>Estimated House Price</p>
                <h1>{fmt_price(predicted)}</h1>
                <p>₹ {predicted:,.0f} &nbsp;|&nbsp; ₹ {predicted/area:,.0f} per sq ft</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Range estimate (±10%)
        lo, hi = predicted * 0.90, predicted * 1.10
        st.markdown(
            f'<div class="info-card">📊 Estimated range: <strong>{fmt_price(lo)}</strong>'
            f' – <strong>{fmt_price(hi)}</strong> (±10%)</div>',
            unsafe_allow_html=True,
        )

        # Comparable properties
        st.markdown('<p class="section-header">📋 Similar Properties in Dataset</p>', unsafe_allow_html=True)
        margin = 0.20
        sim = df[
            (df["area"].between(area * (1 - margin), area * (1 + margin))) &
            (df["bedrooms"] == bedrooms)
        ].sort_values("price").head(10)

        if not sim.empty:
            show_cols = ["price_lakh", "area", "bedrooms", "bathrooms", "stories", "parking", "furnishing_label"]
            st.dataframe(
                sim[show_cols].rename(columns={
                    "price_lakh": "Price (₹L)", "furnishing_label": "Furnishing"
                }).reset_index(drop=True),
                use_container_width=True,
            )
        else:
            st.info("No closely matching properties found in the dataset.")

# ─────────────────────────────────────────────────────────────
# PAGE: FEATURE INSIGHTS
# ─────────────────────────────────────────────────────────────
def page_feature_insights(df, trained, scaler, X_train, y_train):
    st.markdown('<p class="section-header">📈 Feature Insights</p>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🌟 Feature Importance", "💡 Price Drivers", "🔬 What-If Analysis"])

    with tab1:
        model_name = st.selectbox(
            "Select model", [k for k in trained if k not in ("Linear Regression", "Ridge Regression", "Lasso Regression")],
            key="fi_model"
        )
        model, use_scaled = trained[model_name]
        Xtr = scaler.transform(X_train) if use_scaled else X_train.values

        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
            fi_df = pd.DataFrame({"Feature": FEATURE_COLS, "Importance": imp})
        else:
            result = permutation_importance(model, Xtr, y_train, n_repeats=15, random_state=42, n_jobs=-1)
            fi_df = pd.DataFrame({"Feature": FEATURE_COLS, "Importance": result.importances_mean})

        fi_df = fi_df.sort_values("Importance", ascending=True)
        fi_df["Feature"] = fi_df["Feature"].map(lambda c: FEATURE_LABELS.get(c, c))

        fig = px.bar(
            fi_df, x="Importance", y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale="Blues",
            template=PLOTLY_TEMPLATE,
            text_auto=".4f",
        )
        fig.update_layout(height=480, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("#### Avg Price by Binary Features")
        rows = []
        for col in BINARY_COLS:
            yes_avg = df[df[col] == 1]["price_lakh"].mean()
            no_avg  = df[df[col] == 0]["price_lakh"].mean()
            rows.append({"Feature": FEATURE_LABELS[col], "With Feature": yes_avg, "Without Feature": no_avg})
        drv_df = pd.DataFrame(rows)
        drv_df["Premium (₹ L)"] = drv_df["With Feature"] - drv_df["Without Feature"]

        fig = go.Figure()
        fig.add_trace(go.Bar(name="With Feature",    x=drv_df["Feature"], y=drv_df["With Feature"],    marker_color="#2563eb"))
        fig.add_trace(go.Bar(name="Without Feature", x=drv_df["Feature"], y=drv_df["Without Feature"], marker_color="#93c5fd"))
        fig.update_layout(
            barmode="group",
            template=PLOTLY_TEMPLATE,
            height=380,
            yaxis_title="Avg Price (₹ Lakh)",
            legend=dict(x=0.02, y=0.95),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Feature Price Premium")
        fig2 = px.bar(
            drv_df.sort_values("Premium (₹ L)", ascending=False),
            x="Feature", y="Premium (₹ L)",
            color="Premium (₹ L)",
            color_continuous_scale="RdYlGn",
            text_auto=".1f",
            template=PLOTLY_TEMPLATE,
        )
        fig2.update_layout(height=320, coloraxis_showscale=False)
        fig2.update_traces(textposition="outside")
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.markdown("#### Sensitivity: How one feature affects the predicted price")
        vary_col = st.selectbox(
            "Feature to vary", ["area", "bedrooms", "bathrooms", "stories", "parking"],
            key="wi_col"
        )
        model_name_wi = st.selectbox("Model", list(trained.keys()), key="wi_model")
        model_wi, use_scaled_wi = trained[model_name_wi]

        # Default inputs (medians)
        base = {c: int(df[c].median()) for c in FEATURE_COLS}
        vary_range = {
            "area":     np.linspace(df.area.min(), df.area.max(), 60),
            "bedrooms": sorted(df.bedrooms.unique()),
            "bathrooms":sorted(df.bathrooms.unique()),
            "stories":  sorted(df.stories.unique()),
            "parking":  sorted(df.parking.unique()),
        }[vary_col]

        preds_wi = []
        for val in vary_range:
            inp = base.copy()
            inp[vary_col] = val
            row = pd.DataFrame([inp])[FEATURE_COLS]
            Xp = scaler.transform(row) if use_scaled_wi else row.values
            preds_wi.append(model_wi.predict(Xp)[0])

        fig_wi = px.line(
            x=vary_range, y=np.array(preds_wi) / 1e5,
            labels={"x": FEATURE_LABELS[vary_col], "y": "Predicted Price (₹ Lakh)"},
            template=PLOTLY_TEMPLATE,
            markers=True,
            color_discrete_sequence=[ACCENT],
        )
        fig_wi.update_layout(height=380)
        st.plotly_chart(fig_wi, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# PAGE: DATA TABLE
# ─────────────────────────────────────────────────────────────
def page_data_table(df):
    st.markdown('<p class="section-header">📋 Full Dataset</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        price_range = st.slider(
            "Price Range (₹ Lakh)",
            float(df.price_lakh.min()), float(df.price_lakh.max()),
            (float(df.price_lakh.min()), float(df.price_lakh.max())),
        )
    with col2:
        bed_filter = st.multiselect("Bedrooms", sorted(df.bedrooms.unique()), default=sorted(df.bedrooms.unique()))
    with col3:
        furn_filter = st.multiselect(
            "Furnishing", df.furnishing_label.unique().tolist(),
            default=df.furnishing_label.unique().tolist()
        )

    filtered = df[
        (df.price_lakh.between(*price_range)) &
        (df.bedrooms.isin(bed_filter)) &
        (df.furnishing_label.isin(furn_filter))
    ]

    show = filtered[[
        "price", "price_lakh", "price_per_sqft", "area",
        "bedrooms", "bathrooms", "stories", "parking",
        "mainroad", "guestroom", "basement", "hotwaterheating",
        "airconditioning", "prefarea", "furnishing_label"
    ]].rename(columns={
        "price": "Price (₹)",
        "price_lakh": "Price (₹L)",
        "price_per_sqft": "₹/sqft",
        "furnishing_label": "Furnishing",
    })

    st.write(f"**{len(filtered)}** properties match your filters.")
    st.dataframe(show.reset_index(drop=True), use_container_width=True, height=500)

    # Download
    csv_bytes = show.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Filtered CSV",
        csv_bytes,
        "filtered_houses.csv",
        "text/csv",
        use_container_width=True,
    )

    # Summary stats
    st.markdown('<p class="section-header">Summary Statistics</p>', unsafe_allow_html=True)
    st.dataframe(filtered[["price_lakh", "area", "bedrooms", "bathrooms", "stories", "parking"]].describe().round(2), use_container_width=True)

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    # Load data
    if not os.path.exists(DATA_PATH):
        st.error(f"Dataset not found: `{DATA_PATH}`. Please place it in the same directory as app.py.")
        st.stop()

    df = load_data()

    # Train models (cached)
    with st.spinner("Training machine learning models…"):
        trained, results, best_name, scaler, X_test, y_test, X_train, y_train = train_models(df)

    # Sidebar + navigation
    page, chosen_model = render_sidebar(df, results, best_name, trained)

    # Header
    st.markdown(
        f"""
        <div style="background:linear-gradient(135deg,#1e3a5f,#2563eb);
                    padding:22px 30px; border-radius:14px; margin-bottom:18px;
                    display:flex; align-items:center; justify-content:space-between;">
            <div>
                <h2 style="color:white;margin:0;font-size:1.7rem;">🏠 House Price Prediction & Analytics</h2>
                <p style="color:#bfdbfe;margin:4px 0 0 0;font-size:0.95rem;">
                    ML-powered insights &nbsp;|&nbsp; {len(df)} properties &nbsp;|&nbsp; {len(FEATURE_COLS)} features
                </p>
            </div>
            <div style="text-align:right;">
                <span style="background:#1d4ed8;color:white;padding:6px 14px;border-radius:20px;font-size:0.85rem;">
                    Best: {best_name} &nbsp;•&nbsp; R² {results[best_name]['R²']:.4f}
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Route pages
    if page == "Dashboard":
        page_dashboard(df, results, best_name)
    elif page == "Data Explorer":
        page_data_explorer(df)
    elif page == "Model Performance":
        page_model_performance(trained, results, scaler, X_test, y_test, X_train, y_train)
    elif page == "Price Predictor":
        page_price_predictor(df, trained, scaler, chosen_model)
    elif page == "Feature Insights":
        page_feature_insights(df, trained, scaler, X_train, y_train)
    elif page == "Data Table":
        page_data_table(df)


if __name__ == "__main__":
    main()
