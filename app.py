"""
app.py - Crop Yield Predictor
"""
import streamlit as st
import pandas as pd
import numpy as np
import shap
import plotly.express as px
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

TARGET = "yield_quintal_hectare"
CATEGORICAL_FEATURES = ["crop", "state"]
NUMERIC_FEATURES = [
    "cost_of_cultivation_`_hectare_a2+fl",
    "cost_of_cultivation_`_hectare_c2",
    "cost_of_production_`_quintal_c2",
]

@st.cache_data
def load_data():
    return pd.read_csv("data/cleaned_crop_data.csv")

@st.cache_resource
def train_model(df):
    X = df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    y = df[TARGET]
    preprocessor = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES)],
        remainder="passthrough",
    )
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)),
    ])
    pipeline.fit(X, y)
    return pipeline

@st.cache_resource
def get_shap_explainer(_model):
    return shap.TreeExplainer(_model.named_steps["model"])

def show_predict_tab(df, model):
    st.header("Enter details")
    col1, col2 = st.columns(2)
    with col1:
        crop = st.selectbox("Crop", sorted(df["crop"].unique()), key="p_crop")
    valid_states = sorted(df[df["crop"] == crop]["state"].unique())
    with col2:
        state = st.selectbox("State", valid_states, key="p_state")
        st.caption(f"Showing states where {crop} appears in the training data.")

    st.subheader("Cost details (Rs)")
    cost_a2fl = st.slider("Cost of Cultivation - A2+FL (Rs/Hectare)", 0.0, 60000.0, 15000.0, 500.0, key="p_a2fl")
    cost_c2 = st.slider("Cost of Cultivation - C2 (Rs/Hectare)", 0.0, 90000.0, 25000.0, 500.0, key="p_c2")
    cost_prod = st.slider("Cost of Production - C2 (Rs/Quintal)", 0.0, 5000.0, 1500.0, 100.0, key="p_prod")

    input_df = pd.DataFrame([{
        "crop": crop, "state": state,
        "cost_of_cultivation_`_hectare_a2+fl": cost_a2fl,
        "cost_of_cultivation_`_hectare_c2": cost_c2,
        "cost_of_production_`_quintal_c2": cost_prod,
    }])

    live_pred = model.predict(input_df)[0]
    st.metric("Live Preview", f"{live_pred:.2f} Quintal/Hectare")
    st.caption("Updates instantly as you move the sliders. Click below for the full prediction with explanation.")

    if st.button("Predict Yield", type="primary", key="p_btn"):
        st.success(f"### Predicted Yield: {live_pred:.2f} Quintal/Hectare")

        explainer = get_shap_explainer(model)
        preprocessor = model.named_steps["preprocessor"]
        X_t = preprocessor.transform(input_df)
        if hasattr(X_t, "toarray"):
            X_t = X_t.toarray()
        X_t = X_t.astype(float)

        cat_enc = preprocessor.named_transformers_["cat"]
        feat_names = list(cat_enc.get_feature_names_out(CATEGORICAL_FEATURES)) + NUMERIC_FEATURES
        sv = explainer.shap_values(X_t)
        if isinstance(sv, list):
            sv = sv[0]
        sv = np.array(sv).flatten()

        bv = explainer.expected_value
        if isinstance(bv, (list, np.ndarray)):
            bv = bv[0]
        bv = float(bv)

        if len(sv) == len(feat_names):
            imp = pd.DataFrame({"feature": feat_names, "shap_value": sv})
            sel_crop = f"crop_{crop}"
            sel_state = f"state_{state}"
            imp = imp[imp["feature"].apply(
                lambda f: f in (sel_crop, sel_state) or (not f.startswith("crop_") and not f.startswith("state_"))
            )].copy()
            imp["abs"] = imp["shap_value"].abs()
            top = imp.sort_values("abs", ascending=False).head(3)

            with st.expander("🔍 Why this prediction?"):
                st.caption(f"Starting from average yield of {bv:.2f}, here's what pushed this prediction:")
                for _, r in top.iterrows():
                    f, v = r["feature"], r["shap_value"]
                    lbl = f"Growing **{crop}**" if f == sel_crop else (f"Being grown in **{state}**" if f == sel_state else f"**{f.replace('_',' ')}**")
                    st.write(f"- {lbl} {'increased ⬆️' if v > 0 else 'decreased ⬇️'} prediction by **{abs(v):.2f}** Q/Ha")

        crop_avg = df[df["crop"] == crop][TARGET].mean()
        st.caption(f"Historical average yield for {crop}: {crop_avg:.2f} Quintal/Hectare")
        if not ((df["crop"] == crop) & (df["state"] == state)).any():
            st.warning("⚠️ This crop-state combination was not in training data. Prediction may be less reliable.")

    st.divider()
    st.caption("Model: Random Forest Regressor (tuned via GridSearchCV) | 49 records | data.gov.in")

def show_recommend_tab(df, model):
    st.header("🌱 Crop Recommendation")
    st.write("Enter your state and cost details to find which crop gives the best yield.")

    rec_state = st.selectbox("State", sorted(df["state"].unique()), key="r_state")
    st.subheader("Your Cost Details (Rs)")
    r_a2fl = st.slider("Cost of Cultivation - A2+FL (Rs/Hectare)", 0.0, 60000.0, 15000.0, 500.0, key="r_a2fl")
    r_c2 = st.slider("Cost of Cultivation - C2 (Rs/Hectare)", 0.0, 90000.0, 25000.0, 500.0, key="r_c2")
    r_prod = st.slider("Cost of Production - C2 (Rs/Quintal)", 0.0, 5000.0, 1500.0, 100.0, key="r_prod")

    crops = sorted(df[df["state"] == rec_state]["crop"].unique())
    if not crops:
        st.warning("No crops found for this state.")
        return

    rows = []
    for c in crops:
        inp = pd.DataFrame([{
            "crop": c, "state": rec_state,
            "cost_of_cultivation_`_hectare_a2+fl": r_a2fl,
            "cost_of_cultivation_`_hectare_c2": r_c2,
            "cost_of_production_`_quintal_c2": r_prod,
        }])
        rows.append({
            "Crop": c,
            "Predicted Yield (Q/Ha)": round(model.predict(inp)[0], 2),
            "Historical Avg (Q/Ha)": round(df[df["crop"] == c][TARGET].mean(), 2),
        })

    res = pd.DataFrame(rows).sort_values("Predicted Yield (Q/Ha)", ascending=False).reset_index(drop=True)
    st.success(f"### 🏆 Best Crop for {rec_state}: **{res.iloc[0]['Crop']}** — {res.iloc[0]['Predicted Yield (Q/Ha)']:.2f} Q/Ha")

    st.subheader("All Crops Ranked")
    st.dataframe(
        res.style.apply(lambda r: ["background-color: #d4edda" if r.name == 0 else "" for _ in r], axis=1),
        use_container_width=True, hide_index=True,
    )

    fig = px.bar(res, x="Crop", y="Predicted Yield (Q/Ha)", color="Predicted Yield (Q/Ha)",
                 color_continuous_scale=["#8B5A2B", "#C9A227", "#2D5016"],
                 title=f"Predicted Yield by Crop — {rec_state}")
    fig.update_layout(plot_bgcolor="#FAF7F0", paper_bgcolor="#FAF7F0", coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Only crops present in {rec_state}'s training data are shown.")

def show_insights_tab(df):
    st.header("Model Evaluation")
    try:
        cdf = pd.read_csv("outputs/cv_model_comparison.csv")
        ddf = cdf.rename(columns={"model":"Model","rmse_mean":"RMSE","mae_mean":"MAE","r2_mean":"R²","r2_std":"R² Std Dev"})[["Model","RMSE","MAE","R²","R² Std Dev"]]
        bi = ddf["MAE"].idxmin()
        st.dataframe(
            ddf.style.apply(lambda r: ["background-color: #d4edda" if r.name == bi else "" for _ in r], axis=1)
            .format({"RMSE":"{:.2f}","MAE":"{:.2f}","R²":"{:.3f}","R² Std Dev":"{:.3f}"}),
            use_container_width=True, hide_index=True,
        )
        st.caption("Highlighted = best model by MAE.")
    except Exception:
        st.warning("Could not load model comparison data.")

    st.divider()

    st.subheader("Yield Distribution by Crop")
    order = df.groupby("crop")["yield_quintal_hectare"].mean().sort_values(ascending=False).index.tolist()
    fig1 = px.box(df, x="crop", y="yield_quintal_hectare", category_orders={"crop": order}, color="crop",
                  labels={"yield_quintal_hectare": "Yield (Q/Ha)", "crop": "Crop"},
                  title="Yield by Crop — hover for values")
    fig1.update_layout(showlegend=False, plot_bgcolor="#FAF7F0", paper_bgcolor="#FAF7F0")
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Average Yield by State")
    ay = df.groupby("state")["yield_quintal_hectare"].mean().sort_values(ascending=False).reset_index()
    ay.columns = ["State", "Avg Yield (Q/Ha)"]
    fig2 = px.bar(ay, x="Avg Yield (Q/Ha)", y="State", orientation="h",
                  color="Avg Yield (Q/Ha)", color_continuous_scale=["#8B5A2B","#C9A227","#2D5016"],
                  title="Average Yield by State")
    fig2.update_layout(plot_bgcolor="#FAF7F0", paper_bgcolor="#FAF7F0", coloraxis_showscale=False)
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Cost of Cultivation vs Yield")
    fig3 = px.scatter(df, x="cost_of_cultivation_`_hectare_c2", y="yield_quintal_hectare",
                      color="crop", hover_data=["state","crop"],
                      labels={"cost_of_cultivation_`_hectare_c2":"Cost C2 (Rs/Ha)","yield_quintal_hectare":"Yield (Q/Ha)"},
                      title="Cost vs Yield — click legend to filter")
    fig3.update_layout(plot_bgcolor="#FAF7F0", paper_bgcolor="#FAF7F0")
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Correlation Heatmap")
    corr = df.select_dtypes(include="number").corr().round(2)
    fig4 = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                     title="Correlation Heatmap")
    fig4.update_layout(plot_bgcolor="#FAF7F0", paper_bgcolor="#FAF7F0")
    st.plotly_chart(fig4, use_container_width=True)

    st.subheader("Additional Charts")
    for path, cap in [
        ("outputs/feature_importance.png", "Feature Importance (Random Forest)"),
        ("outputs/shap_summary.png", "SHAP Summary"),
        ("outputs/actual_vs_predicted.png", "Actual vs Predicted"),
    ]:
        try:
            st.image(path, caption=cap, use_container_width=True)
        except Exception:
            st.warning(f"Could not load: {path}")

def main():
    st.set_page_config(page_title="Crop Yield Predictor", page_icon="🌾", layout="centered")
    st.markdown("""<style>
        h1{color:#2D5016!important;font-weight:800!important;}
        h2,h3{color:#2D5016!important;font-weight:700!important;}
        [data-testid="stMetricValue"]{color:#8B5A2B!important;font-size:2.5rem!important;font-weight:800!important;}
        [data-testid="stMetricLabel"]{color:#2D5016!important;font-weight:600!important;text-transform:uppercase;letter-spacing:1px;font-size:0.8rem!important;}
        .stTabs [data-baseweb="tab"]{font-weight:600;}
        div[data-testid="stExpander"]{border:1px solid #C9A227!important;border-radius:8px!important;}
        div[data-testid="stSlider"]>div>div>div{background-color:#D4CBB5!important;}
    </style>""", unsafe_allow_html=True)

    st.title("🌾 Crop Yield Prediction (India)")
    st.write("Estimate crop yield (Quintal/Hectare) based on crop type, state, and cultivation/production costs. Powered by a Random Forest model trained on government agricultural cost data.")

    df = load_data()
    model = train_model(df)

    # Session state tab tracking
    if "tab" not in st.session_state:
        st.session_state.tab = "predict"

    tab1, tab2, tab3 = st.tabs(["🔮 Predict Yield", "🌱 Recommend Crop", "📊 Data Insights"])

    with tab1:
        if st.session_state.get("_active_tab", "predict") == "predict" or True:
            show_predict_tab(df, model)

    with tab2:
        show_recommend_tab(df, model)

    with tab3:
        show_insights_tab(df)

if __name__ == "__main__":
    main()