"""
app.py
------
Interactive Streamlit app for crop yield prediction.

Run:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np

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
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ],
        remainder="passthrough",
    )
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        # Using the tuned hyperparameters found via GridSearchCV (see train_model.py)
        ("model", RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)),
    ])
    pipeline.fit(X, y)
    return pipeline

def run_prediction_tab(df, model):
    st.header("Enter details")

    col1, col2 = st.columns(2)
    with col1:
        crop = st.selectbox("Crop", sorted(df["crop"].unique()))

    valid_states = sorted(df[df["crop"] == crop]["state"].unique())
    with col2:
        state = st.selectbox("State", valid_states)
        st.caption(f"Showing states where {crop} appears in the training data.")

    st.subheader("Cost details (Rs)")
    cost_a2fl = st.number_input(
        "Cost of Cultivation - A2+FL (Rs/Hectare)", min_value=0.0, value=15000.0, step=500.0
    )
    cost_c2 = st.number_input(
        "Cost of Cultivation - C2 (Rs/Hectare)", min_value=0.0, value=25000.0, step=500.0
    )
    cost_production = st.number_input(
        "Cost of Production - C2 (Rs/Quintal)", min_value=0.0, value=1500.0, step=100.0
    )

    if st.button("Predict Yield", type="primary"):
        input_df = pd.DataFrame([{
            "crop": crop,
            "state": state,
            "cost_of_cultivation_`_hectare_a2+fl": cost_a2fl,
            "cost_of_cultivation_`_hectare_c2": cost_c2,
            "cost_of_production_`_quintal_c2": cost_production,
        }])

        prediction = model.predict(input_df)[0]

        st.success(f"### Predicted Yield: {prediction:.2f} Quintal/Hectare")

        crop_avg = df[df["crop"] == crop][TARGET].mean()
        st.caption(
            f"Historical average yield for {crop}: {crop_avg:.2f} Quintal/Hectare"
        )

        combo_exists = ((df["crop"] == crop) & (df["state"] == state)).any()
        if not combo_exists:
            st.warning(
                "⚠️ This crop-state combination was not present in the training data. "
                "The prediction is an extrapolation and may be less reliable."
            )

    st.divider()
    st.caption(
        "Model: Random Forest Regressor (tuned via GridSearchCV) | Trained on 49 records from "
        "India's agricultural cost and production dataset (data.gov.in)"
    )


def run_insights_tab():
    st.header("Exploratory Data Analysis")
    st.write("Key charts generated during data exploration and model evaluation.")

    chart_info = [
        ("outputs/yield_by_crop_log.png", "Yield Distribution by Crop (Log Scale)"),
        ("outputs/yield_by_state.png", "Average Yield by State"),
        ("outputs/cost_vs_yield.png", "Cost of Cultivation vs Yield"),
        ("outputs/correlation_heatmap.png", "Correlation Heatmap"),
        ("outputs/feature_importance.png", "Feature Importance (Random Forest)"),
        ("outputs/actual_vs_predicted.png", "Actual vs Predicted Yield"),
    ]

    for path, caption in chart_info:
        try:
            st.image(path, caption=caption, use_container_width=True)
        except Exception:
            st.warning(f"Could not load: {path}")


def main():
    st.set_page_config(page_title="Crop Yield Predictor", page_icon="🌾", layout="centered")

    st.title("🌾 Crop Yield Prediction (India)")
    st.write(
        "Estimate crop yield (Quintal/Hectare) based on crop type, state, "
        "and cultivation/production costs. Powered by a Random Forest model "
        "trained on government agricultural cost data."
    )

    df = load_data()
    model = train_model(df)

    tab1, tab2 = st.tabs(["🔮 Predict Yield", "📊 Data Insights"])

    with tab1:
        run_prediction_tab(df, model)

    with tab2:
        run_insights_tab()

    

if __name__ == "__main__":
    main()