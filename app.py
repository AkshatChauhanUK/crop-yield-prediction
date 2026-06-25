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
import shap
import matplotlib.pyplot as plt

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

@st.cache_resource
def get_shap_explainer(_model):
    rf_model = _model.named_steps["model"]
    return shap.TreeExplainer(rf_model)

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
    cost_a2fl = st.slider(
        "Cost of Cultivation - A2+FL (Rs/Hectare)", min_value=0.0, max_value=60000.0, value=15000.0, step=500.0
    )
    cost_c2 = st.slider(
        "Cost of Cultivation - C2 (Rs/Hectare)", min_value=0.0, max_value=90000.0, value=25000.0, step=500.0
    )
    cost_production = st.slider(
        "Cost of Production - C2 (Rs/Quintal)", min_value=0.0, max_value=5000.0, value=1500.0, step=100.0
    )

    # --- Live preview: updates instantly as sliders move, no button needed ---
    live_input_df = pd.DataFrame([{
        "crop": crop,
        "state": state,
        "cost_of_cultivation_`_hectare_a2+fl": cost_a2fl,
        "cost_of_cultivation_`_hectare_c2": cost_c2,
        "cost_of_production_`_quintal_c2": cost_production,
    }])
    live_prediction = model.predict(live_input_df)[0]
    st.metric("Live Preview", f"{live_prediction:.2f} Quintal/Hectare")
    st.caption("Updates instantly as you move the sliders. Click below for the full prediction with explanation.")

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
        # --- SHAP explanation for this specific prediction ---
        explainer = get_shap_explainer(model)
        preprocessor = model.named_steps["preprocessor"]
        input_transformed = preprocessor.transform(input_df)
        if hasattr(input_transformed, "toarray"):
            input_transformed = input_transformed.toarray()
        input_transformed = input_transformed.astype(float)

        cat_encoder = preprocessor.named_transformers_["cat"]
        cat_feature_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
        all_feature_names = cat_feature_names + NUMERIC_FEATURES

        shap_values = explainer.shap_values(input_transformed)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        shap_values = np.array(shap_values).flatten()

        
        if len(shap_values) != len(all_feature_names):
            st.error(f"Feature mismatch: {len(shap_values)} SHAP values vs {len(all_feature_names)} names. Skipping explanation.")
            shap_values = None

        base_value = explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = base_value[0]
        base_value = float(base_value)

        if shap_values is not None:
            impact_df = pd.DataFrame({
                "feature": all_feature_names,
                "shap_value": shap_values,
            })

            selected_crop_col = f"crop_{crop}"
            selected_state_col = f"state_{state}"

            def is_relevant(feature_name):
                if feature_name in (selected_crop_col, selected_state_col):
                    return True
                if not feature_name.startswith("crop_") and not feature_name.startswith("state_"):
                    return True
                return False

            impact_df = impact_df[impact_df["feature"].apply(is_relevant)].copy()
            impact_df["abs_impact"] = impact_df["shap_value"].abs()
            top_features = impact_df.sort_values("abs_impact", ascending=False).head(3)

            with st.expander("🔍 Why this prediction? (See feature impact)"):
                st.caption(
                    f"Starting from an average yield of {base_value:.2f}, "
                    f"here's what pushed this specific prediction up or down:"
                )
                for _, row in top_features.iterrows():
                    feature_name = row["feature"]
                    shap_val = row["shap_value"]

                    if feature_name == selected_crop_col:
                        label = f"Growing **{crop}**"
                    elif feature_name == selected_state_col:
                        label = f"Being grown in **{state}**"
                    else:
                        clean_name = feature_name.replace("_", " ")
                        label = f"**{clean_name}**"

                    direction = "increased ⬆️" if shap_val > 0 else "decreased ⬇️"
                    st.write(f"- {label} {direction} the prediction by **{abs(shap_val):.2f}** quintal/hectare")

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
    st.header("Model Evaluation")
    st.write("All models were evaluated using 5-fold cross-validation on the 49-row dataset.")

    try:
        comparison_df = pd.read_csv("outputs/cv_model_comparison.csv")
        display_df = comparison_df.rename(columns={
            "model": "Model",
            "rmse_mean": "RMSE",
            "mae_mean": "MAE",
            "r2_mean": "R²",
            "r2_std": "R² Std Dev",
        })[["Model", "RMSE", "MAE", "R²", "R² Std Dev"]]

        best_mae_idx = display_df["MAE"].idxmin()

        st.dataframe(
            display_df.style.apply(
                lambda row: ["background-color: #d4edda" if row.name == best_mae_idx else "" for _ in row],
                axis=1
            ).format({"RMSE": "{:.2f}", "MAE": "{:.2f}", "R²": "{:.3f}", "R² Std Dev": "{:.3f}"}),
            use_container_width=True,
            hide_index=True,
        )
        st.caption("Highlighted row = best model by MAE (lowest error). This is the model powering predictions above.")
    except Exception:
        st.warning("Could not load model comparison data.")

    st.divider()
    st.header("Exploratory Data Analysis")
    st.write("Key charts generated during data exploration and model evaluation.")

    chart_info = [
        ("outputs/yield_by_crop_log.png", "Yield Distribution by Crop (Log Scale)"),
        ("outputs/yield_by_state.png", "Average Yield by State"),
        ("outputs/cost_vs_yield.png", "Cost of Cultivation vs Yield"),
        ("outputs/correlation_heatmap.png", "Correlation Heatmap"),
        ("outputs/feature_importance.png", "Feature Importance (Random Forest)"),
        ("outputs/shap_summary.png", "SHAP Summary (Feature Impact Direction)"),
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