# Crop Yield Prediction in India

## Project Overview
This project predicts crop yield (Quintal/Hectare) across different crops and Indian states using machine learning, based on cultivation and production cost data. It was built as part of a Machine Learning Internship project focused on solving real-world agricultural problems in India.

## Problem Statement
India's agricultural sector supports a large share of its population, yet farmers and policymakers often lack data-driven tools to estimate crop yield based on cultivation costs. This project builds a predictive model that estimates yield given the crop type, state, and cost of cultivation/production — helping identify which factors most influence yield.

## Dataset
- **Source:** Government of India agricultural cost and production dataset (data.gov.in)
- **File used:** `datafile (1).csv`
- **Rows:** 49 (after cleaning)
- **Columns:**
  - `crop` — Crop name (e.g., Wheat, Rice, Maize, Sugarcane)
  - `state` — Indian state where the crop is grown
  - `cost_of_cultivation_hectare_a2fl` — Cost of cultivation (Rs/Hectare, A2+FL method)
  - `cost_of_cultivation_hectare_c2` — Cost of cultivation (Rs/Hectare, C2 method)
  - `cost_of_production_quintal_c2` — Cost of production (Rs/Quintal, C2 method)
  - `yield_quintal_hectare` — **Target variable**: Yield in Quintal/Hectare

## Project Workflow

### 1. Data Cleaning (`src/preprocess.py`)
- Standardized column names
- Cleaned categorical text (Crop, State)
- Converted all numeric columns properly, handling missing/invalid values
- Removed duplicate rows
- Result: 49 clean rows, 0 missing values

### 2. Exploratory Data Analysis (`src/eda.py`)
Key findings:
- **Sugarcane has a dramatically higher yield** (700-1000+ Quintal/Hectare) compared to all other crops, due to its high-biomass nature. A log-scale chart was used to visualize all crops fairly.
- **Tamil Nadu, Karnataka, and Maharashtra** show the highest average yields, largely influenced by sugarcane cultivation in those states.
- **Strong positive correlation (0.87)** between Cost of Cultivation (C2) and Yield — higher investment generally aligns with higher yield.
- **Negative correlation (-0.49)** between Cost of Production per Quintal and Yield — crops with lower per-unit production cost tend to have higher yield.

Charts generated:
- `yield_by_crop.png`, `yield_by_crop_log.png`
- `yield_by_state.png`
- `cost_vs_yield.png`
- `correlation_heatmap.png`

### 3. Model Training (`src/train_model.py`)

Four models were trained and evaluated using **5-fold cross-validation** (more reliable than a single train-test split on a 49-row dataset):

| Model | RMSE | MAE | R² | R² Std Dev |
|---|---|---|---|---|
| Linear Regression | 65.35 | 51.60 | -29.57 | 61.03 |
| Random Forest | 34.37 | 14.65 | 0.959 | 0.032 |
| **Random Forest (Tuned)** | **33.73** | **13.69** | **0.962** | **0.027** |
| XGBoost | 51.53 | 23.84 | 0.468 | 0.925 |

**Key insight — why cross-validation mattered:** An initial single train-test split made Linear Regression look competitive (R² = 0.949). However, 5-fold cross-validation revealed the truth: Linear Regression's R² is actually **-29.57 on average**, with an enormous standard deviation (61.03) — meaning its performance varies wildly depending on which rows land in the test fold (especially when Sugarcane, an extreme outlier, is involved). This shows that small datasets can give misleadingly optimistic results with a single split, and cross-validation is essential for an honest evaluation.

**Random Forest is the clear, consistent winner** — both in raw performance (lowest RMSE/MAE) and in stability (lowest standard deviation across folds). XGBoost, despite being a powerful algorithm, underperforms here because it needs more data than this 49-row dataset provides, leading to inconsistent results across folds.

**Hyperparameter tuning:** Using `GridSearchCV`, the best Random Forest settings were `n_estimators=100` and `max_depth=10`. This gave a small improvement over the default settings (MAE dropped from 14.65 to 13.69). Hyperparameter tuning provided incremental improvement, confirming the model was already well-suited for this dataset size.

Chart: `cv_model_comparison.png` visualizes RMSE and MAE with error bars across all models.
### 4. Feature Importance
Using the Random Forest model, feature importance was extracted to understand what drives yield predictions:

1. **Cost of Production (per Quintal)** — most important factor
2. **Crop type** — second most important
3. **Cost of Cultivation (A2+FL)**
4. **Cost of Cultivation (C2)**
5. **State** — least important

**Key insight:** Yield is primarily driven by *which crop* is grown and *how much it costs* to grow/produce it — not by *which state* it is grown in. This suggests that crop-specific agronomic factors matter more than regional/state-specific factors for this dataset.
### 5. Interactive Prediction App (`app.py`)

A Streamlit web app was built to make the model usable without writing code. It has two tabs:

- **Predict Yield** — Select a crop and state (the state dropdown automatically shows only states where that crop exists in the training data), enter cultivation/production costs, and get an instant yield prediction using the tuned Random Forest model. If a crop-state combination wasn't seen during training, the app shows a warning that the prediction is an extrapolation.
- **Data Insights** — All the EDA and model evaluation charts in one place, viewable without opening any code.

Run it with:
```bash
streamlit run app.py
```
## Tech Stack
- Python 3.13
- pandas, numpy — data handling
- matplotlib, seaborn — visualization
- scikit-learn — machine learning (Linear Regression, Random Forest)
- xgboost — gradient boosting model for comparison
- streamlit — interactive web app for predictions

## How to Run
```bash
# 1. Clean the data
python src/preprocess.py --input "data/datafile (1).csv" --output "data/cleaned_crop_data.csv"

# 2. Run exploratory data analysis
python src/eda.py --input data/cleaned_crop_data.csv --outdir outputs

# 3. Train and evaluate models
python src/train_model.py --input data/cleaned_crop_data.csv --outdir outputs

# 4. Launch the interactive prediction app
streamlit run app.py

## Limitations
- Small dataset (49 rows) limits statistical robustness, especially for test-set evaluation
- Does not account for external factors like rainfall, soil quality, or irrigation access, which likely influence yield significantly
- State-wise sample sizes are uneven, which may bias average comparisons

## Future Improvements
- Incorporate rainfall and soil data for more accurate predictions
- Use cross-validation instead of a single train-test split to get more stable metrics on this small dataset
- Build an interactive prediction tool (e.g., Streamlit) for real-time yield estimation
- Expand the dataset with multi-year data to capture trends over time

## Author
Built as part of a Machine Learning Internship project — UpSkill Campus / UCT / The IoT Academy