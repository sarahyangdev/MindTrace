"""
MindTrace - Week 1: Train the XGBoost distress regressor.

PRD Module 3 specifies exactly 5 features. We use those 5, nothing more.
Target (y) = CDC PLACES MHLTH: % of adults with frequent mental distress.
"""

import pandas as pd
import shap
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# ---------------------------------------------------------------- 1. LOAD
df = pd.read_csv("mindtrace_master_dataset_ca_tracts.csv", dtype={"tract_fips": str})

# The 5 features named in the PRD
FEATURES = [
    "pm25_annual",            # particulate exposure
    "tree_canopy_pct",        # green buffer
    "poverty_rate",           # economic baseline
    "housing_burden_pct",     # household strain
    "dist_to_mental_health",  # spatial accessibility barrier
]
TARGET = "mhlth_crude_prev"

# tree_canopy_pct only exists for Los Angeles County, so train there:
# it is the one region where all 5 PRD features are complete.
df = df.dropna(subset=FEATURES + [TARGET])

X = df[FEATURES]
y = df[TARGET]
print(f"Training on {len(X)} census tracts, {len(FEATURES)} features\n")

# ---------------------------------------------------------------- 2. TRAIN
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBRegressor(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=4,
    random_state=42,
)
model.fit(X_train, y_train)

# ---------------------------------------------------------------- 3. EVALUATE
pred = model.predict(X_test)
print(f"R2  : {r2_score(y_test, pred):.3f}")
print(f"MAE : {mean_absolute_error(y_test, pred):.3f} percentage points\n")

print("Feature importance:")
for name, score in sorted(zip(FEATURES, model.feature_importances_),
                          key=lambda t: -t[1]):
    print(f"  {name:24s} {score:.3f}")

# ---------------------------------------------------------------- 4. SHAP
# TreeExplainer gives the per-tract feature attributions the PRD needs.
explainer = shap.TreeExplainer(model)

tract = X_test.iloc[[0]]                    # explain one tract
tract_id = df.loc[tract.index[0], "tract_fips"]
shap_values = explainer.shap_values(tract)[0]

print(f"\nSHAP explanation for tract {tract_id}")
print(f"Predicted distress: {model.predict(tract)[0]:.1f}%")
print(f"Baseline (state average): {explainer.expected_value:.1f}%")
for name, value in sorted(zip(FEATURES, shap_values), key=lambda t: -abs(t[1])):
    arrow = "raises" if value > 0 else "lowers"
    print(f"  {name:24s} {value:+.2f}  ({arrow} distress)")

# ---------------------------------------------------------------- 5. SAVE
model.save_model("model.json")
print("\nSaved model.json")

# This is the string the FastAPI backend hands to Groq (PRD Module 4).
top = sorted(zip(FEATURES, shap_values), key=lambda t: -abs(t[1]))[:3]
drivers = " | ".join(f"{n} ({v:+.1f})" for n, v in top)
print(f"\nGroq prompt payload:\n  Tract: {tract_id} | "
      f"Predicted Distress: {model.predict(tract)[0]:.1f}% | Top Drivers: {drivers}")
