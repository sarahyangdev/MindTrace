"""
MindTrace - Streamlit test harness for the v2 distress model.

Run with:   streamlit run mindtrace_app.py
"""

import json
import numpy as np
import pandas as pd
import shap
import streamlit as st
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

st.set_page_config(page_title="MindTrace Model Tester", page_icon="🧭", layout="wide")

DATA = "mindtrace_master_dataset_ca_tracts_v2.csv"
MODEL = "mindtrace_model_v2.json"
SPEC = "mindtrace_model_v2_features.json"

PRETTY = {
    "poverty_rate": "Poverty", "housing_burden_pct": "Housing Burden",
    "education_pct_less_hs": "Education Access", "unemployment_rate": "Unemployment",
    "linguistic_isolation_pct": "Language Isolation", "nb10_poverty_rate": "Neighbourhood Poverty",
    "nb10_education_pct_less_hs": "Neighbourhood Education", "poverty_vs_neighbours": "Poverty vs Neighbours",
    "housing_burden_pct_vs_county": "Housing Burden vs County", "pop_density_per_sqmi": "Population Density",
    "pct_adult": "Adult Share", "pm25_annual": "PM2.5", "nb10_pm25_annual": "Neighbourhood PM2.5",
    "pm25_peak_ratio": "Smoke Peak Ratio", "ozone_mean_ppb_2yr": "Ozone",
    "ces4_diesel_pm": "Diesel Exhaust", "dist_to_mental_health": "Distance to Care",
    "gravity_access_mh": "Care Accessibility", "n_primary_care_within_25mi": "Nearby Clinics",
    "poverty_x_pm25": "Poverty x PM2.5",
}


@st.cache_resource
def load():
    spec = json.load(open(SPEC))
    model = XGBRegressor()
    model.load_model(MODEL)
    df = pd.read_csv(DATA, dtype={"tract_fips": str, "county_fips": str})
    return spec, model, df, shap.TreeExplainer(model)


spec, model, df, explainer = load()
FEATURES = spec["features"]

st.title("🧭 MindTrace — Community Distress Model")
st.caption(
    f"XGBoost regressor · target = CDC PLACES MHLTH · {len(FEATURES)} features · "
    f"R² {spec['r2_county_holdout']:.3f} / MAE {spec['mae_county_holdout']:.2f} pts (county-held-out CV)"
)
st.warning(
    "Informational tool, not a clinical or diagnostic instrument. Model output describes "
    "population-level *associations*, never individual risk or causation.",
    icon="⚠️",
)

tab1, tab2, tab3 = st.tabs(["Tract Explorer", "What-If Simulator", "Model Performance"])

# ------------------------------------------------------------------ TAB 1
with tab1:
    c1, c2 = st.columns([1, 2])
    with c1:
        county = st.selectbox("County", sorted(df.county_name.unique()),
                              index=sorted(df.county_name.unique()).index("Los Angeles"))
        sub = df[df.county_name == county]
        tract = st.selectbox("Census tract", sub.tract_fips.tolist())

    row = df[df.tract_fips == tract].iloc[0]
    X = df.loc[df.tract_fips == tract, FEATURES]
    pred = float(model.predict(X)[0])
    actual = row.mhlth_crude_prev
    pct = (df.mhlth_predicted_v2 < pred).mean() * 100

    with c2:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Predicted distress", f"{pred:.1f}%")
        m2.metric("Actual (PLACES)", f"{actual:.1f}%", f"{pred - actual:+.1f}")
        m3.metric("State percentile", f"{pct:.0f}th")
        m4.metric("Population", f"{int(row.total_population):,}")

    st.subheader("What drives this tract")
    sv = explainer.shap_values(X)[0]
    order = np.argsort(-np.abs(sv))[:10]
    labels = [PRETTY.get(FEATURES[i], FEATURES[i]) for i in order][::-1]
    vals = [sv[i] for i in order][::-1]

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.barh(labels, vals, color=["#c0392b" if v > 0 else "#27ae60" for v in vals])
    ax.axvline(0, color="#333", lw=0.8)
    ax.set_xlabel("SHAP value  (← lowers distress   raises distress →)")
    ax.set_title(f"Local feature attribution · baseline {explainer.expected_value:.1f}%")
    fig.tight_layout()
    st.pyplot(fig)

    st.subheader("Groq prompt payload")
    # Composition/context variables help accuracy but are not actionable, and a user cannot
    # act on "Adult Share". Keep them in the model, filter them out of the LLM prompt.
    CONTEXT_ONLY = {"pct_adult", "pop_density_per_sqmi", "total_population"}
    top = sorted(((n, v) for n, v in zip(FEATURES, sv) if n not in CONTEXT_ONLY),
                 key=lambda t: -abs(t[1]))[:3]
    drivers = " | ".join(f"{PRETTY.get(n, n)} ({v:+.1f})" for n, v in top)
    st.code(
        f"Tract: {tract} | County: {county} | Predicted Distress: {pct:.0f}th percentile "
        f"({pred:.1f}%) | Top Drivers: {drivers} | "
        f"Nearest MH facility: {row.dist_to_mental_health:.1f} mi",
        language="text",
    )

    with st.expander("Feature values for this tract"):
        st.dataframe(
            pd.DataFrame({"feature": [PRETTY.get(f, f) for f in FEATURES],
                          "value": [row[f] for f in FEATURES],
                          "state percentile": [(df[f] < row[f]).mean() * 100 for f in FEATURES]}
                         ).round(2),
            use_container_width=True, hide_index=True,
        )

# ------------------------------------------------------------------ TAB 2
with tab2:
    st.markdown("Move a slider to see how the prediction responds. Useful for probing whether "
                "the model behaves sensibly, and for scenario questions "
                "(*what if PM2.5 dropped by 2 µg/m³?*).")

    base = df[df.tract_fips == tract][FEATURES].copy()
    KEY = ["poverty_rate", "housing_burden_pct", "pm25_annual", "ozone_mean_ppb_2yr",
           "dist_to_mental_health", "education_pct_less_hs", "unemployment_rate"]

    sim = base.copy()
    cols = st.columns(2)
    for i, f in enumerate(KEY):
        lo, hi = float(df[f].quantile(0.01)), float(df[f].quantile(0.99))
        sim[f] = cols[i % 2].slider(PRETTY.get(f, f), lo, hi, float(base[f].iloc[0]),
                                    step=(hi - lo) / 100)

    new = float(model.predict(sim)[0])
    d1, d2, d3 = st.columns(3)
    d1.metric("Original prediction", f"{pred:.1f}%")
    d2.metric("Simulated prediction", f"{new:.1f}%", f"{new - pred:+.2f}")
    d3.metric("Change", f"{new - pred:+.2f} pts")

    if abs(new - pred) > 0.01:
        sv2 = explainer.shap_values(sim)[0]
        delta = pd.Series(sv2 - sv, index=[PRETTY.get(f, f) for f in FEATURES])
        delta = delta[delta.abs() > 0.001].sort_values()
        if len(delta):
            st.bar_chart(delta, horizontal=True)

# ------------------------------------------------------------------ TAB 3
with tab3:
    ok = df.mhlth_predicted_oof.notna()
    err = (df.loc[ok, "mhlth_predicted_oof"] - df.loc[ok, "mhlth_crude_prev"]).abs()
    a, b, c, d = st.columns(4)
    a.metric("R² (held-out counties)", f"{spec['r2_county_holdout']:.3f}")
    b.metric("MAE", f"{spec['mae_county_holdout']:.2f} pts")
    c.metric("Within 2 pts", f"{(err <= 2).mean() * 100:.0f}%")
    d.metric("Tracts", f"{int(ok.sum()):,}")

    st.markdown("**Predicted vs actual** (out-of-fold, so every point was predicted by a model "
                "that never saw its county during training)")
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    ax2.scatter(df.loc[ok, "mhlth_crude_prev"], df.loc[ok, "mhlth_predicted_oof"],
                s=4, alpha=0.25, color="#2980b9")
    lims = [df.mhlth_crude_prev.min(), df.mhlth_crude_prev.max()]
    ax2.plot(lims, lims, "k--", lw=1)
    ax2.set_xlabel("Actual MHLTH (%)")
    ax2.set_ylabel("Predicted MHLTH (%)")
    fig2.tight_layout()
    st.pyplot(fig2)

    st.markdown("**Worst-predicted tracts** — where the model is least reliable")
    worst = df.loc[ok].assign(abs_err=err).nlargest(10, "abs_err")
    st.dataframe(
        worst[["tract_fips", "county_name", "mhlth_crude_prev",
               "mhlth_predicted_oof", "abs_err", "poverty_rate"]].round(2),
        use_container_width=True, hide_index=True,
    )

    st.info(
        "The PLACES target is itself a modelled estimate carrying a ±1.8 pt median confidence "
        "interval. An MAE of ~1.2 pts is therefore close to the noise floor — treat the model "
        "as a **ranking** tool (rank correlation 0.86), not a source of precise percentages.",
        icon="ℹ️",
    )
