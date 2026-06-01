import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(
    page_title="Fiber Degradation Model",
    layout="wide"
)

st.title("Organic vs Synthetic Fiber Degradation Model")

st.markdown(
    """
    Compares how quickly organic and synthetic fibers break down in soil.
    It uses mass-loss data from two sources and models the fraction of each fiber remaining over time.
    """
)

st.subheader("Model Used")

st.markdown(
    """
    Uses an **exponential decay model**:
    """
)

st.latex(r"R(t) = e^{-kt}")

st.markdown(
    """
    In this equation, \(R(t)\) is the fraction of the fiber remaining after \(t\) days in soil.
    The value \(k\) is the degradation speed. A larger \(k\) means the fiber breaks down faster.
    """
)

st.subheader("How the Model Is Fit")

st.markdown(
    """
    Although the model is exponential, we estimate \(k\) using **linear regression**.
    First, the mass-loss data is converted into fraction remaining:
    """
)

st.latex(r"R = 1 - \frac{\text{mass lost percent}}{100}")

st.markdown(
    """
    Then we take the natural log of the exponential model:
    """
)

st.latex(r"\ln(R) = -kt")

st.markdown(
    """
    This changes the exponential model into a linear relationship.
    We fit a straight line between time and the natural log of the fraction remaining.
    The slope of that line is \(-k\), so the degradation speed is the negative of the slope.
    """
)

st.latex(r"k = -\frac{\sum t\ln(R)}{\sum t^2}")

st.markdown(
    """
    This is the linear regression formula used when the line is forced to pass through the origin.
    That makes sense here because at day 0, the fiber has 100% remaining, so \(R=1\) and \(\ln(1)=0\).
    """
)

st.subheader("Main Output")

st.markdown(
    """
    After finding the degradation speed, we calculate how many days it takes until only half of the fiber remains:
    """
)

st.latex(r"\text{Days until 50 percent remains} = \frac{\ln(2)}{k}")

st.markdown(
    """
    A lower number of days means the fiber breaks down faster.
    A higher number of days means the fiber stays in the soil longer.
    """
)

sular_data = [
    ["Sular & Devrim 2019", "Cotton", "Organic", 30, 12.9],
    ["Sular & Devrim 2019", "Cotton", "Organic", 120, 89.4],
    ["Sular & Devrim 2019", "Polyester", "Synthetic", 30, 0.7],
    ["Sular & Devrim 2019", "Polyester", "Synthetic", 120, 1.4],
    ["Sular & Devrim 2019", "PAN/Acrylic-like", "Synthetic", 30, 0.6],
    ["Sular & Devrim 2019", "PAN/Acrylic-like", "Synthetic", 120, 2.4],
]

wool_report_data = [
    ["Wool Soil-Burial Report", "Wool", "Organic", 30, 3.9],
    ["Wool Soil-Burial Report", "Wool", "Organic", 60, 35.8],
    ["Wool Soil-Burial Report", "Wool", "Organic", 90, 93.3],
    ["Wool Soil-Burial Report", "Wool", "Organic", 180, 94.6],
    ["Wool Soil-Burial Report", "Wool", "Organic", 270, 97.9],
    ["Wool Soil-Burial Report", "Polyester", "Synthetic", 30, 0.0],
    ["Wool Soil-Burial Report", "Polyester", "Synthetic", 60, 0.0],
    ["Wool Soil-Burial Report", "Polyester", "Synthetic", 90, 0.0],
    ["Wool Soil-Burial Report", "Polyester", "Synthetic", 180, 0.0],
    ["Wool Soil-Burial Report", "Polyester", "Synthetic", 270, 0.0],
]

df = pd.DataFrame(
    sular_data + wool_report_data,
    columns=["Source", "Fiber", "Fiber Type", "Days in Soil", "Mass Lost (%)"]
)

df["Fraction Remaining"] = 1 - df["Mass Lost (%)"] / 100
df["Model Fraction Remaining"] = df["Fraction Remaining"].clip(0.001, 0.999)

def remaining_fraction_after_time(t, degradation_speed):
    return np.exp(-degradation_speed * t)

def fit_models(data):
    rows = []

    for fiber in data["Fiber"].unique():
        subset = data[data["Fiber"] == fiber].copy()

        t = subset["Days in Soil"].values.astype(float)
        y = subset["Model Fraction Remaining"].values.astype(float)

        log_y = np.log(y)
        degradation_speed = -np.sum(t * log_y) / np.sum(t ** 2)

        if degradation_speed <= 0:
            days_until_half_remains = np.inf
        else:
            days_until_half_remains = np.log(2) / degradation_speed

        rows.append({
            "Fiber": fiber,
            "Fiber Type": subset["Fiber Type"].iloc[0],
            "Degradation Speed": degradation_speed,
            "Days Until 50% Remains": days_until_half_remains
        })

    return pd.DataFrame(rows)

results_df = fit_models(df)

filtered_df = df.copy()
filtered_results = results_df.copy()
max_days = 500

st.subheader("Model Results")

display_results = filtered_results.copy()
display_results["Degradation Speed"] = display_results["Degradation Speed"].map(lambda x: f"{x:.6f}")
display_results["Days Until 50% Remains"] = display_results["Days Until 50% Remains"].map(
    lambda x: "More than 10,000 days" if np.isinf(x) or x > 10000 else f"{x:.1f}"
)

st.dataframe(display_results, width="stretch")

col1, col2 = st.columns(2)

with col1:
    organic_avg = results_df[results_df["Fiber Type"] == "Organic"]["Degradation Speed"].mean()
    st.metric("Average Organic Degradation Speed", f"{organic_avg:.6f}")

with col2:
    synthetic_avg = results_df[results_df["Fiber Type"] == "Synthetic"]["Degradation Speed"].mean()
    st.metric("Average Synthetic Degradation Speed", f"{synthetic_avg:.6f}")

st.subheader("Predicted Fiber Remaining Over Time")

time_grid = np.linspace(0, max_days, 300)

fig, ax = plt.subplots(figsize=(10, 6))

for _, row in filtered_results.iterrows():
    fiber = row["Fiber"]
    degradation_speed = row["Degradation Speed"]

    predicted_remaining = remaining_fraction_after_time(time_grid, degradation_speed)
    subset = filtered_df[filtered_df["Fiber"] == fiber]

    ax.plot(time_grid, predicted_remaining, label=f"{fiber} prediction")
    ax.scatter(subset["Days in Soil"], subset["Fraction Remaining"], s=60)

ax.set_xlabel("Days in Soil")
ax.set_ylabel("Fraction of Fiber Remaining")
ax.set_title("Predicted Fiber Remaining Over Time")
ax.set_ylim(0, 1.05)
ax.legend()
ax.grid(True)

st.pyplot(fig)

st.subheader("Degradation Speed by Fiber")

fig2, ax2 = plt.subplots(figsize=(9, 5))

ax2.bar(
    filtered_results["Fiber"],
    filtered_results["Degradation Speed"]
)

ax2.set_xlabel("Fiber")
ax2.set_ylabel("Degradation Speed")
ax2.set_title("Higher Values Mean Faster Degradation")
ax2.tick_params(axis="x", rotation=30)
ax2.grid(axis="y")

st.pyplot(fig2)

st.subheader("Days Until 50% of the Fiber Remains")

fig3, ax3 = plt.subplots(figsize=(9, 5))

plot_days = filtered_results["Days Until 50% Remains"].replace(np.inf, np.nan)

ax3.bar(
    filtered_results["Fiber"],
    plot_days
)

ax3.set_xlabel("Fiber")
ax3.set_ylabel("Days Until 50% Remains")
ax3.set_title("Lower Values Mean Faster Degradation")
ax3.tick_params(axis="x", rotation=30)
ax3.grid(axis="y")

st.pyplot(fig3)

st.subheader("Organic vs Synthetic Summary")

summary_df = filtered_results.groupby("Fiber Type").agg(
    Average_Degradation_Speed=("Degradation Speed", "mean"),
    Average_Days_Until_50_Percent_Remains=("Days Until 50% Remains", "mean")
).reset_index()

summary_df["Average_Degradation_Speed"] = summary_df["Average_Degradation_Speed"].map(lambda x: f"{x:.6f}")
summary_df["Average_Days_Until_50_Percent_Remains"] = summary_df["Average_Days_Until_50_Percent_Remains"].map(
    lambda x: "More than 10,000 days" if np.isinf(x) or x > 10000 else f"{x:.1f}"
)

st.dataframe(summary_df, width="stretch")

st.subheader("Data Used")

clean_display = filtered_df[
    [
        "Source",
        "Fiber",
        "Fiber Type",
        "Days in Soil",
        "Mass Lost (%)",
        "Fraction Remaining"
    ]
].copy()

clean_display["Fraction Remaining"] = clean_display["Fraction Remaining"].round(3)

st.dataframe(clean_display, width="stretch")

st.subheader("Conclusion")

st.markdown(
    """
    The model predicts that organic fibers such as cotton and wool break down much faster in soil than synthetic fibers such as polyester and PAN/acrylic-like material.
    Organic fibers have higher degradation speeds and reach 50% remaining material sooner.
    Synthetic fibers show very little mass loss over the same time periods, meaning they remain in the soil much longer.
    """
)

st.subheader("Sources")

st.markdown(
    """
    1. Sular & Devrim, 2019:  
    https://www.researchgate.net/publication/330986655_Biodegradation_Behaviour_of_Different_Textile_Fibres_Visual_Morphological_Structural_Properties_and_Soil_Analyses

    2. Wool Soil-Burial Report:  
    https://transparencycatalog.com/assets/uploads/files/Wool-Biodegredation.pdf
    """
)