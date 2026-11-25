from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="C-MAPSS RUL Model Comparison",
    layout="wide",
)

ROOT_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = ROOT_DIR / "outputs"
SUMMARY_PATH = OUTPUTS_DIR / "summary_metrics.csv"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"

SUBSETS = ["FD001", "FD002", "FD003", "FD004"]


@st.cache_data
def load_summary_metrics(path: Path) -> pd.DataFrame:
  """Load the aggregate metrics table across subsets and models."""
  df = pd.read_csv(path)
  df["subset"] = df["subset"].astype(str).str.upper()
  return df


@st.cache_data
def load_subset_predictions(subset: str) -> pd.DataFrame | None:
  """Load per-engine RUL predictions for a single subset if available."""
  csv_path = PREDICTIONS_DIR / f"{subset.lower()}_rul_predictions.csv"
  if not csv_path.exists():
    return None
  return pd.read_csv(csv_path)


st.title("NASA C-MAPSS RUL Comparison (FD001–FD004)")
st.caption(
    "Compare LSTM and GRU across subsets using capped/raw RMSE and PHM08 score"
)

if not SUMMARY_PATH.exists():
  st.error(f"Missing `{SUMMARY_PATH}`. Run `python run_all.py` first.")
  st.stop()

summary_df = load_summary_metrics(SUMMARY_PATH)

# Sidebar controls
st.sidebar.header("Filters")
available_subsets = [s for s in SUBSETS if s in summary_df["subset"].unique()]
selected_subsets = st.sidebar.multiselect(
    "Subsets",
    options=available_subsets or SUBSETS,
    default=available_subsets or SUBSETS,
)

models_in_data = sorted(summary_df["model"].dropna().unique().tolist())
selected_models = st.sidebar.multiselect(
    "Models",
    options=models_in_data,
    default=models_in_data,
)

metric_options = {
    "Capped Test RMSE (cycles)": "test_rmse_capped",
    "Raw Test RMSE (cycles)": "test_rmse_raw",
    "Validation RMSE (cycles)": "val_rmse",
    "NASA PHM08 Score (lower is better)": "phm08_score",
}
selected_metric_label = st.sidebar.selectbox(
    "Chart Metric",
    options=list(metric_options.keys()),
    index=0,
)
selected_metric_col = metric_options[selected_metric_label]

filtered_df = summary_df[
    summary_df["subset"].isin(selected_subsets)
    & summary_df["model"].isin(selected_models)
].copy()

if filtered_df.empty:
  st.warning("No rows match the current filter selection.")
  st.stop()

# Top-level best-model callouts for the selected metric
st.subheader("Best Model per Selected Subset")
best_cols = st.columns(len(selected_subsets) if selected_subsets else 1)
for idx, subset in enumerate(selected_subsets):
  subset_rows = filtered_df[filtered_df["subset"] == subset]
  with best_cols[idx % len(best_cols)]:
    if subset_rows.empty:
      st.metric(label=subset, value="N/A")
      continue
    best_row = subset_rows.loc[subset_rows[selected_metric_col].idxmin()]
    st.metric(
        label=f"{subset} ({best_row['model']})",
        value=f"{best_row[selected_metric_col]:.2f}",
        delta=f"Epoch {int(best_row['best_epoch'])}",
        delta_color="off",
    )

# Side-by-side comparison table and grouped bar chart
col_table, col_chart = st.columns([1.25, 1.0])

with col_table:
  st.subheader("Summary Metrics Table")
  display_df = filtered_df.sort_values(["subset", "model"]).reset_index(
      drop=True
  )
  st.dataframe(
      display_df,
      hide_index=True,
      use_container_width=True,
      column_config={
          "subset": st.column_config.TextColumn("Subset"),
          "model": st.column_config.TextColumn("Model"),
          "best_epoch": st.column_config.NumberColumn(
              "Best Epoch", format="%d"
          ),
          "val_rmse": st.column_config.NumberColumn(
              "Val RMSE", format="%.2f"
          ),
          "test_rmse_capped": st.column_config.NumberColumn(
              "Test RMSE (Capped)", format="%.2f"
          ),
          "test_rmse_raw": st.column_config.NumberColumn(
              "Test RMSE (Raw)", format="%.2f"
          ),
          "phm08_score": st.column_config.NumberColumn(
              "PHM08 Score", format="%.2f"
          ),
      },
  )

with col_chart:
  st.subheader(selected_metric_label)
  chart_df = filtered_df.pivot(
      index="subset", columns="model", values=selected_metric_col
  ).reindex(selected_subsets)
  st.bar_chart(
      chart_df,
      stack=False,
      x_label="C-MAPSS Subset",
      y_label=selected_metric_label,
  )

# Per-engine prediction inspector for a chosen subset
st.divider()
st.subheader("Per-Engine Test Predictions")
inspect_subset = st.selectbox(
    "Inspect Engine Predictions for Subset",
    options=selected_subsets or SUBSETS,
)
pred_df = load_subset_predictions(inspect_subset)
if pred_df is None:
  st.info(
      f"Prediction CSV for `{inspect_subset}` not found in"
      " `outputs/predictions/`."
  )
else:
  st.dataframe(pred_df, hide_index=True, use_container_width=True)