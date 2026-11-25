import os
from typing import Dict, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error


def nasa_phm08_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
  """Asymmetric NASA PHM08 score (penalizes late RUL predictions more heavily)."""
  diff = y_pred - y_true
  scores = np.where(
      diff < 0, np.exp(-diff / 13.0) - 1.0, np.exp(diff / 10.0) - 1.0
  )
  return float(np.sum(scores))


def evaluate_predictions(
    y_true_raw: np.ndarray,
    y_pred: np.ndarray,
    rul_cap: float = 125.0,
) -> Dict[str, float]:
  """Compute capped RMSE, raw RMSE, and NASA PHM08 score for RUL predictions."""
  y_true_capped = np.minimum(y_true_raw, rul_cap)
  rmse_capped = float(root_mean_squared_error(y_true_capped, y_pred))
  rmse_raw = float(root_mean_squared_error(y_true_raw, y_pred))
  phm_score = nasa_phm08_score(y_true_capped, y_pred)
  return {
      "test_rmse_capped": round(rmse_capped, 2),
      "test_rmse_raw": round(rmse_raw, 2),
      "phm08_score": round(phm_score, 2),
  }


def save_subset_predictions(
    output_dir: str,
    subset: str,
    unit_ids: np.ndarray,
    y_true_raw: np.ndarray,
    preds_by_model: Dict[str, np.ndarray],
    rul_cap: float = 125.0,
    last_observed_cycles: Optional[np.ndarray] = None,
) -> pd.DataFrame:
  """Save per-engine remaining useful cycle predictions to CSV."""
  pred_dir = os.path.join(output_dir, "predictions")
  os.makedirs(pred_dir, exist_ok=True)

  y_true_capped = np.minimum(y_true_raw, rul_cap)
  df = pd.DataFrame({
      "unit_id": unit_ids,
      "actual_RUL_capped": np.round(y_true_capped, 1),
      "actual_RUL_raw": np.round(y_true_raw, 1),
  })

  if last_observed_cycles is not None:
    df["last_observed_cycle"] = last_observed_cycles

  for model_name, preds in preds_by_model.items():
    prefix = model_name.lower()
    col_pred = f"{prefix}_pred_RUL_cycles"
    col_err = f"{prefix}_error_cycles"
    df[col_pred] = np.round(preds, 2)
    df[col_err] = np.round(preds - y_true_capped, 2)
    if last_observed_cycles is not None:
      df[f"{prefix}_est_failure_cycle"] = np.round(
          last_observed_cycles + preds, 1
      )

  out_path = os.path.join(pred_dir, f"{subset.lower()}_rul_predictions.csv")
  df.to_csv(out_path, index=False)
  return df