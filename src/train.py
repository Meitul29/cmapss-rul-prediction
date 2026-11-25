import copy
import random
from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import root_mean_squared_error
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def seed_everything(seed: int = 42) -> None:
  """Set random seeds across Python, NumPy, and PyTorch for reproducibility."""
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


def predict_sequences(
    model: nn.Module,
    X_seq: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
  """Run inference on sequence windows in mini-batches."""
  model.eval()
  preds = []
  loader = DataLoader(
      TensorDataset(torch.from_numpy(X_seq)), batch_size=batch_size, shuffle=False
  )
  with torch.no_grad():
    for (xb,) in loader:
      xb = xb.to(device)
      preds.append(model(xb).cpu().numpy())
  return np.concatenate(preds, axis=0)


def train_sequence_model(
    model: nn.Module,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    device: torch.device,
    epochs: int = 40,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    patience: int = 8,
    rul_cap: float = 125.0,
    log_every: int = 5,
) -> Tuple[nn.Module, Dict[str, float]]:
  """Train LSTM/GRU sequence model with AdamW, scheduler, and early stopping."""
  model = model.to(device)
  optimizer = torch.optim.AdamW(
      model.parameters(), lr=lr, weight_decay=weight_decay
  )
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
      optimizer, mode="min", factor=0.5, patience=3
  )
  criterion = nn.MSELoss()

  tr_loader = DataLoader(
      TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
      batch_size=batch_size,
      shuffle=True,
  )

  best_val_rmse = float("inf")
  best_epoch = 0
  best_weights = copy.deepcopy(model.state_dict())
  epochs_no_improve = 0

  for epoch in range(1, epochs + 1):
    model.train()
    train_sq_err = 0.0
    for xb, yb in tr_loader:
      xb, yb = xb.to(device), yb.to(device)
      optimizer.zero_grad()
      preds = model(xb)
      loss = criterion(preds, yb)
      loss.backward()
      nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
      optimizer.step()
      train_sq_err += loss.item() * len(yb)

    train_rmse = float(np.sqrt(train_sq_err / max(len(y_tr), 1)))
    val_preds = np.clip(predict_sequences(model, X_val, device), 0.0, rul_cap)
    val_rmse = float(root_mean_squared_error(y_val, val_preds))
    scheduler.step(val_rmse)

    if val_rmse < best_val_rmse - 1e-4:
      best_val_rmse = val_rmse
      best_epoch = epoch
      best_weights = copy.deepcopy(model.state_dict())
      epochs_no_improve = 0
    else:
      epochs_no_improve += 1

    if log_every > 0 and (epoch == 1 or epoch % log_every == 0):
      print(
          f"Epoch {epoch:02d}/{epochs:02d} | "
          f"Train RMSE: {train_rmse:.2f} | "
          f"Val RMSE: {val_rmse:.2f}"
      )

    if epochs_no_improve >= patience:
      print(
          f"Early stopping at epoch {epoch} (best epoch: {best_epoch}, "
          f"best val RMSE: {best_val_rmse:.2f})."
      )
      break

  model.load_state_dict(best_weights)
  history_summary = {
      "best_epoch": best_epoch,
      "best_val_rmse": round(best_val_rmse, 4),
  }
  return model, history_summary