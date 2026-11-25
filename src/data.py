import os
from typing import Dict, List, Tuple  # noqa: UP035

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, StandardScaler

INDEX_COLS = ["unit_id", "time_cycles"]
SETTING_COLS = ["op_setting_1", "op_setting_2", "op_setting_3"]
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
ALL_COLS = INDEX_COLS + SETTING_COLS + SENSOR_COLS

MULTI_REGIME_SUBSETS = {"FD002", "FD004"}


def load_cmapss_subset(
    data_dir: str, subset: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, test, and ground-truth RUL files for a C-MAPSS subset."""
    subset = subset.upper()
    train_path = os.path.join(data_dir, f"train_{subset}.txt")
    test_path = os.path.join(data_dir, f"test_{subset}.txt")
    rul_path = os.path.join(data_dir, f"RUL_{subset}.txt")

    for path in (train_path, test_path, rul_path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing dataset file: {path}. Place NASA C-MAPSS .txt files in '{data_dir}'."
            )

    train_df = pd.read_csv(
        train_path, sep=r"\s+", header=None, names=ALL_COLS
    ).dropna(axis=1, how="all")
    test_df = pd.read_csv(
        test_path, sep=r"\s+", header=None, names=ALL_COLS
    ).dropna(axis=1, how="all")
    rul_df = pd.read_csv(
        rul_path, sep=r"\s+", header=None, names=["RUL"]
    ).dropna(axis=1, how="all")
    return train_df, test_df, rul_df


def add_piecewise_rul(df: pd.DataFrame, rul_cap: int = 125) -> pd.DataFrame:
    """Add raw and piecewise-capped Remaining Useful Life targets per engine."""
    out = df.copy()
    max_cycles = out.groupby("unit_id")["time_cycles"].transform("max")
    out["RUL_raw"] = max_cycles - out["time_cycles"]
    out["RUL"] = out["RUL_raw"].clip(lower=0, upper=rul_cap)
    return out


def select_active_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame = None,
    std_threshold: float = 1e-6,
) -> List[str]:
    """Select non-constant operational and sensor features from training data."""
    candidate_cols = SETTING_COLS + SENSOR_COLS
    stds = train_df[candidate_cols].std(ddof=0)
    active = [col for col in candidate_cols if stds[col] > std_threshold]
    if test_df is not None:
        active = [col for col in active if test_df[col].notna().all()]
    return active


class RegimeAwareScaler:
    """Scale features globally or per operating regime for multi-regime subsets."""

    def __init__(
        self,
        scaler_type: str = "standard",
        multi_regime: bool = False,
        n_regimes: int = 6,
        random_state: int = 42,
    ):
        self.scaler_type = scaler_type.lower()
        self.multi_regime = multi_regime
        self.n_regimes = n_regimes
        self.random_state = random_state
        self.kmeans = None
        self.scalers: Dict[int, object] = {}
        self.global_scaler = self._make_scaler()

    def _make_scaler(self):
        if self.scaler_type == "minmax":
            return MinMaxScaler()
        return StandardScaler()

    def fit_transform(
        self, df: pd.DataFrame, feature_cols: List[str]
    ) -> pd.DataFrame:
        out = df.copy()
        if not self.multi_regime:
            out[feature_cols] = self.global_scaler.fit_transform(
                out[feature_cols]
            )
            return out

        self.kmeans = KMeans(
            n_clusters=self.n_regimes, n_init=10, random_state=self.random_state
        )
        regimes = self.kmeans.fit_predict(out[SETTING_COLS])
        out["_regime"] = regimes

        for r in range(self.n_regimes):
            mask = out["_regime"] == r
            if mask.any():
                scaler = self._make_scaler()
                out.loc[mask, feature_cols] = scaler.fit_transform(
                    out.loc[mask, feature_cols]
                )
                self.scalers[r] = scaler

        return out.drop(columns=["_regime"])

    def transform(
        self, df: pd.DataFrame, feature_cols: List[str]
    ) -> pd.DataFrame:
        out = df.copy()
        if not self.multi_regime:
            out[feature_cols] = self.global_scaler.transform(out[feature_cols])
            return out

        regimes = self.kmeans.predict(out[SETTING_COLS])
        out["_regime"] = regimes

        for r in range(self.n_regimes):
            mask = out["_regime"] == r
            if mask.any() and r in self.scalers:
                out.loc[mask, feature_cols] = self.scalers[r].transform(
                    out.loc[mask, feature_cols]
                )

        return out.drop(columns=["_regime"])


def build_train_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "RUL",
    seq_len: int = 30,
    pad_short: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build sliding sequence windows per engine for training/validation."""
    X_list, y_list, unit_list = [], [], []
    ordered = df.sort_values(["unit_id", "time_cycles"])

    for unit_id, grp in ordered.groupby("unit_id", sort=False):
        feats = grp[feature_cols].to_numpy(dtype=np.float32)
        targets = grp[target_col].to_numpy(dtype=np.float32)
        n_rows = len(feats)

        if n_rows < seq_len:
            if not pad_short:
                continue
            pad_block = np.repeat(feats[:1], seq_len - n_rows, axis=0)
            X_list.append(np.vstack([pad_block, feats]))
            y_list.append(targets[-1])
            unit_list.append(unit_id)
        else:
            for end_idx in range(seq_len - 1, n_rows):
                X_list.append(feats[end_idx - seq_len + 1 : end_idx + 1])
                y_list.append(targets[end_idx])
                unit_list.append(unit_id)

    return (
        np.stack(X_list, axis=0),
        np.array(y_list, dtype=np.float32),
        np.array(unit_list),
    )


def build_test_sequences(
    df: pd.DataFrame, feature_cols: List[str], seq_len: int = 30
) -> Tuple[np.ndarray, np.ndarray]:
    """Build the final `seq_len` sequence per test engine, front-padding if needed."""
    X_list, unit_ids = [], []
    ordered = df.sort_values(["unit_id", "time_cycles"])

    for unit_id, grp in ordered.groupby("unit_id", sort=True):
        feats = grp[feature_cols].to_numpy(dtype=np.float32)
        if len(feats) >= seq_len:
            window = feats[-seq_len:]
        else:
            pad_block = np.repeat(feats[:1], seq_len - len(feats), axis=0)
            window = np.vstack([pad_block, feats])
        X_list.append(window)
        unit_ids.append(unit_id)

    return np.stack(X_list, axis=0), np.array(unit_ids)