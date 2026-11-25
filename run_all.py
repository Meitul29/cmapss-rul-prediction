import argparse
import os
from typing import List  # noqa: UP035

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit

from src.data import (
    MULTI_REGIME_SUBSETS,
    RegimeAwareScaler,
    add_piecewise_rul,
    build_test_sequences,
    build_train_sequences,
    load_cmapss_subset,
    select_active_features,
)
from src.evaluate import evaluate_predictions, save_subset_predictions
from src.model import SequenceRULNet
from src.train import predict_sequences, seed_everything, train_sequence_model


def run_subset(
    subset: str,
    data_dir: str,
    output_dir: str,
    models: list[str],
    seq_len: int = 30,
    rul_cap: int = 125,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    epochs: int = 40,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    patience: int = 8,
    seed: int = 42,
) -> List[dict]:
    """Run training and evaluation for all requested models on a single subset."""
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df, test_df, rul_df = load_cmapss_subset(data_dir, subset)
    train_df = add_piecewise_rul(train_df, rul_cap=rul_cap)

    # Split strictly by engine unit_id so sensor history never leaks across splits
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr_idx, val_idx = next(
        splitter.split(train_df, train_df["RUL"], train_df["unit_id"])
    )
    tr_df = train_df.iloc[tr_idx].copy()
    val_df = train_df.iloc[val_idx].copy()

    active_features = select_active_features(tr_df, test_df=test_df)
    is_multi_regime = subset.upper() in MULTI_REGIME_SUBSETS
    scaler = RegimeAwareScaler(
        scaler_type="standard",
        multi_regime=is_multi_regime,
        n_regimes=6,
        random_state=seed,
    )

    tr_scaled = scaler.fit_transform(tr_df, active_features)
    val_scaled = scaler.transform(val_df, active_features)
    test_scaled = scaler.transform(test_df, active_features)

    X_tr, y_tr, _ = build_train_sequences(
        tr_scaled, active_features, seq_len=seq_len, pad_short=False
    )
    X_val, y_val, _ = build_train_sequences(
        val_scaled, active_features, seq_len=seq_len, pad_short=False
    )
    X_test, test_unit_ids = build_test_sequences(
        test_scaled, active_features, seq_len=seq_len
    )
    last_cycles = (
        test_df.sort_values(["unit_id", "time_cycles"])
        .groupby("unit_id", sort=True)["time_cycles"]
        .last()
        .to_numpy()
    )
    y_test_raw = rul_df["RUL"].to_numpy(dtype=np.float32)[: len(test_unit_ids)]

    print(
        f"\n=== {subset.upper()} ({'Multi-Regime' if is_multi_regime else 'Single-Regime'}) | "
        f"Features: {len(active_features)} | Train seqs: {len(X_tr)} | "
        f"Val seqs: {len(X_val)} | Test engines: {len(X_test)} ==="
    )

    preds_by_model = {}
    summary_rows = []

    for arch in models:
        arch_upper = arch.upper()
        print(f"\n--- Training {arch_upper} on {subset.upper()} ---")
        model = SequenceRULNet(
            num_features=len(active_features),
            cell_type=arch.lower(),
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
        model, train_info = train_sequence_model(
            model=model,
            X_tr=X_tr,
            y_tr=y_tr,
            X_val=X_val,
            y_val=y_val,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            patience=patience,
            rul_cap=float(rul_cap),
        )

        test_preds = np.clip(
            predict_sequences(model, X_test, device), 0.0, float(rul_cap)
        )
        preds_by_model[arch_upper] = test_preds

        metrics = evaluate_predictions(
            y_test_raw, test_preds, rul_cap=float(rul_cap)
        )
        row = {
            "subset": subset.upper(),
            "model": arch_upper,
            "best_epoch": train_info["best_epoch"],
            "val_rmse": round(train_info["best_val_rmse"], 2),
            **metrics,
        }
        summary_rows.append(row)
        print(
            f"[{subset.upper()} - {arch_upper}] Best Epoch: {row['best_epoch']} | "
            f"Val RMSE: {row['val_rmse']:.2f} | "
            f"Test RMSE (capped): {row['test_rmse_capped']:.2f} | "
            f"PHM08 Score: {row['phm08_score']:.2f}"
        )

    pred_path = save_subset_predictions(
        output_dir=output_dir,
        subset=subset,
        unit_ids=test_unit_ids,
        y_true_raw=y_test_raw,
        preds_by_model=preds_by_model,
        rul_cap=float(rul_cap),
        last_observed_cycles=last_cycles,
    )
    print(f"Saved {subset.upper()} RUL predictions to {pred_path}")
    return summary_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run NASA C-MAPSS RUL prediction across FD001-FD004."
    )
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=["FD001", "FD002", "FD003", "FD004"],
        help="Subsets to run (default: FD001 FD002 FD003 FD004)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["lstm", "gru"],
        help="Sequence architectures to run (default: lstm gru)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Directory containing raw C-MAPSS .txt files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory for prediction CSVs and summary_metrics.csv",
    )
    parser.add_argument("--seq-len", type=int, default=30)
    parser.add_argument("--rul-cap", type=int, default=125)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    all_rows = []
    for subset in args.subsets:
        subset_rows = run_subset(
            subset=subset,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            models=args.models,
            seq_len=args.seq_len,
            rul_cap=args.rul_cap,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            seed=args.seed,
        )
        all_rows.extend(subset_rows)

    summary_df = pd.DataFrame(all_rows)
    summary_path = os.path.join(args.output_dir, "summary_metrics.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved summary metrics to {summary_path}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()