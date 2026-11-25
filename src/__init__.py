"""NASA C-MAPSS Remaining Useful Life (RUL) prediction package."""

from src.data import (
    MULTI_REGIME_SUBSETS,
    RegimeAwareScaler,
    add_piecewise_rul,
    build_test_sequences,
    build_train_sequences,
    load_cmapss_subset,
    select_active_features,
)
from src.evaluate import evaluate_predictions, nasa_phm08_score, save_subset_predictions
from src.model import RULSTM1, SequenceRULNet
from src.train import predict_sequences, seed_everything, train_sequence_model

__all__ = [
    "MULTI_REGIME_SUBSETS",
    "RULSTM1",
    "RegimeAwareScaler",
    "SequenceRULNet",
    "add_piecewise_rul",
    "build_test_sequences",
    "build_train_sequences",
    "evaluate_predictions",
    "load_cmapss_subset",
    "nasa_phm08_score",
    "predict_sequences",
    "save_subset_predictions",
    "seed_everything",
    "select_active_features",
    "train_sequence_model",
]