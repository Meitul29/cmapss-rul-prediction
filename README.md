# 🛠️ NASA C-MAPSS Remaining Useful Life (RUL) Prediction

An end-to-end Remaining Useful Life (RUL) prediction pipeline and interactive comparison dashboard for turbofan engines across the **NASA C-MAPSS** subsets (**FD001–FD004**) using **LSTM** and **GRU** sequence models in PyTorch. 🚀

Models are evaluated using **capped test RMSE**, **raw test RMSE**, **validation RMSE**, and the asymmetric **NASA PHM08 score** (lower is better). 📊

---

## 📌 Project Phases

1. **Phase 1 — Data Loading & Sequence Preparation** 📂
   - Ingests train, test, and ground-truth RUL trajectories for `FD001`–`FD004` from `data/`.
   - Builds time-series windows/sequences and prepares target RUL labels for sequence modeling.

2. **Phase 2 — Sequence Model Training (`LSTM` & `GRU`)** 🧠
   - Trains both **LSTM** and **GRU** architectures across all four subsets (`FD001`–`FD004`) with validation tracking and early stopping.

3. **Phase 3 — Evaluation & Metric Export** 📈
   - Evaluates trained models on validation and test sets using capped RMSE, raw RMSE, and the PHM08 score.
   - Exports consolidated comparison metrics to `outputs/summary_metrics.csv` and per-engine predictions to `outputs/predictions/`.

4. **Phase 4 — Interactive Dashboard & Repository Publishing** 🖥️
   - Visualizes subset and model comparisons alongside per-engine predictions through a Streamlit interface (`app.py`).
   - Keeps raw dataset files out of Git via `.gitignore` while preserving the `data/` directory structure with `.gitkeep`.

---

## 🗂️ Repository Structure

```text
cmapss-rul-prediction/
├── data/
│   └── .gitkeep                      # Keeps data/ tracked while ignoring raw dataset files
├── outputs/
│   ├── summary_metrics.csv           # Aggregate metrics across FD001-FD004 (LSTM & GRU)
│   └── predictions/
│       ├── fd001_rul_predictions.csv
│       ├── fd002_rul_predictions.csv
│       ├── fd003_rul_predictions.csv
│       └── fd004_rul_predictions.csv
├── src/                              # Modular pipeline modules (data, models, training, evaluation)
├── app.py                            # Streamlit dashboard for comparing metrics & predictions
├── run_all.py                        # End-to-end training, evaluation, and CSV export runner
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Ignored files (raw data, caches, virtual environments)
└── README.md
```

## ⚙️ Execute the code

```
pip install -r requirements.txt
python run_all.py
```

## 📊 Launch the streamlit app
```
streamlit run app.py
```
