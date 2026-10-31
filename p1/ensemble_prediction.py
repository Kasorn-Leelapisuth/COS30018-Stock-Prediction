import os
import matplotlib
matplotlib.use('Agg')  # Headless mode to prevent Mac display errors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, GRU, LSTM
from tensorflow.keras.models import Sequential
import yfinance as yf


# Helper to get cached data or download clean META stock data
def load_meta_data(ticker="META", start_date="2021-01-01", end_date="2023-12-31"):
    cache_path = os.path.join("data", f"{ticker}_task6_{start_date}_{end_date}.csv")
    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
    else:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        os.makedirs("data", exist_ok=True)
        df.to_csv(cache_path)

    df.index = pd.to_datetime(df.index)
    df = df.ffill().bfill().dropna()
    return df


# Prepare sliding window sequences (50 days lookback to predict next-day Close)
def prepare_sequences(prices, n_steps=50):
    X, y = [], []
    for i in range(n_steps, len(prices)):
        X.append(prices[i - n_steps : i])
        y.append(prices[i])
    return np.array(X), np.array(y)


# Build DL model (LSTM or GRU)
def build_dl_model(cell_type="LSTM", n_steps=50, units=50):
    cell = LSTM if cell_type.upper() == "LSTM" else GRU
    model = Sequential([
        cell(units, return_sequences=True, input_shape=(n_steps, 1)),
        Dropout(0.2),
        cell(units, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="huber", metrics=["mae"])
    return model


if __name__ == "__main__":
    print("=== COS30018 Task 6: Ensemble Forecasting (META) ===")
    df = load_meta_data("META", "2021-01-01", "2023-12-31")
    raw_close = df["Close"].values
    n_steps = 50

    # Split into train and test sets chronologically (80/20)
    split_idx = int(len(raw_close) * 0.8)
    train_raw = raw_close[:split_idx]
    test_raw = raw_close[split_idx:]

    print(f"Total trading days: {len(raw_close)} | Train days: {len(train_raw)} | Test days: {len(test_raw)}")

    # 1. Base Model 1: Statistical ARIMA Model
    print("\n[1/5] Fitting Statistical Model: ARIMA(2, 1, 2)...")
    # Fit ARIMA on training data, then apply parameters to the full series
    # to obtain 1-step ahead walk-forward test forecasts efficiently
    arima_order = (2, 1, 2)
    arima_fit = ARIMA(train_raw, order=arima_order).fit()
    full_arima_results = arima_fit.apply(raw_close)
    # Align predictions to the test period
    arima_test_preds = full_arima_results.fittedvalues[split_idx:]

    # 2. Prepare Scaled Sliding Windows for DL & Machine Learning
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_raw.reshape(-1, 1)).flatten()
    test_scaled = scaler.transform(test_raw.reshape(-1, 1)).flatten()

    # Create sequences: to forecast test_raw, we prepend the last n_steps from train
    full_scaled = np.concatenate([train_scaled, test_scaled])
    X_full, y_full = prepare_sequences(full_scaled, n_steps=n_steps)

    # Exact split point for test sequences
    n_train_seq = len(train_raw) - n_steps
    X_train, y_train = X_full[:n_train_seq], y_full[:n_train_seq]
    X_test = X_full[n_train_seq:]
    y_test_real = test_raw  # Ground truth actual test prices

    # Ensure length alignment between ARIMA and ML test samples
    min_len = min(len(y_test_real), len(arima_test_preds), len(X_test))
    y_test_real = y_test_real[:min_len]
    arima_test_preds = arima_test_preds[:min_len]
    X_test = X_test[:min_len]

    # Reshape for recurrent layers: (samples, timesteps, features)
    X_train_dl = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test_dl = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # 3. Base Model 2: LSTM
    print("\n[2/5] Training Deep Learning Model: LSTM...")
    lstm_model = build_dl_model("LSTM", n_steps=n_steps, units=50)
    lstm_model.fit(X_train_dl, y_train, epochs=30, batch_size=32, verbose=0, shuffle=False)
    lstm_pred_scaled = lstm_model.predict(X_test_dl, verbose=0)
    lstm_test_preds = scaler.inverse_transform(lstm_pred_scaled).flatten()

    # 4. Base Model 3: Random Forest Regressor
    print("\n[3/5] Training Machine Learning Model: Random Forest Regressor...")
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred_scaled = rf_model.predict(X_test)
    rf_test_preds = scaler.inverse_transform(rf_pred_scaled.reshape(-1, 1)).flatten()

    # 5. Base Model 4: GRU
 
    print("\n[4/5] Training Alternative Deep Learning Model: GRU...")
    gru_model = build_dl_model("GRU", n_steps=n_steps, units=50)
    gru_model.fit(X_train_dl, y_train, epochs=30, batch_size=32, verbose=0, shuffle=False)
    gru_pred_scaled = gru_model.predict(X_test_dl, verbose=0)
    gru_test_preds = scaler.inverse_transform(gru_pred_scaled).flatten()

    # 6. Ensemble Combinations
   
    print("\n[5/5] Computing Ensemble Predictions...")
    # Ensemble 1: Equal-weight blend (50% ARIMA + 50% LSTM)
    ens_arima_lstm_equal = 0.5 * arima_test_preds + 0.5 * lstm_test_preds

    # Ensemble 2: Weighted blend (30% ARIMA + 70% LSTM)
    ens_arima_lstm_weighted = 0.3 * arima_test_preds + 0.7 * lstm_test_preds

    # Ensemble 3: Multi-paradigm blend (30% ARIMA + 40% LSTM + 30% Random Forest)
    ens_multi_paradigm = 0.3 * arima_test_preds + 0.4 * lstm_test_preds + 0.3 * rf_test_preds

    # Ensemble 4: Statistical + GRU blend (40% ARIMA + 60% GRU)
    ens_arima_gru = 0.4 * arima_test_preds + 0.6 * gru_test_preds

   
    # Evaluate All Models
   
    models_dict = {
        "Base 1: ARIMA(2,1,2)": arima_test_preds,
        "Base 2: LSTM": lstm_test_preds,
        "Base 3: Random Forest": rf_test_preds,
        "Base 4: GRU": gru_test_preds,
        "Ensemble 1: Equal (50% ARIMA + 50% LSTM)": ens_arima_lstm_equal,
        "Ensemble 2: Weighted (30% ARIMA + 70% LSTM)": ens_arima_lstm_weighted,
        "Ensemble 3: Multi-Paradigm (ARIMA+LSTM+RF)": ens_multi_paradigm,
        "Ensemble 4: Hybrid (ARIMA + GRU)": ens_arima_gru,
    }

    summary_rows = []
    for name, preds in models_dict.items():
        mae = mean_absolute_error(y_test_real, preds)
        rmse = np.sqrt(mean_squared_error(y_test_real, preds))
        summary_rows.append({
            "Model / Ensemble Configuration": name,
            "MAE ($)": round(mae, 2),
            "RMSE ($)": round(rmse, 2),
        })

    summary_df = pd.DataFrame(summary_rows)
    print("\nTASK 6 ENSEMBLE EXPERIMENT SUMMARY")
    print(summary_df.to_string(index=False))

    summary_df.to_csv("task6_experiment_results.csv", index=False)
    print("\n[SAVED] Experiment summary table saved to: task6_experiment_results.csv")
    
    
    # Plotting Ground Truth vs Predictions
    plt.figure(figsize=(14, 7))
    plt.plot(y_test_real, label="Actual META Close Price", color="black", linewidth=2.0)
    plt.plot(arima_test_preds, label="Base ARIMA(2,1,2)", color="red", linestyle=":", alpha=0.7)
    plt.plot(lstm_test_preds, label="Base LSTM", color="blue", linestyle="--", alpha=0.7)
    plt.plot(ens_arima_lstm_weighted, label="Ensemble 2: Weighted (30% ARIMA + 70% LSTM)", color="green", linewidth=2.2)
    plt.plot(ens_multi_paradigm, label="Ensemble 3: Multi-Paradigm (ARIMA+LSTM+RF)", color="darkorange", linestyle="-.", linewidth=1.8)

    plt.title("META Stock: Base Models vs Ensemble Methods (Test Horizon)", fontsize=13, fontweight="bold")
    plt.xlabel("Test Set Trading Days", fontsize=11)
    plt.ylabel("Price in USD ($)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper left")
    plt.savefig("task6_ensemble_plot.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("[SAVED] Comparison plot saved to: task6_ensemble_plot.png")
    print("\n[SUCCESS] Task 6 script execution complete.")