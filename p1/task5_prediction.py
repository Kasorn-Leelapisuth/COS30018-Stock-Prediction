
import os
import matplotlib
matplotlib.use('Agg')  # Run in background to prevent macOS GUI window crashes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential
import yfinance as yf


# Helper: load cached CSV if present, otherwise download via yfinance
def get_stock_data(ticker="META", start_date="2021-01-01", end_date="2023-12-31"):
    cache_file = os.path.join("data", f"{ticker}_task5_{start_date}_{end_date}.csv")
    if os.path.exists(cache_file):
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
    else:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        os.makedirs("data", exist_ok=True)
        df.to_csv(cache_file)

    df.index = pd.to_datetime(df.index)
    df = df.ffill().bfill().dropna()
    return df


# Prepares sliding windows for any feature set and any forecast horizon k
def prepare_dataset(
    df,
    feature_cols=["Open", "High", "Low", "Close", "Adj Close", "Volume"],
    target_col="Close",
    n_steps=50,
    k_future=5,
    test_ratio=0.2,
):
    # Scale each feature independently so Volume doesn't dwarf price columns
    scalers = {}
    scaled_df = pd.DataFrame(index=df.index)

    for col in feature_cols:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_df[col] = scaler.fit_transform(df[[col]])
        scalers[col] = scaler

    target_scaler = scalers[target_col]
    feature_values = scaled_df[feature_cols].values
    target_values = scaled_df[target_col].values

    X, y = [], []
    total_len = len(df)

    # Sliding window: past n_steps days to predict the next k_future days
    for i in range(n_steps, total_len - k_future + 1):
        X.append(feature_values[i - n_steps : i])
        y.append(target_values[i : i + k_future])

    X = np.array(X)
    y = np.array(y)

    # Chronological train/test split to prevent lookahead bias
    train_size = int(len(X) * (1 - test_ratio))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    return X_train, y_train, X_test, y_test, target_scaler


# Deep Learning model architecture supporting vector output for k future days
def build_model(input_steps, n_features, output_steps, units=50):
    model = Sequential(
        [
            LSTM(
                units,
                return_sequences=True,
                input_shape=(input_steps, n_features),
            ),
            Dropout(0.2),
            LSTM(units, return_sequences=False),
            Dropout(0.2),
            # Final Dense layer has output_steps neurons (k future days)
            Dense(output_steps),
        ]
    )
    model.compile(optimizer="adam", loss="huber", metrics=["mae"])
    return model


# Trains the network and calculates MAE and RMSE in real US dollars
def evaluate_experiment(name, X_train, y_train, X_test, y_test, target_scaler):
    print(f"\nRunning: {name}")
    print(f"Input shape: {X_train.shape}, Target shape: {y_train.shape}")

    model = build_model(
        input_steps=X_train.shape[1],
        n_features=X_train.shape[2],
        output_steps=y_train.shape[1],
        units=50,
    )

    model.fit(
        X_train,
        y_train,
        epochs=30,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=0,
        shuffle=False,
    )

    # Predict on test set
    y_pred_scaled = model.predict(X_test, verbose=0)

    # Invert predictions and actual values back to real dollar amounts
    y_test_real = target_scaler.inverse_transform(y_test)
    y_pred_real = target_scaler.inverse_transform(y_pred_scaled)

    mae = mean_absolute_error(y_test_real, y_pred_real)
    rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))

    print(f"Results for {name}:")
    print(f"  MAE:  ${mae:.2f}")
    print(f"  RMSE: ${rmse:.2f}")

    return {
        "Model Configuration": name,
        "Input Features": X_train.shape[2],
        "Forecast Horizon (k)": y_train.shape[1],
        "MAE ($)": round(mae, 2),
        "RMSE ($)": round(rmse, 2),
    }, y_test_real, y_pred_real


if __name__ == "__main__":
    print("Loading META stock data (2021-01-01 to 2023-12-31)...")
    df = get_stock_data(
        ticker="META", start_date="2021-01-01", end_date="2023-12-31"
    )

    # Ensure all required assignment columns are present
    required_cols = [
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
    ]
    for col in required_cols:
        if col not in df.columns:
            df["Adj Close"] = df["Close"]

    results = []

    # Requirement 1: Univariate Multi-step (Close only, predict k = 5 days)
    X_tr1, y_tr1, X_te1, y_te1, scaler1 = prepare_dataset(
        df,
        feature_cols=["Close"],
        target_col="Close",
        n_steps=50,
        k_future=5,
    )
    res1, _, _ = evaluate_experiment(
        "1. Univariate Multi-step (k=5)",
        X_tr1,
        y_tr1,
        X_te1,
        y_te1,
        scaler1,
    )
    results.append(res1)

    # Requirement 2: Multivariate Single-step (all 6 features, predict k = 1 day)
    X_tr2, y_tr2, X_te2, y_te2, scaler2 = prepare_dataset(
        df,
        feature_cols=required_cols,
        target_col="Close",
        n_steps=50,
        k_future=1,
    )
    res2, _, _ = evaluate_experiment(
        "2. Multivariate Single-step (k=1)",
        X_tr2,
        y_tr2,
        X_te2,
        y_te2,
        scaler2,
    )
    results.append(res2)

    #Requirement 3: Multivariate Multi-step (all 6 features, predict k = 5 days)
    X_tr3, y_tr3, X_te3, y_te3, scaler3 = prepare_dataset(
        df,
        feature_cols=required_cols,
        target_col="Close",
        n_steps=50,
        k_future=5,
    )
    res3, y_true_multi, y_pred_multi = evaluate_experiment(
        "3. Multivariate Multi-step (k=5)",
        X_tr3,
        y_tr3,
        X_te3,
        y_te3,
        scaler3,
    )
    results.append(res3)

    # Display final results table
    summary_df = pd.DataFrame(results)
    print("\nTASK 5 FINAL EXPERIMENT SUMMARY")
    print(summary_df.to_string(index=False))

    summary_df.to_csv("task5_experiment_results.csv", index=False)
    print("\n[SAVED] Experiment summary saved to: task5_experiment_results.csv")

    # Generate and save actual vs predicted plot for the Multivariate Multi-step model
    plt.figure(figsize=(12, 6))
    plt.plot(
        y_true_multi[:, 0],
        label="Actual Next-Day Close",
        color="black",
        linewidth=1.5,
    )
    plt.plot(
        y_pred_multi[:, 0],
        label="Predicted Next-Day Close (Multivariate Multi-step)",
        color="royalblue",
        linestyle="--",
    )
    plt.title(
        "META Stock: Multivariate Multi-step Prediction (Test Set Evaluation)",
        fontsize=13,
    )
    plt.xlabel("Test Samples (Trading Days)", fontsize=11)
    plt.ylabel("Price (USD)", fontsize=11)
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.savefig(
        "task5_multivariate_multistep_plot.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print("[SAVED] Plot saved to: task5_multistep_plot.png")