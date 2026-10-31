import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, GRU, LSTM, Bidirectional, SimpleRNN
from tensorflow.keras.models import Sequential

# Reusing the data loading function we built in Task 2
from data_processing import load_and_process_data


# Task 4 - Requirement 1: Function to dynamically construct Deep Learning models
def create_model(
    sequence_length=50,
    n_features=5,
    cell_type="LSTM",
    n_layers=2,
    units=50,
    dropout_rate=0.2,
    bidirectional=False,
    loss="huber",
    optimizer="adam",
):
    """
    Builds a flexible sequential recurrent neural network based on input parameters.
    - cell_type: 'LSTM', 'GRU', or 'RNN'
    - n_layers: number of recurrent layers stacked together
    - units: number of neurons inside each recurrent layer
    - dropout_rate: percentage of neurons to randomly drop out to prevent overfitting
    - bidirectional: whether to read sequence forward and backward
    """
    # Mapping string names to actual Keras recurrent layer classes
    layer_mapping = {
        "LSTM": LSTM,
        "GRU": GRU,
        "RNN": SimpleRNN,
        "SIMPLERNN": SimpleRNN,
    }

    selected_cell = layer_mapping.get(cell_type.upper())
    if selected_cell is None:
        raise ValueError(f"Unsupported cell_type: {cell_type}. Choose LSTM, GRU, or RNN.")

    model = Sequential()

    for i in range(n_layers):
        # In Keras, stacked recurrent layers need return_sequences=True for every layer
        # except the last one. The last layer only outputs the final hidden vector.
        is_last_layer = (i == n_layers - 1)
        return_sequences = not is_last_layer

        # First layer must know the input shape: (sequence_length, n_features)
        if i == 0:
            layer = selected_cell(
                units,
                return_sequences=return_sequences,
                input_shape=(sequence_length, n_features),
            )
        else:
            layer = selected_cell(units, return_sequences=return_sequences)

        # Optional bidirectional wrapper
        if bidirectional:
            layer = Bidirectional(layer)

        model.add(layer)
        # Dropout helps regularize the model and stops it memorizing noise
        model.add(Dropout(dropout_rate))

    # Final dense layer outputs 1 single number: the predicted next Close price
    model.add(Dense(1, activation="linear"))

    # Huber loss is used because it behaves like MSE for small errors
    # but behaves like MAE for large outliers, making it stable for stock swings
    model.compile(loss=loss, metrics=["mean_absolute_error"], optimizer=optimizer)

    return model


# Helper function to train and evaluate a single model configuration
def run_experiment(exp_name, data, cell_type, n_layers, units, epochs, batch_size, dropout=0.2):
    print(f"\n=======================================================")
    print(f"Running Experiment: {exp_name}")
    print(f"Config: {cell_type} | Layers={n_layers} | Units={units} | Epochs={epochs} | Batch={batch_size}")
    print(f"=======================================================")

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    # Build model using our dynamic function
    model = create_model(
        sequence_length=X_train.shape[1],
        n_features=X_train.shape[2],
        cell_type=cell_type,
        n_layers=n_layers,
        units=units,
        dropout_rate=dropout,
    )

    # Train the model
    history = model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_test, y_test),
        verbose=0,
        shuffle=False,  # Keep chronological order intact
    )

    # Make predictions on test set
    y_pred_scaled = model.predict(X_test, verbose=0)

    # Inverse transform predictions back to real dollar values using our scaler dictionary from Task 2
    close_scaler = data["column_scaler"]["Close"]
    y_test_real = close_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_real = close_scaler.inverse_transform(y_pred_scaled).flatten()

    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test_real, y_pred_real)
    mse = mean_squared_error(y_test_real, y_pred_real)
    rmse = np.sqrt(mse)

    print(f"Results for {exp_name}:")
    print(f"  Test MAE:  ${mae:.2f}")
    print(f"  Test RMSE: ${rmse:.2f}")

    return {
        "Experiment": exp_name,
        "Cell Type": cell_type,
        "Layers": n_layers,
        "Units": units,
        "Epochs": epochs,
        "Batch Size": batch_size,
        "MAE ($)": round(mae, 2),
        "RMSE ($)": round(rmse, 2),
    }


if __name__ == "__main__":
    print("Loading META stock data (2021-01-01 to 2023-12-31)...")
    # Using 5 features: Open, High, Low, Close, Volume
    data = load_and_process_data(
        ticker="META",
        start_date="2021-01-01",
        end_date="2023-12-31",
        n_steps=50,
        lookup_step=1,
        test_size=0.2,
        feature_columns=["Open", "High", "Low", "Close", "Volume"],
        split_by_date=True,
    )

    results = []

    # -------------------------------------------------------------
    # Experiment Set 1: Comparing Architectures (LSTM vs GRU vs RNN)
    # -------------------------------------------------------------
    results.append(run_experiment("Exp 1: Base LSTM", data, "LSTM", n_layers=2, units=50, epochs=25, batch_size=32))
    results.append(run_experiment("Exp 2: Base GRU", data, "GRU", n_layers=2, units=50, epochs=25, batch_size=32))
    results.append(run_experiment("Exp 3: Base SimpleRNN", data, "RNN", n_layers=2, units=50, epochs=25, batch_size=32))

    # -------------------------------------------------------------
    # Experiment Set 2: Hyperparameter Variations (Depth, Epochs, Batch)
    # -------------------------------------------------------------
    results.append(run_experiment("Exp 4: Deep LSTM (3 Layers)", data, "LSTM", n_layers=3, units=50, epochs=25, batch_size=32))
    results.append(run_experiment("Exp 5: High Epochs (50 Epochs)", data, "LSTM", n_layers=2, units=50, epochs=50, batch_size=32))
    results.append(run_experiment("Exp 6: Small Batch (Batch 16)", data, "LSTM", n_layers=2, units=50, epochs=25, batch_size=16))

    # Summary Table
    summary_df = pd.DataFrame(results)
    print("\n=======================================================")
    print("FINAL EXPERIMENT SUMMARY TABLE")
    print("=======================================================")
    print(summary_df.to_string(index=False))

    # Save summary table to CSV for the report
    summary_df.to_csv("task4_experiment_results.csv", index=False)
    print("\nSaved experiment table to: task4_experiment_results.csv")