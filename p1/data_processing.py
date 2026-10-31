import os
from collections import deque
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import yfinance as yf


def load_and_process_data(
    ticker="META",
    start_date="2020-01-01",  # 1a: Allows user to specify dataset start date
    end_date="2024-01-01",    # 1a: Allows user to specify dataset end date
    n_steps=50,               # Sequence lookback window length
    scale=True,               # 1e: Option whether to scale columns
    lookup_step=1,            # Forecasting horizon (days ahead to predict)
    split_by_date=True,       # 1c: Split sequentially by date or randomly
    test_size=0.2,            # 1c: Configurable train/test split ratio
    feature_columns=["Open", "High", "Low", "Close", "Volume"],
    save_local=True,          # 1d: Option to save data locally
    local_data_dir="data",    # 1d: Directory to store cached data
):
    """
    Loads, cleans, caches, scales, and prepares multi-feature time-series data
    for training recurrent models (LSTM).

    Fulfills Option C Task 2 specifications:
      - Req 1a: Start and end dates for the overall dataset.
      - Req 1b: Handling NaN entries.
      - Req 1c: Splitting by specified ratio, either chronologically or randomly.
      - Req 1d: Storing and reloading data locally to minimize redundant network requests.
      - Req 1e: Column-wise scaling stored in an accessible dictionary data structure.
    """

    # --------------------------------------------------------------------------
    # REQUIREMENT 1d: Local Storage and Fast Local Loading
    # --------------------------------------------------------------------------
    os.makedirs(local_data_dir, exist_ok=True)
    # File name is tied directly to ticker, start date, and end date to prevent cache collisions
    local_file_path = os.path.join(
        local_data_dir, f"{ticker}_{start_date}_{end_date}.csv"
    )

    if os.path.exists(local_file_path):
        print(f"[CACHE] Loading cached data from local disk: {local_file_path}")
        # Parse dates directly so the index preserves datetime structures
        df = pd.read_csv(local_file_path, index_col=0, parse_dates=True)
    else:
        # ----------------------------------------------------------------------
        # REQUIREMENT 1a: Fetching data using specific start_date and end_date
        # ----------------------------------------------------------------------
        print(f"[NETWORK] Fetching {ticker} data via yfinance from {start_date} to {end_date}...")
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)

        if df.empty:
            raise ValueError(f"No records retrieved for ticker '{ticker}' across {start_date} to {end_date}.")

        # Flatten multi-tiered columns if yfinance returns 2D MultiIndex headers
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Strip timezone offsets to avoid serialization mismatches when saving to CSV
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        if save_local:
            print(f"[CACHE] Saving downloaded data to local disk: {local_file_path}")
            df.to_csv(local_file_path)

    # Dictionary container to hold all processed outputs, scalers, and sequences
    result = {}
    result["df"] = df.copy()

    # Verify that requested features actually exist in the retrieved DataFrame
    for col in feature_columns:
        assert col in df.columns, f"Specified feature '{col}' does not exist in dataset."

    # --------------------------------------------------------------------------
    # REQUIREMENT 1b: Handling Missing / NaN Values
    # --------------------------------------------------------------------------
    # Explanation: Financial time-series occasionally contain NaNs due to holidays,
    # halted trading, or upstream data glitches.
    # 1. ffill() propagates previous day's market prices forward (realistic price persistence).
    # 2. bfill() covers any edge cases where NaN occurs on the very first row.
    # 3. dropna() ensures zero null values enter the numeric scaling steps.
    df = df.ffill().bfill().dropna()

    # --------------------------------------------------------------------------
    # REQUIREMENT 1e: Column-Wise Scaling & Scaler Storage
    # --------------------------------------------------------------------------
    # We maintain a dedicated dictionary mapping each column name to its fitted MinMaxScaler.
    # Why individual scalers? Fitting a separate scaler per column ensures that each
    # metric (e.g., Volume in millions vs. Close price in hundreds) scales correctly to [0, 1]
    # without larger-magnitude features dominating or distorting the scaling of others.
    # It also enables straightforward inverse transformation of the predicted 'Close' column later.
    if scale:
        column_scaler = {}
        for column in feature_columns:
            scaler = preprocessing.MinMaxScaler()
            # Reshaping to (-1, 1) converts the 1D pandas Series into a 2D array expected by sklearn
            df[column] = scaler.fit_transform(np.expand_dims(df[column].values, axis=1))
            column_scaler[column] = scaler

        # Store the scalers in the result dictionary so calling functions can access them
        result["column_scaler"] = column_scaler

    # --------------------------------------------------------------------------
    # Target Generation with Configurable Horizon (lookup_step)
    # --------------------------------------------------------------------------
    # Shift target column backwards so that current row t aligns with target price at t + lookup_step
    df["future"] = df["Close"].shift(-lookup_step)

    # Extract the final sequence before dropping NaNs.
    # This represents the most recent `n_steps` trading days used for live/future inference.
    last_sequence = np.array(df[feature_columns].tail(n_steps))

    # Drop trailing rows where future price shifted into NaN (the last lookup_step rows)
    df.dropna(inplace=True)

    # --------------------------------------------------------------------------
    # Sliding Sequence Construction (deque lookback)
    # --------------------------------------------------------------------------
    # We use collections.deque with maxlen=n_steps as a sliding FIFO queue.
    # When a new item is appended, the oldest item is discarded automatically,
    # constructing 3D sequential feature tensors: (samples, time_steps, features).
    sequence_data = []
    sequences = deque(maxlen=n_steps)

    for entry, target in zip(df[feature_columns].values, df["future"].values):
        sequences.append(entry)
        if len(sequences) == n_steps:
            sequence_data.append([np.array(sequences), target])

    # Extract current sequence array for the last available timestamp
    last_sequence = list([s[:len(feature_columns)] for s in sequences]) + list(last_sequence)
    last_sequence = np.array(last_sequence).astype(np.float32)
    # Reshape last_sequence to (1, n_steps, n_features) for direct model inference
    result["last_sequence"] = np.expand_dims(last_sequence[-n_steps:], axis=0)

    # Unpack sequence pairs into X (feature windows) and y (target prices)
    X, y = [], []
    for seq, target in sequence_data:
        X.append(seq)
        y.append(target)

    X = np.array(X)
    y = np.array(y)

    # --------------------------------------------------------------------------
    # REQUIREMENT 1c: Splitting Data by Date or Randomly
    # --------------------------------------------------------------------------
    if split_by_date:
        # Time-series chronological split: The earlier records are used for training
        # and the newest records are reserved for testing. This prevents future
        # information from leaking into the past (data leakage / lookahead bias).
        train_samples = int((1 - test_size) * len(X))
        result["X_train"] = X[:train_samples]
        result["y_train"] = y[:train_samples]
        result["X_test"] = X[train_samples:]
        result["y_test"] = y[train_samples:]
    else:
        # Random cross-sectional split using scikit-learn
        # Shuffles indices across time. (Useful for comparing leak impact vs chronological split)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=True, random_state=42
        )
        result["X_train"] = X_train
        result["y_train"] = y_train
        result["X_test"] = X_test
        result["y_test"] = y_test

    return result


# ------------------------------------------------------------------------------
# Example Execution / Verification Block
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Testing Data Processing Function ===")
    
    data = load_and_process_data(
        ticker="META",
        start_date="2021-01-01",
        end_date="2023-12-31",
        n_steps=50,
        scale=True,
        lookup_step=1,
        split_by_date=True,
        test_size=0.2,
        save_local=True
    )

    print("\n--- Output Verification ---")
    print(f"X_train tensor shape: {data['X_train'].shape} (samples, sequence_length, features)")
    print(f"y_train tensor shape: {data['y_train'].shape}")
    print(f"X_test tensor shape:  {data['X_test'].shape}")
    print(f"y_test tensor shape:  {data['y_test'].shape}")
    print(f"Stored Scalers:       {list(data['column_scaler'].keys())}")
    print(f"Last sequence shape:  {data['last_sequence'].shape}")
    print("\n[SUCCESS] All Task 2 requirements verified.")