import os
import matplotlib
matplotlib.use('Agg')  # Run headless so it saves all files directly without window crashes
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
import yfinance as yf


# Quick helper to grab META data. Reuses my csv from task 2
def load_stock_data(ticker="META", start_date="2023-01-01", end_date="2023-12-31"):
    cache_path = os.path.join("data", f"{ticker}_{start_date}_{end_date}.csv")
    
    # check if file already exists in data folder
    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
    else:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        # fix header if yfinance gives multi level columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        os.makedirs("data", exist_ok=True)
        df.to_csv(cache_path)

    # mplfinance needs index to be datetime or it throws error
    df.index = pd.to_datetime(df.index)
    # clean missing stuff just in case
    df = df.ffill().bfill().dropna()
    return df


# Task 1: Candlestick chart with n days option
def plot_candlestick(df, n=1, ticker="META", save_fig=True):
    # n must be at least 1 day
    if n < 1:
        raise ValueError("n has to be 1 or bigger")

    # If n is 1, just plot normal daily data
    if n == 1:
        plot_df = df.copy()
        title_suffix = "Daily (n=1)"
    else:
        # Tried pd.resample('5D') first but weekends made empty/NaN candles
        # So instead i group every n rows together using integer division
        group_ids = np.arange(len(df)) // n

        # How to combine n days into one candle:
        # Open = first day open
        # High = highest point in those n days
        # Low = lowest point in those n days
        # Close = last day close
        # Volume = total traded volume
        agg_rules = {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }

        # squash the rows together
        plot_df = df.groupby(group_ids).agg(agg_rules)

        # set the date back to the first date of each chunk so chart shows dates properly
        first_dates = df.groupby(group_ids).apply(lambda x: x.index[0])
        plot_df.index = pd.to_datetime(first_dates.values)

        title_suffix = f"{n} Trading Days combined (n={n})"

    # Setting up the plot with mplfinance
    # type='candle' makes it candlestick instead of line
    # style='charles' gives classic green for up and red for down
    # volume=True adds the bar chart below it
    output_filename = f"{ticker}_candlestick_n{n}.png"
    fig, axes = mpf.plot(
        plot_df,
        type="candle",
        style="charles",
        title=f"{ticker} Candlestick Chart - {title_suffix}",
        ylabel="Price ($)",
        volume=True,
        ylabel_lower="Volume",
        returnfig=True,
        figsize=(12, 6),
    )

    if save_fig:
        fig.savefig(output_filename, dpi=300, bbox_inches="tight")
        print(f"Saved candle chart: {output_filename}")

    plt.show()


# Task 2: Boxplot for moving windows of n days
def plot_moving_window_boxplot(df, n=10, feature="Close", ticker="META", max_windows=15, save_fig=True):
    # need at least 2 points to make a boxplot
    if n < 2:
        raise ValueError("Need n >= 2 for boxplot")

    prices = df[feature].values
    total_len = len(prices)

    window_data = []
    window_labels = []

    # slide window of size n across the data, jumping by n so they don't overlap
    for start in range(0, total_len - n + 1, n):
        end = start + n
        chunk = prices[start:end]
        window_data.append(chunk)

        # date format MM/DD for x axis label
        start_date = df.index[start].strftime("%m/%d")
        end_date = df.index[end - 1].strftime("%m/%d")
        window_labels.append(f"{start_date}-{end_date}")

    # only keep latest windows so x axis doesn't get squished and messy
    if len(window_data) > max_windows:
        window_data = window_data[-max_windows:]
        window_labels = window_labels[-max_windows:]

    # draw boxplot with matplotlib
    fig, ax = plt.subplots(figsize=(13, 6))

    # patch_artist=True fills the box with color
    # showmeans=True adds a line for the average price
    # meanline=True makes mean a line instead of a dot
    ax.boxplot(
        window_data,
        tick_labels=window_labels,
        patch_artist=True,
        showmeans=True,
        meanline=True,
        boxprops=dict(facecolor="lightblue", color="blue"),
        medianprops=dict(color="red", linewidth=2),
        meanprops=dict(color="green", linestyle="--", linewidth=1.5),
        whiskerprops=dict(color="blue", linewidth=1.2),
        capprops=dict(color="blue", linewidth=1.2),
        flierprops=dict(marker="o", color="orange", alpha=0.7),
    )

    ax.set_title(f"{ticker} Boxplot - {feature} Price ({n}-Day Windows)", fontsize=13)
    ax.set_xlabel("Date Range", fontsize=11)
    ax.set_ylabel(f"{feature} Price ($)", fontsize=11)
    ax.grid(axis="y", linestyle=":", alpha=0.7)
    plt.xticks(rotation=45, ha="right")

    # dummy lines just to show red is median and green is mean in legend
    ax.plot([], [], color="red", linewidth=2, label="Median")
    ax.plot([], [], color="green", linestyle="--", linewidth=1.5, label="Mean")
    ax.legend(loc="upper left")

    output_filename = f"{ticker}_boxplot_window_n{n}.png"
    if save_fig:
        fig.savefig(output_filename, dpi=300, bbox_inches="tight")
        print(f"Saved boxplot chart: {output_filename}")

    plt.tight_layout()
    plt.show()


# test run
if __name__ == "__main__":
    print("Testing charts...")
    meta_df = load_stock_data(ticker="META", start_date="2023-01-01", end_date="2023-12-31")

    # normal daily candle
    plot_candlestick(meta_df, n=1, ticker="META")

    # 5 day combined candle (1 week trading)
    plot_candlestick(meta_df, n=5, ticker="META")

    # boxplot with 10-day windows
    plot_moving_window_boxplot(meta_df, n=10, feature="Close", ticker="META")

    print("Done! Check saved png files.")