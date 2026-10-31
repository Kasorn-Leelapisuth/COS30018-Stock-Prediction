# COS30018 Intelligent Systems - Stock Price Prediction and Classification Pipeline

An end to end financial machine learning and time series pipeline developed for Meta Platforms, Inc. (META) and Apple Inc. (AAPL) stock data, spanning continuous price forecasting to multimodal directional movement classification.

## Author Information
- Student Name: Kasorn Leelapisuth (Kay)
- Student ID: 105963593
- Unit Code: COS30018 Intelligent Systems
- Repository: https://github.com/Kasorn-Leelapisuth/COS30018-Stock-Prediction

---

## Project Structure Overview (Tasks 1 to 7)

```text
COS30018-Task1/
├── p1/
│   ├── data_processing.py            # Task 1 and 2: Ingestion, scaling, and sequences
│   ├── data_visualization.py         # Task 2: Candlesticks and rolling boxplots
│   ├── model_training.py             # Task 3 and 4: Baseline, single step LSTM and GRU
│   ├── task5_prediction.py           # Task 5: Multivariate and multi step forecasting
│   ├── ensemble_prediction.py        # Task 6: ARIMA and Deep Learning ensembles
│   └── sentiment_classification.py   # Task 7: Sentiment enhanced binary classification
├── task4_experiment_results.csv      # Task 4 evaluation metrics
├── task5_experiment_results.csv      # Task 5 evaluation metrics
├── task6_experiment_results.csv      # Task 6 ensemble metrics
├── task5_multivariate_multistep_plot.png
├── task6_ensemble_plot.png
├── task7_confusion_matrices.png
└── README.md
```
### Tasks Summary
**Task 1:** Environment Setup and Historical Data Ingestion
Configured a Python virtual environment (stock_env) with native Apple Silicon arm64 architecture.

Built automated pipelines via yfinance to ingest daily OHLCV equity data for META and AAPL across 2021-01-01 to 2023-12-31.

Implemented sequence windowing with forward and backward fill to preserve chronological market ordering without data leakage.

**Task 2:** Exploratory Data Analysis and Financial Feature Engineering
Implemented financial visualization tools using mplfinance and matplotlib:

1 day and 5 day candlestick representations highlighting market regimes and volume breakouts.

Rolling 10 day volatility boxplots capturing distributional dispersion.

Computed moving averages and normalization scalers (MinMaxScaler) for downstream deep learning models.

**Task 3:** Deep Recurrent Neural Networks (Baseline Modeling)
Designed univariate single step forecasting architectures using Recurrent Neural Networks (RNN) and Long Short Term Memory (LSTM) cells.

Trained sequential models on lookback windows of 50 trading days to predict the next closing price.

**Task 4:** Advanced Architectures and Hyperparameter Optimization
Expanded recurrent architectures to Gated Recurrent Units (GRU) and stacked deep LSTM configurations.

Conducted hyperparameter tuning across hidden units, dropout rates, learning rates, and batch sizes.

Evaluated performance using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Mean Absolute Percentage Error (MAPE).

**Task 5:** Multivariate and Multi Step Sequence Forecasting
Extended models to ingest multivariate inputs: Open, High, Low, Close, Volume, and technical oscillators (RSI, SMA).

Implemented direct multi step forecasting outputs (k equals 5 and k equals 10 trading days ahead) to capture multi day trajectory paths rather than single step shifts.

**Task 6:** Hybrid Multi Paradigm Ensemble Modeling
Integrated linear statistical time series modeling with non linear neural regressors:

Statistical: Autoregressive Integrated Moving Average ARIMA(2, 1, 2) trained on raw dollar series.

Deep Learning: LSTM and GRU recurrent networks trained on normalized features.

Machine Learning: Random Forest regressors.

Combined predictions using weighted ensemble blending, reducing single model tracking lag and achieving an ensemble MAE of 8.57 dollars.

**Task 7:** Sentiment Based Stock Price Movement Prediction (Extension)
Reframed equity forecasting into a binary classification task: predicting whether Close on day t plus 1 is greater than Close on day t (UP) or less than or equal to Close on day t (DOWN).

Independent Research Component: Developed a customized financial VADER sentiment analyzer by injecting domain specific weighting for corporate earnings and market terminology (such as surge, rally, outperform, layoffs).

Evaluation: Compared a technical baseline Random Forest (50.68 percent accuracy) against a sentiment enhanced classifier (50.00 percent accuracy, 53.62 percent precision), noting reduced false positive buy signals.

Real World Live API Demo: Built real time inference connecting to Yahoo Finance live endpoints to poll breaking news, score sentiment, compute live RSI, and issue directional predictions for upcoming sessions.

Installation and Environment Setup
``` bash
# Clone the repository
git clone [https://github.com/Kasorn-Leelapisuth/COS30018-Stock-Prediction.git](https://github.com/Kasorn-Leelapisuth/COS30018-Stock-Prediction.git)
cd COS30018-Stock-Prediction

# Create and activate virtual environment
python3 -m venv stock_env
source stock_env/bin/activate

# Upgrade pip and install core dependencies
pip install --upgrade pip
pip install yfinance nltk scikit-learn pandas numpy matplotlib seaborn statsmodels mplfinance certifi

# Download NLTK VADER lexicon with SSL handling
python -c "import ssl, nltk; ssl._create_default_https_context = ssl._create_unverified_context; nltk.download('vader_lexicon')"
```

Execution Guide
Run Task 5 (Multivariate and Multi Step Forecasting)
``` bash
python p1/task5_prediction.py
```
Run Task 6 (Hybrid Ensemble Modeling)
``` bash
python p1/ensemble_prediction.py
```
Run Task 7 (Sentiment Classification and Live API Inference)
``` bash
python p1/sentiment_classification.py
```
