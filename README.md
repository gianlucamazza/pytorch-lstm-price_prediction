# [PYTORCH] Price Prediction with LSTM

## Overview

This project aims to predict stock prices using Long Short-Term Memory (LSTM) networks.
The model is trained on historical stock data, which includes various technical indicators.
The pipeline includes data preprocessing, feature engineering, model training, and prediction.

## Pipeline

- **Data Loading**: Load historical stock data using the `yfinance` library.
- **Data Preprocessing**: Preprocess the data by filling missing values and resampling the data.
- **Feature Engineering**: Calculate technical indicators and add them to the dataset.
- **Feature Selection**: Select the best features using the `feature_engineering.py` script.
- **Model Training**: Train the LSTM model using the selected features.
- **Model Evaluation**: Evaluate the model using the Mean Squared Error (MSE) and Mean Absolute Error (MAE) metrics.
- **Prediction**: Generate predictions using the trained model.
- **Hyperparameters Optimization**: Optimize the hyperparameters of the LSTM model using the Optuna library.
- **Backtesting**: Backtest the model on historical data to evaluate its performance.
- **Visualization**: Plot the predictions and evaluation metrics using plotly.
- **Logging**: Log the training and prediction results using the `logging` library.

## Data

The historical stock data is loaded using the `yfinance` library.
The data includes the following columns:

- `Date`: The date of the stock price.
- `Open`: The opening price of the stock.
- `High`: The highest price of the stock.
- `Low`: The lowest price of the stock.
- `Close`: The closing price of the stock.
- `Adj Close`: The adjusted closing price of the stock.
- `Volume`: The volume of the stock traded.

## Setup

### Requirements

Ensure you have Python 3.11 installed. Install the necessary packages using:

```bash
pip install -r requirements.txt
```

## Usage

### Configuration

Update the `config.json` file with the desired parameters.

```json
{
    "data_settings": {
        "ticker": "BTC-USD",
        "symbol": "BTCUSD",
        "historical_data_path": "data/BTCUSD_1d_historical.csv",
        "asset_type": "commodity",
        "data_sampling_interval": "1d",
        "start_date": "2013-01-01",
        "data_resampling_frequency": "D",
        "technical_indicators": {
            "SMA_50": 50,
            "SMA_200": 200,
            "EMA": 20,
            "MACD": [12, 26, 9],
            "RSI": 14,
            "Bollinger": 20,
            "ATR": 14,
            "Stochastic": 14,
            "Aroon": 25,
            "Williams_R": 14,
            "CMF": 20,
            "CCI": 20,
            "Ichimoku": {
                "window1": 9,
                "window2": 26,
                "window3": 52
            }
        },
        "targets": ["Open", "High", "Low", "Close", "Volume"],
        "disabled_features": ["Adj Close"]
    },
    "model_settings": {
        "hidden_size": 128,
        "num_layers": 3,
        "dropout": 0.3,
        "learning_rate": 0.002,
        "fc_output_size": 5,
        "weight_decay": 0.01
    },
    "training_settings": {
        "look_back": 120,
        "look_forward": 30,
        "epochs": 150,
        "batch_size": 32,
        "model_dir": "models"
    },
    "logging_settings": {
        "log_dir": "logs"
    },
    "backtesting_params": {
        "initial_cash": 100000,
        "trading_fee": 0.001,
        "take_profit": 1.5,
        "stop_loss": 1.2,
        "trade_allocation": 0.1,
        "max_open_trades": 5
    },
    "feature_settings": {
        "selected_features": [
            "Bollinger_High",
            "Bollinger_Low",
            "Ichimoku_Tenkan",
            "Ichimoku_Kijun",
            "Ichimoku_Senkou_Span_A",
            "Ichimoku_Senkou_Span_B"
        ]
    }
}
```

- `data_settings`: Dictionary containing data-related settings.
  - `ticker`: Stock ticker symbol.
  - `symbol`: Stock symbol.
  - `historical_data_path`: Path to the historical data CSV file.
  - `asset_type`: Type of asset, e.g., 'commodity'.
  - `data_sampling_interval`: Interval for the historical data, e.g., '1d' for daily or '1h' for hourly.
  - `start_date`: Start date for the historical data.
  - `data_resampling_frequency`: Frequency of the data, e.g., 'D' for daily.
  - `technical_indicators`: Dictionary containing the window sizes for the technical indicators.
    - `SMA_50`: Window size for the 50-day Simple Moving Average.
    - `SMA_200`: Window size for the 200-day Simple Moving Average.
    - `EMA`: Window size for the Exponential Moving Average.
    - `MACD`: List of three values representing the windows for the MACD indicator.
    - `RSI`: Window size for the Relative Strength Index.
    - `Bollinger`: Window size for the Bollinger Bands.
    - `ATR`: Window size for the Average True Range.
    - `Stochastic`: Window size for the Stochastic Oscillator.
    - `Aroon`: Window size for the Aroon indicator.
    - `Williams_R`: Window size for the Williams %R indicator.
    - `CMF`: Window size for the Chaikin Money Flow indicator.
    - `CCI`: Window size for the Commodity Channel Index.
    - `Ichimoku`: Dictionary containing the window sizes for the Ichimoku Cloud.
      - `window1`: Window size for the Tenkan-sen.
      - `window2`: Window size for the Kijun-sen.
      - `window3`: Window size for the Senkou Span B.
  - `targets`: List of target variables for prediction.
  - `disabled_features`: List of features to be disabled.

- `model_settings`: Dictionary containing model parameters.
  - `hidden_size`: Number of hidden units in the LSTM layer.
  - `num_layers`: Number of LSTM layers.
  - `dropout`: Dropout rate.
  - `learning_rate`: Learning rate for the optimizer.
  - `fc_output_size`: Output size of the fully connected layer.
  - `weight_decay`: Weight decay for the optimizer.

- `training_settings`: Dictionary containing training parameters.
  - `look_back`: Number of days to look back for the LSTM model.
  - `look_forward`: Number of days to predict.
  - `epochs`: Number of epochs for training.
  - `batch_size`: Batch size for training.
  - `model_dir`: Directory to store the trained model.

- `logging_settings`: Dictionary containing logging settings.
  - `log_dir`: Directory to store the logs.

- `backtesting_params`: Dictionary containing backtesting parameters.
  - `initial_cash`: Initial cash for backtesting.
  - `trading_fee`: Trading fee for backtesting.
  - `take_profit`: Take profit ratio.
  - `stop_loss`: Stop loss ratio.
  - `trade_allocation`: Trade allocation percentage.
  - `max_open_trades`: Maximum number of open trades.

- `feature_settings`: Dictionary containing feature selection settings.
  - `selected_features`: List of best features selected for training from the feature selection process.

If `selected_features` is not present in the config file, the feature discovery process will be executed.

### Features

The model uses various technical indicators as features for training. The feature engineering script calculates these indicators from historical data based on the asset type.

The following technical indicators are used:

#### Simple Moving Average (SMA)
- **SMA 50**: The simple moving average calculated over the past 50 periods. It indicates the medium-term trend.
- **SMA 200**: The simple moving average calculated over the past 200 periods. It indicates the long-term trend.
- **SMA Agg**: The average of SMA 50 and SMA 200, used to reduce dimensionality and retain crucial information.

#### Exponential Moving Average (EMA)
- **EMA**: The exponential moving average calculated over a user-defined window (typically 20 periods). It gives more weight to recent prices compared to the SMA.

#### Moving Average Convergence Divergence (MACD)
- **MACD**: The difference between the 12-period EMA and the 26-period EMA. Used to identify trends and turning points.
- **MACD Signal**: The 9-period EMA of the MACD. It serves as a signal for trading operations.
- **MACD Agg**: The average of MACD and MACD Signal, used to aggregate information.

#### Relative Strength Index (RSI)
- **RSI**: A momentum indicator that measures the speed and change of price movements. Calculated over a 14-period window, it indicates whether an asset is overbought or oversold.

#### Bollinger Bands
- **Bollinger High**: The upper band calculated as the 20-period simple moving average (SMA) plus two standard deviations.
- **Bollinger Low**: The lower band calculated as the 20-period SMA minus two standard deviations.
- **Bollinger Bandwidth**: The difference between Bollinger High and Bollinger Low, used to measure volatility.

#### Average True Range (ATR)
- **ATR**: A measure of volatility calculated as the exponential moving average (EMA) of the true range over a defined period (typically 14 periods).

#### Stochastic Oscillator
- **Stochastic**: Indicates the closing position relative to the high-low range over a 14-period window. Used to identify overbought and oversold conditions.

#### Aroon Indicator
- **Aroon Up**: Calculated over a 25-period window, it measures the number of periods since the last highest high.
- **Aroon Down**: Calculated over a 25-period window, it measures the number of periods since the last lowest low.

#### Chaikin Money Flow (CMF)
- **CMF**: A measure of the money flow in an asset calculated over a 20-period window. Used to confirm trends and identify buying or selling pressure.

#### Commodity Channel Index (CCI)
- **CCI**: Calculated over a 20-period window, it measures the variation of price from its average. Used to identify overbought and oversold conditions.

#### Ichimoku Cloud
- **Ichimoku Tenkan**: The conversion line calculated as the average of the highest high and lowest low over the past 9 periods.
- **Ichimoku Kijun**: The base line calculated as the average of the highest high and lowest low over the past 26 periods.
- **Ichimoku Senkou Span A**: The average of the Tenkan and Kijun lines, projected 26 periods ahead.
- **Ichimoku Senkou Span B**: The average of the highest high and lowest low over the past 52 periods, projected 26 periods ahead.

### Description of Indicators and Features

1. **SMA 50** and **SMA 200**: Used to identify medium- and long-term trends. The crossover between SMA 50 and SMA 200 is often used as a trading signal.
2. **EMA**: Responds more quickly to price changes compared to SMA, useful for detecting recent trends.
3. **MACD and MACD Signal**: Used together to identify trend reversals and trend strength.
4. **RSI**: A momentum indicator that helps identify overbought and oversold conditions.
5. **Bollinger Bands**: Used to measure volatility. The bands expand during periods of high volatility and contract during periods of low volatility.
6. **ATR**: A measure of market volatility, useful for setting stop loss levels.
7. **Stochastic Oscillator**: Helps identify potential trend reversals in overbought and oversold conditions.
8. **Aroon Indicator**: Measures the strength of the current trend and the likelihood of a possible reversal.
9. **CMF**: Indicates buying and selling pressure, useful for confirming trends.
10. **CCI**: Indicates the strength of a trend and possible overbought or oversold conditions.
11. **Ichimoku Cloud**: Provides information on support, resistance, momentum, and future trends.

These technical indicators are calculated using the `ta` library and added to historical data to improve the accuracy of the prediction model.

### Hyperparameters Optimization [optuna]

The hyperparameters can be optimized using the `hyperparameter_optimization.py` script.
The script uses the Optuna library to optimize the hyperparameters of the LSTM model.

```bash
python src/hyperparameter_optimization.py --config config.json
```

Optional arguments include `--rebuild-features` to recalculate the features.

### Training

To train the model, run:

```bash
python src/train.py  --config config.json # optional --rebuild-features
```

Optional arguments include `--rebuild-features` to recalculate the features.

### Prediction

To predict stock prices, run:

```bash
python src/predict.py --config config.json
```

#### Models

`models/{symbol}_model.pth` contains the trained model.

## Contributors

- `Gianluca Mazza` - [LinkedIn](https://www.linkedin.com/in/gianlucamazza/)
- `Matteo Garbelli` - [LinkedIn](https://www.linkedin.com/in/matteo-garbelli-1a0bb3b1/)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
