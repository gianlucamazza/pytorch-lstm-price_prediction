from typing import List, Tuple
import pandas as pd
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import (
    SMAIndicator,
    EMAIndicator,
    MACD,
    ADXIndicator,
    AroonIndicator,
    CCIIndicator,
    IchimokuIndicator,
)
from ta.volatility import BollingerBands
from ta.volume import OnBalanceVolumeIndicator, ChaikinMoneyFlowIndicator

from src.logger import setup_logger

logger = setup_logger("feature_engineering_logger", "logs/feature_engineering.log")


def calculate_common_indicators(
    historical_data: pd.DataFrame, windows: dict
) -> pd.DataFrame:
    """
    Calculate common technical indicators for all asset types.

    Parameters
    ----------
    historical_data : pd.DataFrame
        Historical data containing OHLCV values.
    windows : dict
        Dictionary containing window sizes for technical indicators.

    Returns
    -------
    pd.DataFrame
        Historical data with calculated technical indicators.
    """
    logger.info("Calculating common indicators")

    historical_data["SMA_50"] = SMAIndicator(
        historical_data["Close"], window=50
    ).sma_indicator()
    historical_data["SMA_200"] = SMAIndicator(
        historical_data["Close"], window=200
    ).sma_indicator()
    historical_data["EMA"] = EMAIndicator(
        historical_data["Close"], window=windows.get("EMA", 20)
    ).ema_indicator()

    macd = MACD(historical_data["Close"])
    historical_data["MACD"] = macd.macd()
    historical_data["MACD_Signal"] = macd.macd_signal()

    historical_data["RSI"] = RSIIndicator(
        historical_data["Close"], window=windows.get("RSI", 14)
    ).rsi()

    bollinger = BollingerBands(
        historical_data["Close"], window=windows.get("Bollinger", 20)
    )
    historical_data["Bollinger_High"] = bollinger.bollinger_hband()
    historical_data["Bollinger_Low"] = bollinger.bollinger_lband()

    historical_data["ADX"] = ADXIndicator(
        high=historical_data["High"],
        low=historical_data["Low"],
        close=historical_data["Close"],
        window=windows.get("ADX", 14),
    ).adx()

    historical_data = historical_data.dropna()
    logger.info("Dropped NA values from historical data")

    return historical_data


def calculate_specific_indicators(
    historical_data: pd.DataFrame, windows: dict, asset_type: str
) -> pd.DataFrame:
    """
    Calculate specific technical indicators based on the asset type.

    Parameters
    ----------
    historical_data : pd.DataFrame
        Historical data containing OHLCV values.
    windows : dict
        Dictionary containing window sizes for technical indicators.
    asset_type : str
        Type of asset for which the indicators need to be calculated.

    Returns
    -------
    pd.DataFrame
        Historical data with calculated technical indicators.

    Raises
    ------
    ValueError
        If the asset type is not supported.
    """
    logger.info(f"Calculating specific indicators for {asset_type}")

    if asset_type == "stocks":
        historical_data["OBV"] = OnBalanceVolumeIndicator(
            close=historical_data["Close"], volume=historical_data["Volume"]
        ).on_balance_volume()
        historical_data["VWAP"] = (
            historical_data["Close"] * historical_data["Volume"]
        ).cumsum() / historical_data["Volume"].cumsum()
    elif asset_type == "forex":
        historical_data["Stochastic"] = StochasticOscillator(
            historical_data["High"],
            historical_data["Low"],
            historical_data["Close"],
            window=windows.get("Stochastic", 14),
        ).stoch()
        historical_data["Aroon_Up"] = AroonIndicator(
            high=historical_data["High"],
            low=historical_data["Low"],
            window=windows.get("Aroon", 25),
        ).aroon_up()
        historical_data["Aroon_Down"] = AroonIndicator(
            high=historical_data["High"],
            low=historical_data["Low"],
            window=windows.get("Aroon", 25),
        ).aroon_down()
    elif asset_type == "commodity":
        historical_data["CCI"] = CCIIndicator(
            high=historical_data["High"],
            low=historical_data["Low"],
            close=historical_data["Close"],
            window=windows.get("CCI", 20),
        ).cci()
        historical_data["CMF"] = ChaikinMoneyFlowIndicator(
            high=historical_data["High"],
            low=historical_data["Low"],
            close=historical_data["Close"],
            volume=historical_data["Volume"],
            window=windows.get("CMF", 20),
        ).chaikin_money_flow()
    elif asset_type == "etf":
        historical_data["CMF"] = ChaikinMoneyFlowIndicator(
            high=historical_data["High"],
            low=historical_data["Low"],
            close=historical_data["Close"],
            volume=historical_data["Volume"],
            window=windows.get("CMF", 20),
        ).chaikin_money_flow()
    else:
        logger.error(f"Unsupported asset type: {asset_type}")
        raise ValueError(f"Unsupported asset type: {asset_type}")

    return historical_data


def calculate_ichimoku(historical_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Ichimoku indicators for all asset types.

    Parameters
    ----------
    historical_data : pd.DataFrame
        Historical data containing OHLCV values.

    Returns
    -------
    pd.DataFrame
        Historical data with calculated Ichimoku indicators.
    """
    logger.info("Calculating Ichimoku indicators")

    ichimoku = IchimokuIndicator(
        historical_data["High"],
        historical_data["Low"],
        window1=9,
        window2=26,
        window3=52,
    )
    historical_data["Ichimoku_Tenkan"] = ichimoku.ichimoku_conversion_line()
    historical_data["Ichimoku_Kijun"] = ichimoku.ichimoku_base_line()
    historical_data["Ichimoku_Senkou_Span_A"] = ichimoku.ichimoku_a()
    historical_data["Ichimoku_Senkou_Span_B"] = ichimoku.ichimoku_b()

    return historical_data


def calculate_technical_indicators(
    historical_data: pd.DataFrame, windows: dict, asset_type: str, frequency: str
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Calculate technical indicators based on asset type and add them to the historical data.

    Parameters
    ----------
    historical_data : pd.DataFrame
        Historical data containing OHLCV values.
    windows : dict
        Dictionary containing window sizes for technical indicators.
    asset_type : str
        Type of asset for which the indicators need to be calculated.
    frequency : str
        Frequency of the data.

    Returns
    -------
    pd.DataFrame
        Historical data with calculated technical indicators.
    """
    logger.info(
        f"Starting calculation of technical indicators for {asset_type}")
    historical_data.index = pd.to_datetime(historical_data.index)
    historical_data = historical_data.asfreq(frequency)

    historical_data = calculate_common_indicators(historical_data, windows)
    historical_data = calculate_specific_indicators(
        historical_data, windows, asset_type
    )
    historical_data = calculate_ichimoku(historical_data)

    historical_data = historical_data.dropna()
    logger.info("Dropped NA values from historical data")

    features = list(historical_data.columns)
    return historical_data, features
