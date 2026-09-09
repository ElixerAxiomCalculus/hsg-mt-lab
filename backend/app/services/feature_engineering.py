from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureConfig:
    short_window: int = 5
    medium_window: int = 20
    volatility_window: int = 20
    rsi_window: int = 14
    atr_window: int = 14


def build_features(frame: pd.DataFrame, config: FeatureConfig | None = None) -> pd.DataFrame:
    cfg = config or FeatureConfig()
    data = frame.sort_values("date", kind="stable").copy()
    close = data["close"].astype(float)
    high = data["high"].astype(float)
    low = data["low"].astype(float)
    volume = data.get("volume", pd.Series(np.nan, index=data.index)).astype(float)
    previous_close = close.shift(1)

    data["log_return"] = np.log(close / previous_close)
    data["return_1"] = close.pct_change(1, fill_method=None)
    data["return_5"] = close.pct_change(cfg.short_window, fill_method=None)
    data["return_20"] = close.pct_change(cfg.medium_window, fill_method=None)
    data["realized_volatility"] = data["log_return"].rolling(cfg.volatility_window).std()
    data["rolling_std"] = close.rolling(cfg.volatility_window).std()
    data["high_low_range"] = (high - low) / previous_close
    true_range = pd.concat(
        [(high - low), (high - previous_close).abs(), (low - previous_close).abs()], axis=1
    ).max(axis=1)
    data["atr"] = true_range.rolling(cfg.atr_window).mean()
    data["sma_short"] = close.rolling(cfg.short_window).mean()
    data["sma_medium"] = close.rolling(cfg.medium_window).mean()
    data["ema_short"] = close.ewm(span=cfg.short_window, adjust=False).mean()
    data["ema_medium"] = close.ewm(span=cfg.medium_window, adjust=False).mean()
    data["price_sma_ratio"] = close / data["sma_medium"]
    data["ema_gap"] = data["ema_short"] / data["ema_medium"] - 1
    data["ma_slope"] = data["sma_short"].diff()
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(cfg.rsi_window).mean()
    loss = -delta.clip(upper=0).rolling(cfg.rsi_window).mean()
    rs = gain / loss.replace(0, np.nan)
    data["rsi"] = 100 - (100 / (1 + rs))
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    data["macd"] = ema12 - ema26
    data["macd_signal"] = data["macd"].ewm(span=9, adjust=False).mean()
    data["macd_histogram"] = data["macd"] - data["macd_signal"]
    data["rate_of_change"] = close.pct_change(cfg.medium_window, fill_method=None)
    data["average_volume"] = volume.rolling(cfg.medium_window).mean()
    data["volume_ratio"] = volume / data["average_volume"]
    volume_std = volume.rolling(cfg.medium_window).std()
    data["volume_zscore"] = (volume - data["average_volume"]) / volume_std.replace(0, np.nan)
    return data
