from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd


@dataclass(frozen=True)
class ScannerConfig:
    limit: int = 5
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "intraday_return": 1.0,
            "return_5": 1.0,
            "return_20": 1.0,
            "rsi": 0.8,
            "price_vs_ema_short": 0.8,
            "price_vs_ema_medium": 0.8,
            "macd_histogram": 0.8,
            "negative_volume": 0.6,
        }
    )


def _zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if not std or pd.isna(std):
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


def rank_bearish_candidates(
    snapshot: pd.DataFrame, at: datetime, config: ScannerConfig | None = None
) -> list[dict[str, object]]:
    cfg = config or ScannerConfig()
    frame = snapshot.copy()
    frame["intraday_return"] = frame["close"] / frame["open"] - 1
    components = {
        "intraday_return": -_zscore(frame["intraday_return"]),
        "return_5": -_zscore(frame["return_5"]),
        "return_20": -_zscore(frame["return_20"]),
        "rsi": -_zscore(frame["rsi"]),
        "price_vs_ema_short": -_zscore(frame["close"] / frame["ema_short"] - 1),
        "price_vs_ema_medium": -_zscore(frame["close"] / frame["ema_medium"] - 1),
        "macd_histogram": -_zscore(frame["macd_histogram"]),
        "negative_volume": _zscore(frame["volume_ratio"]) * (frame["intraday_return"] < 0),
    }
    for name, values in components.items():
        frame[f"score_{name}"] = values * cfg.weights.get(name, 0.0)
    score_columns = [f"score_{name}" for name in components]
    frame["total_score"] = frame[score_columns].sum(axis=1)
    ranked = frame.sort_values(["total_score", "symbol"], ascending=[False, True]).head(cfg.limit)
    result: list[dict[str, object]] = []
    for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
        result.append(
            {
                "rank": rank,
                "symbol": str(row["symbol"]),
                "total_score": float(row["total_score"]),
                "components": {name: float(row[f"score_{name}"]) for name in components},
                "universe_snapshot_timestamp": at,
            }
        )
    return result
