from datetime import UTC, datetime

import pandas as pd

from app.services.feature_engineering import FeatureConfig, build_features
from app.services.scanner import rank_bearish_candidates


def price_frame(length: int = 40) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=length, tz="UTC"),
            "open": range(100, 100 + length),
            "high": range(102, 102 + length),
            "low": range(98, 98 + length),
            "close": range(101, 101 + length),
            "volume": range(1000, 1000 + length),
        }
    )


def test_features_do_not_change_when_future_is_appended() -> None:
    frame = price_frame()
    first = build_features(frame.iloc[:30], FeatureConfig()).iloc[-1]
    extended = pd.concat(
        [
            frame,
            pd.DataFrame(
                [
                    {
                        "date": pd.Timestamp("2027-01-01", tz="UTC"),
                        "open": 1,
                        "high": 9000,
                        "low": 1,
                        "close": 9000,
                        "volume": 999999,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    second = build_features(extended, FeatureConfig()).iloc[29]
    assert first["sma_medium"] == second["sma_medium"]
    assert first["macd"] == second["macd"]


def test_scanner_ranking_is_deterministic() -> None:
    rows = []
    for index, symbol in enumerate(["A.NS", "B.NS", "C.NS"]):
        rows.append(
            {
                "symbol": symbol,
                "open": 100,
                "close": 100 - index * 5,
                "return_5": -index * 0.01,
                "return_20": -index * 0.02,
                "rsi": 55 - index * 10,
                "ema_short": 100,
                "ema_medium": 101,
                "macd_histogram": -index,
                "volume_ratio": 1 + index,
            }
        )
    result = rank_bearish_candidates(pd.DataFrame(rows), datetime.now(UTC))
    assert result[0]["symbol"] == "C.NS"
    assert [item["rank"] for item in result] == [1, 2, 3]
