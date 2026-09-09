from pathlib import Path

import pandas as pd

from app.research.dataset_loader import fingerprint, inspect_csv, normalize_frame


def test_column_normalization() -> None:
    result = normalize_frame(pd.DataFrame({"Date ": ["2026-01-01"], "Adj Close": [10]}))
    assert list(result.columns) == ["date", "adj_close"]


def test_fingerprint_is_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    path.write_text("Date,Close\n2026-01-01,10\n", encoding="utf-8")
    assert fingerprint(path) == fingerprint(path)


def test_inspection_reports_invalid_price(tmp_path: Path) -> None:
    path = tmp_path / "stocks.csv"
    path.write_text(
        "Date,Symbol,Open,High,Low,Close,Volume\n2026-01-01,X,10,11,9,0,4\n",
        encoding="utf-8",
    )
    result = inspect_csv(path)
    assert result.detected_type == "STOCK_OHLCV"
    assert result.invalid_price_rows == 1
