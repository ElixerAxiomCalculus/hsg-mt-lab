import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

COLUMN_ALIASES = {
    "date": "date",
    "datetime": "date",
    "timestamp": "date",
    "symbol": "symbol",
    "ticker": "symbol",
    "security": "security",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "adjclose": "adj_close",
    "adjustedclose": "adj_close",
    "volume": "volume",
    "prevclose": "prev_close",
    "previousclose": "prev_close",
    "change": "change",
    "percentchange": "percent_change",
    "changepercent": "percent_change",
    "weightage": "weightage",
}


@dataclass(frozen=True)
class DatasetInspection:
    filename: str
    sha256: str
    file_size: int
    row_count: int
    columns: list[str]
    normalized_columns: list[str]
    detected_type: str
    min_date: str | None
    max_date: str | None
    unique_symbols: int | None
    null_counts: dict[str, int]
    duplicate_rows: int
    invalid_price_rows: int
    invalid_dates: int
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_column_name(value: str) -> str:
    key = re.sub(r"[^a-z0-9]", "", value.strip().lower())
    if "%" in value and key == "change":
        return "percent_change"
    return COLUMN_ALIASES.get(key, re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_"))


def normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.rename(
        columns={column: normalize_column_name(str(column)) for column in frame}
    )
    if len(set(normalized.columns)) != len(normalized.columns):
        raise ValueError("Duplicate columns after normalization")
    if "date" in normalized:
        normalized["date"] = pd.to_datetime(
            normalized["date"], errors="coerce", utc=True, format="mixed"
        )
    if "symbol" in normalized:
        normalized["symbol"] = normalized["symbol"].astype("string").str.strip().str.upper()
    return normalized


def classify_dataset(columns: set[str]) -> str:
    prices = {"date", "open", "high", "low", "close"}
    if prices | {"symbol", "volume"} <= columns:
        return "STOCK_OHLCV"
    if prices <= columns and ({"prev_close", "percent_change"} & columns):
        return "INDIA_VIX"
    if prices <= columns:
        return "INDEX_OHLC"
    if {"symbol", "security", "weightage"} <= columns:
        return "UNIVERSE_WEIGHTS"
    return "UNKNOWN"


def inspect_csv(path: Path) -> DatasetInspection:
    raw = pd.read_csv(path, low_memory=False)
    original_columns = [str(column) for column in raw.columns]
    frame = normalize_frame(raw)
    warnings: list[str] = []
    dataset_type = classify_dataset(set(frame.columns))
    if dataset_type == "UNKNOWN":
        warnings.append("Schema could not be classified")
    invalid_dates = int(frame["date"].isna().sum()) if "date" in frame else len(frame)
    if invalid_dates:
        warnings.append(f"{invalid_dates} rows have invalid or missing dates")
    price_columns = [column for column in ("open", "high", "low", "close") if column in frame]
    invalid_price_rows = 0
    if price_columns:
        numeric_prices = frame[price_columns].apply(pd.to_numeric, errors="coerce")
        invalid_price_rows = int(
            (numeric_prices.isna().any(axis=1) | (numeric_prices <= 0).any(axis=1)).sum()
        )
    if invalid_price_rows:
        warnings.append(f"{invalid_price_rows} rows have missing, zero, or negative prices")
    duplicate_rows = int(frame.duplicated().sum())
    if duplicate_rows:
        warnings.append(f"{duplicate_rows} exact duplicate rows detected")
    dated = frame["date"].dropna() if "date" in frame else pd.Series(dtype="datetime64[ns, UTC]")
    return DatasetInspection(
        filename=path.name,
        sha256=fingerprint(path),
        file_size=path.stat().st_size,
        row_count=len(frame),
        columns=original_columns,
        normalized_columns=[str(column) for column in frame.columns],
        detected_type=dataset_type,
        min_date=dated.min().isoformat() if not dated.empty else None,
        max_date=dated.max().isoformat() if not dated.empty else None,
        unique_symbols=int(frame["symbol"].nunique()) if "symbol" in frame else None,
        null_counts={str(key): int(value) for key, value in frame.isna().sum().items()},
        duplicate_rows=duplicate_rows,
        invalid_price_rows=invalid_price_rows,
        invalid_dates=invalid_dates,
        warnings=warnings,
    )


def inspect_directory(directory: Path) -> list[DatasetInspection]:
    return [inspect_csv(path) for path in sorted(directory.glob("*.csv"))]


def load_validated_csv(path: Path) -> pd.DataFrame:
    frame = normalize_frame(pd.read_csv(path, low_memory=False))
    if "date" not in frame:
        raise ValueError(f"{path.name}: date column is required")
    frame = frame.dropna(subset=["date"])
    subset = [column for column in ("date", "symbol") if column in frame]
    frame = frame.drop_duplicates(subset=subset, keep="last")
    return frame.sort_values(subset, kind="stable").reset_index(drop=True)
