from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ChronologicalSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def chronological_split(
    frame: pd.DataFrame, train_fraction: float = 0.70, validation_fraction: float = 0.15
) -> ChronologicalSplit:
    if not 0 < train_fraction < 1 or not 0 <= validation_fraction < 1:
        raise ValueError("Invalid split fractions")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Train and validation fractions leave no test set")
    ordered = frame.sort_values("date", kind="stable").reset_index(drop=True)
    train_end = int(len(ordered) * train_fraction)
    validation_end = int(len(ordered) * (train_fraction + validation_fraction))
    return ChronologicalSplit(
        ordered.iloc[:train_end].copy(),
        ordered.iloc[train_end:validation_end].copy(),
        ordered.iloc[validation_end:].copy(),
    )


def walk_forward_windows(
    frame: pd.DataFrame, minimum_train_rows: int, evaluation_rows: int
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    ordered = frame.sort_values("date", kind="stable").reset_index(drop=True)
    windows: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    cursor = minimum_train_rows
    while cursor < len(ordered):
        windows.append(
            (ordered.iloc[:cursor].copy(), ordered.iloc[cursor : cursor + evaluation_rows].copy())
        )
        cursor += evaluation_rows
    return windows
