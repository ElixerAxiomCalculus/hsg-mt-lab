from __future__ import annotations

import hashlib
import io
import json
import math
import random
import threading
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ALGORITHM_NAME = "HSG-MT"
ALGORITHM_VERSION = "1.0.0"
REGIMES = ("bearish", "neutral", "bullish", "reversal", "breakdown")
REGIME_TO_INDEX = {name: index for index, name in enumerate(REGIMES)}
EPSILON = 1e-8


@dataclass
class AlgorithmState:
    model: HSGMTModel | None = None
    online_optimizer: torch.optim.Optimizer | None = None
    feature_names: list[str] = field(default_factory=list)
    scaler_mean: np.ndarray | None = None
    scaler_std: np.ndarray | None = None
    configuration: dict[str, Any] = field(default_factory=dict)
    model_version: str = ""
    base_model_version: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    training_history: list[dict[str, float]] = field(default_factory=list)
    histories: dict[str, deque[tuple[str, list[float]]]] = field(default_factory=dict)
    trajectory_state: dict[str, dict[str, Any]] = field(default_factory=dict)
    pending: dict[str, dict[str, Any]] = field(default_factory=dict)
    global_step: int = 0
    online_updates: int = 0
    device: str = "cpu"
    regime_threshold: float = 0.0
    latest_checkpoint_metadata: dict[str, Any] = field(default_factory=dict)


_STATE = AlgorithmState()
_LOCK = threading.RLock()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Non-finite value cannot be serialized")
        return value
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, torch.Tensor):
        return _json_safe(value.detach().cpu().numpy())
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Datetime must be timezone-aware")
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def _normalize_name(value: str) -> str:
    return "".join(character for character in str(value).strip().lower() if character.isalnum())


def _parse_timestamp(value: Any) -> datetime:
    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        raise ValueError("timestamp must be an ISO-8601 string or datetime")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _set_determinism(seed: int, strict: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if strict:
        torch.use_deterministic_algorithms(True, warn_only=True)


def _choose_device(configuration: Mapping[str, Any]) -> str:
    requested = str(configuration.get("device", "cpu")).lower()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested not in {"cpu", "cuda"}:
        raise ValueError("device must be 'cpu', 'cuda', or 'auto'")
    return requested


def _icosahedron_vertices() -> torch.Tensor:
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    vertices = []
    for first in (-1.0, 1.0):
        for second in (-phi, phi):
            vertices.append((0.0, first, second))
            vertices.append((first, second, 0.0))
            vertices.append((second, 0.0, first))
    array = np.unique(np.asarray(vertices, dtype=np.float32), axis=0)
    array = array / np.linalg.norm(array, axis=1, keepdims=True)
    if array.shape != (12, 3):
        raise RuntimeError("Failed to construct icosahedron")
    return torch.from_numpy(array)


def _icosahedron_graph(vertices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    array = vertices.detach().cpu().numpy().astype(np.float64)
    distances = np.linalg.norm(array[:, None, :] - array[None, :, :], axis=-1)
    positive = distances[distances > EPSILON]
    edge_length = float(np.min(positive))
    adjacency = ((distances <= edge_length * 1.01) & (distances > EPSILON)).astype(np.float32)
    degrees = adjacency.sum(axis=1)
    if not np.all(degrees == 5):
        raise RuntimeError("Invalid icosahedral adjacency")
    laplacian = np.diag(degrees) - adjacency
    _, eigenvectors = np.linalg.eigh(laplacian)
    return torch.from_numpy(laplacian.astype(np.float32)), torch.from_numpy(eigenvectors.astype(np.float32))


class HSGMTModel(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        gru_layers: int,
        dropout: float,
        vertex_temperature: float,
        harmonic_components: int,
    ) -> None:
        super().__init__()
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if hidden_dim < 8:
            raise ValueError("hidden_dim must be at least 8")
        if gru_layers < 1:
            raise ValueError("gru_layers must be positive")
        if not 1 <= harmonic_components <= 10:
            raise ValueError("harmonic_components must be between 1 and 10")
        vertices = _icosahedron_vertices()
        laplacian, basis = _icosahedron_graph(vertices)
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.gru_layers = gru_layers
        self.dropout_rate = dropout
        self.vertex_temperature = vertex_temperature
        self.harmonic_components = harmonic_components
        self.register_buffer("vertices", vertices)
        self.register_buffer("graph_laplacian", laplacian)
        self.register_buffer("graph_basis", basis)
        self.encoder = nn.GRU(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=gru_layers,
            dropout=dropout if gru_layers > 1 else 0.0,
            batch_first=True,
        )
        self.to_manifold = nn.Linear(hidden_dim, 3)
        trajectory_dim = 8
        geometric_dim = 3 + 12 + harmonic_components + trajectory_dim
        self.geometry_fusion = nn.Sequential(
            nn.Linear(geometric_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        head_dim = max(16, hidden_dim // 2)
        self.return_head = nn.Sequential(
            nn.Linear(hidden_dim, head_dim),
            nn.GELU(),
            nn.Linear(head_dim, 1),
        )
        self.log_variance_head = nn.Sequential(
            nn.Linear(hidden_dim, head_dim),
            nn.GELU(),
            nn.Linear(head_dim, 1),
        )
        self.regime_head = nn.Sequential(
            nn.Linear(hidden_dim, head_dim),
            nn.GELU(),
            nn.Linear(head_dim, len(REGIMES)),
        )

    def forward(self, sequence: torch.Tensor) -> dict[str, torch.Tensor]:
        encoded, _ = self.encoder(sequence)
        manifold = F.normalize(self.to_manifold(encoded), p=2, dim=-1, eps=EPSILON)
        vertex_logits = self.vertex_temperature * torch.einsum("blc,vc->blv", manifold, self.vertices)
        spectrum = torch.softmax(vertex_logits, dim=-1)
        basis = self.graph_basis[:, 1 : self.harmonic_components + 1]
        harmonics = torch.matmul(spectrum[:, -1, :], basis)
        trajectory = self._trajectory_features(manifold, spectrum)
        geometry = torch.cat((manifold[:, -1, :], spectrum[:, -1, :], harmonics, trajectory), dim=-1)
        geometric_hidden = self.geometry_fusion(geometry)
        fused = self.fusion(torch.cat((encoded[:, -1, :], geometric_hidden), dim=-1))
        expected_return_scaled = self.return_head(fused).squeeze(-1)
        log_variance = self.log_variance_head(fused).squeeze(-1).clamp(-8.0, 6.0)
        regime_logits = self.regime_head(fused)
        return {
            "expected_return_scaled": expected_return_scaled,
            "log_variance": log_variance,
            "regime_logits": regime_logits,
            "manifold": manifold,
            "spectrum": spectrum,
            "harmonics": harmonics,
            "trajectory": trajectory,
        }

    def _trajectory_features(self, manifold: torch.Tensor, spectrum: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = manifold.shape
        if sequence_length > 1:
            dots = torch.sum(manifold[:, 1:, :] * manifold[:, :-1, :], dim=-1).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
            steps = torch.acos(dots)
            last_velocity = steps[:, -1]
            mean_velocity = steps.mean(dim=1)
            path_length = steps.sum(dim=1)
        else:
            zeros = torch.zeros(batch_size, device=manifold.device, dtype=manifold.dtype)
            steps = torch.zeros((batch_size, 0), device=manifold.device, dtype=manifold.dtype)
            last_velocity = zeros
            mean_velocity = zeros
            path_length = zeros
        if sequence_length > 2:
            acceleration = steps[:, -1] - steps[:, -2]
            first_direction = F.normalize(manifold[:, -2, :] - manifold[:, -3, :], dim=-1, eps=EPSILON)
            second_direction = F.normalize(manifold[:, -1, :] - manifold[:, -2, :], dim=-1, eps=EPSILON)
            direction_dot = torch.sum(first_direction * second_direction, dim=-1).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
            curvature = torch.acos(direction_dot)
        else:
            acceleration = torch.zeros(batch_size, device=manifold.device, dtype=manifold.dtype)
            curvature = torch.zeros(batch_size, device=manifold.device, dtype=manifold.dtype)
        last_spectrum = spectrum[:, -1, :]
        entropy = -(last_spectrum * torch.log(last_spectrum.clamp_min(EPSILON))).sum(dim=-1)
        concentration = last_spectrum.max(dim=-1).values
        energy = mean_velocity.square()
        return torch.stack(
            (last_velocity, mean_velocity, acceleration, curvature, path_length, entropy, concentration, energy),
            dim=-1,
        )


def _loss_components(
    model: HSGMTModel,
    outputs: Mapping[str, torch.Tensor],
    target_return_scaled: torch.Tensor,
    target_regime: torch.Tensor,
    class_weights: torch.Tensor | None,
    configuration: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    mean = outputs["expected_return_scaled"]
    log_variance = outputs["log_variance"]
    squared_error = (target_return_scaled - mean).square()
    return_nll = 0.5 * (torch.exp(-log_variance) * squared_error + log_variance)
    return_loss = return_nll.mean()
    regime_loss = F.cross_entropy(outputs["regime_logits"], target_regime, weight=class_weights)
    spectrum = outputs["spectrum"]
    sample_entropy = -(spectrum * torch.log(spectrum.clamp_min(EPSILON))).sum(dim=-1).mean()
    usage = spectrum.mean(dim=(0, 1))
    usage_entropy = -(usage * torch.log(usage.clamp_min(EPSILON))).sum()
    spectrum_diversity_loss = sample_entropy - usage_entropy
    laplacian_energy = torch.einsum("blv,vw,blw->", spectrum, model.graph_laplacian, spectrum) / spectrum.shape[0] / spectrum.shape[1]
    manifold = outputs["manifold"]
    if manifold.shape[1] > 1:
        temporal_smoothness = (1.0 - torch.sum(manifold[:, 1:, :] * manifold[:, :-1, :], dim=-1)).mean()
    else:
        temporal_smoothness = torch.zeros((), device=manifold.device, dtype=manifold.dtype)
    total = (
        return_loss
        + float(configuration.get("regime_loss_weight", 0.7)) * regime_loss
        + float(configuration.get("spectrum_diversity_weight", 0.02)) * spectrum_diversity_loss
        + float(configuration.get("graph_energy_weight", 0.005)) * laplacian_energy
        + float(configuration.get("temporal_smoothness_weight", 0.002)) * temporal_smoothness
    )
    return total, {
        "total": total.detach(),
        "return_nll": return_loss.detach(),
        "regime_ce": regime_loss.detach(),
        "spectrum_diversity": spectrum_diversity_loss.detach(),
        "graph_energy": laplacian_energy.detach(),
        "temporal_smoothness": temporal_smoothness.detach(),
    }


def _frame_from_value(value: Any) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        frame = value.copy()
    elif isinstance(value, Mapping):
        for key in ("data", "rows", "records", "frame"):
            if key in value:
                return _frame_from_value(value[key])
        frame = pd.DataFrame(value)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        frame = pd.DataFrame(value)
    else:
        raise TypeError("dataset must contain pandas DataFrames, mappings, or record sequences")
    if isinstance(frame.index, pd.DatetimeIndex) and not any(_normalize_name(column) in {"date", "datetime", "timestamp", "time"} for column in frame.columns):
        frame = frame.reset_index().rename(columns={frame.index.name or "index": "timestamp"})
    return frame


def _extract_partitions(dataset: Any, configuration: Mapping[str, Any]) -> dict[str, pd.DataFrame]:
    if isinstance(dataset, Mapping) and any(key in dataset for key in ("train", "validation", "val", "test")):
        if "train" not in dataset:
            raise ValueError("dataset partitions must contain 'train'")
        train = _frame_from_value(dataset["train"])
        validation = _frame_from_value(dataset["validation"] if "validation" in dataset else dataset.get("val", []))
        test = _frame_from_value(dataset.get("test", []))
        return {"train": train, "validation": validation, "test": test}
    frame = _frame_from_value(dataset)
    timestamp_column = _resolve_column(frame, configuration.get("timestamp_column"), ("timestamp", "datetime", "date", "time"), required=True)
    parsed = pd.to_datetime(frame[timestamp_column], utc=True, errors="coerce")
    valid = frame.loc[parsed.notna()].copy()
    valid[timestamp_column] = parsed.loc[parsed.notna()]
    if valid.empty:
        raise ValueError("dataset has no valid timestamps")
    unique_times = np.array(sorted(valid[timestamp_column].drop_duplicates().tolist()), dtype=object)
    if len(unique_times) < 10:
        raise ValueError("dataset requires at least 10 distinct timestamps")
    train_fraction = float(configuration.get("train_fraction", 0.70))
    validation_fraction = float(configuration.get("validation_fraction", 0.15))
    if not 0.5 <= train_fraction < 1.0:
        raise ValueError("train_fraction must be within [0.5, 1.0)")
    if not 0.0 <= validation_fraction < 0.5:
        raise ValueError("validation_fraction must be within [0.0, 0.5)")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("train_fraction + validation_fraction must be less than 1")
    train_index = max(1, min(len(unique_times) - 2, int(len(unique_times) * train_fraction)))
    validation_index = max(train_index + 1, min(len(unique_times) - 1, int(len(unique_times) * (train_fraction + validation_fraction))))
    train_end = unique_times[train_index - 1]
    validation_end = unique_times[validation_index - 1]
    train = valid[valid[timestamp_column] <= train_end].copy()
    validation = valid[(valid[timestamp_column] > train_end) & (valid[timestamp_column] <= validation_end)].copy()
    test = valid[valid[timestamp_column] > validation_end].copy()
    return {"train": train, "validation": validation, "test": test}


def _resolve_column(
    frame: pd.DataFrame,
    explicit: Any,
    aliases: Iterable[str],
    required: bool,
) -> str | None:
    if explicit is not None:
        requested = str(explicit)
        if requested in frame.columns:
            return requested
        normalized_requested = _normalize_name(requested)
        for column in frame.columns:
            if _normalize_name(column) == normalized_requested:
                return str(column)
        if required:
            raise ValueError(f"Required column '{requested}' is missing")
        return None
    normalized = {_normalize_name(column): str(column) for column in frame.columns}
    for alias in aliases:
        candidate = normalized.get(_normalize_name(alias))
        if candidate is not None:
            return candidate
    if required:
        raise ValueError(f"Required column matching {tuple(aliases)} is missing")
    return None


def _prepare_frame(frame: pd.DataFrame, configuration: Mapping[str, Any]) -> tuple[pd.DataFrame, dict[str, str | None]]:
    if frame.empty:
        return frame.copy(), {"timestamp": None, "symbol": None, "price": None}
    timestamp_column = _resolve_column(frame, configuration.get("timestamp_column"), ("timestamp", "datetime", "date", "time"), required=True)
    symbol_column = _resolve_column(frame, configuration.get("symbol_column"), ("symbol", "ticker", "security", "instrument"), required=False)
    adjusted = _resolve_column(frame, configuration.get("adjusted_close_column"), ("adj close", "adj_close", "adjusted close", "adjusted_close"), required=False)
    close = _resolve_column(frame, configuration.get("close_column"), ("close", "closing price", "closeprice"), required=adjusted is None)
    price_column = adjusted if bool(configuration.get("prefer_adjusted_close", True)) and adjusted is not None else close or adjusted
    if price_column is None:
        raise ValueError("A close or adjusted-close column is required")
    prepared = frame.copy()
    prepared[timestamp_column] = pd.to_datetime(prepared[timestamp_column], utc=True, errors="coerce")
    prepared = prepared.loc[prepared[timestamp_column].notna()].copy()
    if symbol_column is None:
        prepared["__hsg_symbol__"] = "__MARKET__"
        symbol_column = "__hsg_symbol__"
    prepared[symbol_column] = prepared[symbol_column].astype(str)
    prepared[price_column] = pd.to_numeric(prepared[price_column], errors="coerce")
    prepared = prepared.loc[np.isfinite(prepared[price_column].to_numpy(dtype=np.float64)) & (prepared[price_column] > 0)].copy()
    prepared = prepared.sort_values([symbol_column, timestamp_column], kind="mergesort")
    prepared = prepared.drop_duplicates(subset=[symbol_column, timestamp_column], keep="last")
    return prepared, {"timestamp": timestamp_column, "symbol": symbol_column, "price": price_column}


def _infer_feature_names(frame: pd.DataFrame, schema: Mapping[str, str | None], configuration: Mapping[str, Any]) -> list[str]:
    configured = configuration.get("feature_columns")
    if configured is not None:
        names = []
        normalized_columns = {_normalize_name(column): str(column) for column in frame.columns}
        for value in configured:
            key = _normalize_name(str(value))
            if key not in normalized_columns:
                raise ValueError(f"Configured feature column '{value}' is missing")
            names.append(normalized_columns[key])
        if not names:
            raise ValueError("feature_columns cannot be empty")
        return names
    excluded = {
        schema.get("timestamp"),
        schema.get("symbol"),
        "target",
        "target_return",
        "future_return",
        "regime",
        "label",
    }
    names = []
    for column in frame.columns:
        if column in excluded:
            continue
        numeric = pd.to_numeric(frame[column], errors="coerce")
        finite_fraction = float(np.isfinite(numeric.to_numpy(dtype=np.float64, na_value=np.nan)).mean()) if len(numeric) else 0.0
        if finite_fraction >= float(configuration.get("minimum_numeric_fraction", 0.95)):
            names.append(str(column))
    if not names:
        raise ValueError("No numeric feature columns were detected")
    return names


def _coerce_numeric_features(frame: pd.DataFrame, feature_names: Sequence[str]) -> pd.DataFrame:
    output = frame.copy()
    for column in feature_names:
        output[column] = pd.to_numeric(output[column], errors="coerce")
    matrix = output[list(feature_names)].to_numpy(dtype=np.float64)
    valid = np.isfinite(matrix).all(axis=1)
    return output.loc[valid].copy()


def _fit_scaler(frame: pd.DataFrame, feature_names: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    matrix = frame[list(feature_names)].to_numpy(dtype=np.float64)
    if matrix.size == 0:
        raise ValueError("Training feature matrix is empty")
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("Scaler contains non-finite values")
    return mean.astype(np.float32), std.astype(np.float32)


def _scaled_matrix(frame: pd.DataFrame, feature_names: Sequence[str], mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    matrix = frame[list(feature_names)].to_numpy(dtype=np.float32)
    return ((matrix - mean) / std).astype(np.float32)


def _build_samples(
    frame: pd.DataFrame,
    schema: Mapping[str, str | None],
    feature_names: Sequence[str],
    scaler_mean: np.ndarray,
    scaler_std: np.ndarray,
    sequence_length: int,
    horizon_steps: int,
    momentum_lookback: int,
    max_abs_target_return: float | None,
) -> dict[str, Any]:
    if frame.empty:
        return {"x": np.empty((0, sequence_length, len(feature_names)), dtype=np.float32), "y": np.empty(0, dtype=np.float32), "momentum": np.empty(0, dtype=np.float32), "symbol": [], "timestamp": []}
    symbol_column = schema["symbol"]
    timestamp_column = schema["timestamp"]
    price_column = schema["price"]
    if symbol_column is None or timestamp_column is None or price_column is None:
        raise ValueError("Invalid dataset schema")
    sequences: list[np.ndarray] = []
    targets: list[float] = []
    momenta: list[float] = []
    symbols: list[str] = []
    timestamps: list[str] = []
    for symbol, group in frame.groupby(symbol_column, sort=False):
        group = group.sort_values(timestamp_column, kind="mergesort")
        if len(group) < sequence_length + horizon_steps:
            continue
        scaled = _scaled_matrix(group, feature_names, scaler_mean, scaler_std)
        prices = group[price_column].to_numpy(dtype=np.float64)
        times = pd.to_datetime(group[timestamp_column], utc=True)
        for end in range(sequence_length - 1, len(group) - horizon_steps):
            start = end - sequence_length + 1
            current_price = float(prices[end])
            future_price = float(prices[end + horizon_steps])
            target = future_price / current_price - 1.0
            if not math.isfinite(target):
                continue
            if max_abs_target_return is not None and abs(target) > max_abs_target_return:
                continue
            momentum_index = max(start, end - momentum_lookback)
            momentum = current_price / float(prices[momentum_index]) - 1.0 if prices[momentum_index] > 0 else 0.0
            sequence = scaled[start : end + 1]
            if not np.isfinite(sequence).all():
                continue
            sequences.append(sequence)
            targets.append(float(target))
            momenta.append(float(momentum))
            symbols.append(str(symbol))
            timestamps.append(times.iloc[end].isoformat().replace("+00:00", "Z"))
    if not sequences:
        return {"x": np.empty((0, sequence_length, len(feature_names)), dtype=np.float32), "y": np.empty(0, dtype=np.float32), "momentum": np.empty(0, dtype=np.float32), "symbol": [], "timestamp": []}
    return {
        "x": np.stack(sequences).astype(np.float32),
        "y": np.asarray(targets, dtype=np.float32),
        "momentum": np.asarray(momenta, dtype=np.float32),
        "symbol": symbols,
        "timestamp": timestamps,
    }


def _derive_regime_threshold(targets: np.ndarray, configuration: Mapping[str, Any]) -> float:
    explicit = configuration.get("regime_return_threshold")
    if explicit is not None:
        threshold = float(explicit)
        if threshold <= 0 or not math.isfinite(threshold):
            raise ValueError("regime_return_threshold must be finite and positive")
        return threshold
    absolute = np.abs(targets[np.isfinite(targets)])
    if absolute.size == 0:
        raise ValueError("Cannot derive regime threshold without targets")
    quantile = float(configuration.get("regime_threshold_quantile", 0.35))
    if not 0.05 <= quantile <= 0.95:
        raise ValueError("regime_threshold_quantile must be within [0.05, 0.95]")
    derived = float(np.quantile(absolute, quantile))
    floor = float(configuration.get("regime_threshold_floor", 1e-4))
    return max(derived, floor)


def _regime_labels(targets: np.ndarray, momentum: np.ndarray, threshold: float) -> np.ndarray:
    labels = np.full(len(targets), REGIME_TO_INDEX["neutral"], dtype=np.int64)
    reversal = (targets > threshold) & (momentum < -threshold)
    breakdown = (targets < -threshold) & (momentum > threshold)
    bullish = (targets > threshold) & ~reversal
    bearish = (targets < -threshold) & ~breakdown
    labels[bearish] = REGIME_TO_INDEX["bearish"]
    labels[bullish] = REGIME_TO_INDEX["bullish"]
    labels[reversal] = REGIME_TO_INDEX["reversal"]
    labels[breakdown] = REGIME_TO_INDEX["breakdown"]
    return labels


def _tensor_dataset(samples: Mapping[str, Any], threshold: float, return_scale: float) -> TensorDataset:
    labels = _regime_labels(samples["y"], samples["momentum"], threshold)
    x = torch.from_numpy(samples["x"]).float()
    y = torch.from_numpy(samples["y"] * return_scale).float()
    regime = torch.from_numpy(labels).long()
    return TensorDataset(x, y, regime)


def _class_weights(labels: np.ndarray, device: str) -> torch.Tensor:
    counts = np.bincount(labels, minlength=len(REGIMES)).astype(np.float64)
    total = counts.sum()
    weights = np.ones(len(REGIMES), dtype=np.float32)
    present = counts > 0
    if total > 0 and present.any():
        weights[present] = (total / (len(REGIMES) * counts[present])).astype(np.float32)
        weights = np.clip(weights, 0.25, 4.0)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _evaluate(
    model: HSGMTModel,
    dataset: TensorDataset,
    device: str,
    return_scale: float,
    batch_size: int,
) -> dict[str, float]:
    if len(dataset) == 0:
        return {}
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    predicted_returns: list[np.ndarray] = []
    true_returns: list[np.ndarray] = []
    regime_probabilities: list[np.ndarray] = []
    true_regimes: list[np.ndarray] = []
    nll_values: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for x, y_scaled, regimes in loader:
            x = x.to(device)
            y_scaled = y_scaled.to(device)
            outputs = model(x)
            mean_scaled = outputs["expected_return_scaled"]
            log_variance = outputs["log_variance"]
            nll = 0.5 * (torch.exp(-log_variance) * (y_scaled - mean_scaled).square() + log_variance)
            predicted_returns.append((mean_scaled / return_scale).cpu().numpy())
            true_returns.append((y_scaled / return_scale).cpu().numpy())
            regime_probabilities.append(torch.softmax(outputs["regime_logits"], dim=-1).cpu().numpy())
            true_regimes.append(regimes.numpy())
            nll_values.append(nll.cpu().numpy())
    predicted = np.concatenate(predicted_returns)
    actual = np.concatenate(true_returns)
    probabilities = np.concatenate(regime_probabilities)
    labels = np.concatenate(true_regimes)
    nll = np.concatenate(nll_values)
    errors = predicted - actual
    predicted_labels = probabilities.argmax(axis=1)
    directional_accuracy = float(np.mean(np.sign(predicted) == np.sign(actual)))
    regime_accuracy = float(np.mean(predicted_labels == labels))
    one_hot = np.eye(len(REGIMES), dtype=np.float32)[labels]
    brier = float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))
    confidence = probabilities.max(axis=1)
    correctness = (predicted_labels == labels).astype(np.float32)
    bins = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for lower, upper in zip(bins[:-1], bins[1:]):
        mask = (confidence >= lower) & (confidence < upper if upper < 1.0 else confidence <= upper)
        if mask.any():
            ece += float(mask.mean()) * abs(float(correctness[mask].mean()) - float(confidence[mask].mean()))
    return {
        "mae": float(np.mean(np.abs(errors))),
        "mse": float(np.mean(errors**2)),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "directional_accuracy": directional_accuracy,
        "regime_accuracy": regime_accuracy,
        "multiclass_brier": brier,
        "regime_ece": float(ece),
        "return_nll": float(np.mean(nll)),
    }


def _training_signature(configuration: Mapping[str, Any], feature_names: Sequence[str], samples: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    payload = {
        "algorithm": ALGORITHM_VERSION,
        "configuration": _json_safe(dict(configuration)),
        "feature_names": list(feature_names),
        "sample_count": int(len(samples["y"])),
        "target_mean": float(np.mean(samples["y"])) if len(samples["y"]) else 0.0,
        "target_std": float(np.std(samples["y"])) if len(samples["y"]) else 0.0,
    }
    digest.update(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    if len(samples["x"]):
        stride = max(1, len(samples["x"]) // 1024)
        digest.update(np.ascontiguousarray(samples["x"][::stride]).tobytes())
        digest.update(np.ascontiguousarray(samples["y"][::stride]).tobytes())
    return digest.hexdigest()


def _serialize_state(context: Mapping[str, Any] | None = None) -> tuple[bytes, dict[str, Any]]:
    if _STATE.model is None or _STATE.scaler_mean is None or _STATE.scaler_std is None:
        raise RuntimeError("MODEL_NOT_TRAINED")
    histories = {
        symbol: list(entries)
        for symbol, entries in _STATE.histories.items()
    }
    pending = {}
    for prediction_id, record in _STATE.pending.items():
        pending[prediction_id] = {
            **{key: value for key, value in record.items() if key != "sequence"},
            "sequence": np.asarray(record["sequence"], dtype=np.float32),
        }
    payload = {
        "format_version": 1,
        "algorithm_name": ALGORITHM_NAME,
        "algorithm_version": ALGORITHM_VERSION,
        "model_version": _STATE.model_version,
        "base_model_version": _STATE.base_model_version,
        "model_parameters": {
            "feature_dim": _STATE.model.feature_dim,
            "hidden_dim": _STATE.model.hidden_dim,
            "gru_layers": _STATE.model.gru_layers,
            "dropout": _STATE.model.dropout_rate,
            "vertex_temperature": _STATE.model.vertex_temperature,
            "harmonic_components": _STATE.model.harmonic_components,
        },
        "model_state": _STATE.model.state_dict(),
        "online_optimizer_state": _STATE.online_optimizer.state_dict() if _STATE.online_optimizer is not None else None,
        "feature_names": _STATE.feature_names,
        "scaler_mean": _STATE.scaler_mean,
        "scaler_std": _STATE.scaler_std,
        "configuration": _STATE.configuration,
        "metrics": _STATE.metrics,
        "training_history": _STATE.training_history,
        "histories": histories,
        "trajectory_state": _STATE.trajectory_state,
        "pending": pending,
        "global_step": _STATE.global_step,
        "online_updates": _STATE.online_updates,
        "regime_threshold": _STATE.regime_threshold,
        "checkpoint_context": dict(context or {}),
        "saved_at": _utc_now().isoformat(),
    }
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    data = buffer.getvalue()
    metadata = {
        "algorithm_name": ALGORITHM_NAME,
        "algorithm_version": ALGORITHM_VERSION,
        "model_version": _STATE.model_version,
        "byte_size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "saved_at": _utc_now().isoformat().replace("+00:00", "Z"),
        "global_step": _STATE.global_step,
        "online_updates": _STATE.online_updates,
    }
    return data, metadata


def pretrain(dataset: Any, configuration: Mapping[str, Any] | None = None) -> dict[str, Any]:
    configuration = dict(configuration or {})
    progress_callback = configuration.pop("_progress_callback", None)
    if progress_callback is not None and not callable(progress_callback):
        raise ValueError("_progress_callback must be callable")
    progress_callback = progress_callback if callable(progress_callback) else None
    with _LOCK:
        seed = int(configuration.get("seed", 1729))
        strict_determinism = bool(configuration.get("strict_determinism", True))
        _set_determinism(seed, strict_determinism)
        device = _choose_device(configuration)
        partitions = _extract_partitions(dataset, configuration)
        prepared: dict[str, pd.DataFrame] = {}
        schemas: dict[str, dict[str, str | None]] = {}
        for name, frame in partitions.items():
            if frame.empty:
                prepared[name] = frame.copy()
                schemas[name] = {"timestamp": None, "symbol": None, "price": None}
                continue
            prepared_frame, schema = _prepare_frame(frame, configuration)
            prepared[name] = prepared_frame
            schemas[name] = schema
        if prepared["train"].empty:
            raise ValueError("Training partition is empty after validation")
        feature_names = _infer_feature_names(prepared["train"], schemas["train"], configuration)
        for name in prepared:
            if prepared[name].empty:
                continue
            missing = [feature for feature in feature_names if feature not in prepared[name].columns]
            if missing:
                raise ValueError(f"Partition '{name}' is missing features: {missing}")
            prepared[name] = _coerce_numeric_features(prepared[name], feature_names)
        scaler_mean, scaler_std = _fit_scaler(prepared["train"], feature_names)
        sequence_length = int(configuration.get("sequence_length", 32))
        horizon_steps = int(configuration.get("horizon_steps", 1))
        momentum_lookback = int(configuration.get("momentum_lookback", 5))
        if sequence_length < 4:
            raise ValueError("sequence_length must be at least 4")
        if horizon_steps < 1:
            raise ValueError("horizon_steps must be positive")
        if momentum_lookback < 1:
            raise ValueError("momentum_lookback must be positive")
        max_abs_target_return_value = configuration.get("max_abs_target_return", 0.50)
        max_abs_target_return = None if max_abs_target_return_value is None else float(max_abs_target_return_value)
        samples: dict[str, dict[str, Any]] = {}
        for name in prepared:
            schema = schemas[name]
            if prepared[name].empty:
                samples[name] = {"x": np.empty((0, sequence_length, len(feature_names)), dtype=np.float32), "y": np.empty(0, dtype=np.float32), "momentum": np.empty(0, dtype=np.float32), "symbol": [], "timestamp": []}
                continue
            if schema["timestamp"] is None:
                _, schema = _prepare_frame(prepared[name], configuration)
            samples[name] = _build_samples(
                prepared[name],
                schema,
                feature_names,
                scaler_mean,
                scaler_std,
                sequence_length,
                horizon_steps,
                momentum_lookback,
                max_abs_target_return,
            )
        if len(samples["train"]["y"]) < int(configuration.get("minimum_training_samples", 128)):
            raise ValueError("Insufficient training sequences")
        regime_threshold = _derive_regime_threshold(samples["train"]["y"], configuration)
        return_scale = float(configuration.get("return_scale", 100.0))
        if return_scale <= 0 or not math.isfinite(return_scale):
            raise ValueError("return_scale must be finite and positive")
        train_dataset = _tensor_dataset(samples["train"], regime_threshold, return_scale)
        validation_dataset = _tensor_dataset(samples["validation"], regime_threshold, return_scale)
        test_dataset = _tensor_dataset(samples["test"], regime_threshold, return_scale)
        hidden_dim = int(configuration.get("hidden_dim", 96))
        gru_layers = int(configuration.get("gru_layers", 2))
        dropout = float(configuration.get("dropout", 0.15))
        vertex_temperature = float(configuration.get("vertex_temperature", 8.0))
        harmonic_components = int(configuration.get("harmonic_components", 6))
        model = HSGMTModel(
            feature_dim=len(feature_names),
            hidden_dim=hidden_dim,
            gru_layers=gru_layers,
            dropout=dropout,
            vertex_temperature=vertex_temperature,
            harmonic_components=harmonic_components,
        ).to(device)
        learning_rate = float(configuration.get("learning_rate", 3e-4))
        weight_decay = float(configuration.get("weight_decay", 1e-4))
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        labels = _regime_labels(samples["train"]["y"], samples["train"]["momentum"], regime_threshold)
        class_weights = _class_weights(labels, device)
        batch_size = int(configuration.get("batch_size", 256))
        epochs = int(configuration.get("epochs", 30))
        patience = int(configuration.get("early_stopping_patience", 6))
        gradient_clip = float(configuration.get("gradient_clip_norm", 1.0))
        if batch_size < 1 or epochs < 1 or patience < 1:
            raise ValueError("batch_size, epochs, and patience must be positive")
        generator = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, generator=generator)
        validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        best_state: dict[str, torch.Tensor] | None = None
        best_validation = math.inf
        stale_epochs = 0
        history: list[dict[str, float]] = []
        for epoch in range(epochs):
            model.train()
            train_loss_total = 0.0
            train_count = 0
            for x, y_scaled, regimes in train_loader:
                x = x.to(device)
                y_scaled = y_scaled.to(device)
                regimes = regimes.to(device)
                optimizer.zero_grad(set_to_none=True)
                outputs = model(x)
                loss, _ = _loss_components(model, outputs, y_scaled, regimes, class_weights, configuration)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()
                train_loss_total += float(loss.detach().cpu()) * x.shape[0]
                train_count += x.shape[0]
            train_loss = train_loss_total / max(train_count, 1)
            validation_loss = train_loss
            if len(validation_dataset):
                model.eval()
                validation_total = 0.0
                validation_count = 0
                with torch.no_grad():
                    for x, y_scaled, regimes in validation_loader:
                        x = x.to(device)
                        y_scaled = y_scaled.to(device)
                        regimes = regimes.to(device)
                        outputs = model(x)
                        loss, _ = _loss_components(model, outputs, y_scaled, regimes, class_weights, configuration)
                        validation_total += float(loss.detach().cpu()) * x.shape[0]
                        validation_count += x.shape[0]
                validation_loss = validation_total / max(validation_count, 1)
            history.append({"epoch": float(epoch + 1), "train_loss": float(train_loss), "validation_loss": float(validation_loss)})
            if validation_loss < best_validation - float(configuration.get("early_stopping_min_delta", 1e-5)):
                best_validation = validation_loss
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                stale_epochs = 0
            else:
                stale_epochs += 1
            if progress_callback is not None:
                callback = progress_callback
                callback(
                    {
                        "phase": "TRAINING",
                        "epoch": epoch + 1,
                        "total_epochs": epochs,
                        "train_loss": float(train_loss),
                        "validation_loss": float(validation_loss),
                        "best_validation_loss": float(best_validation),
                        "stale_epochs": stale_epochs,
                        "early_stopping_patience": patience,
                    }
                )
            if stale_epochs >= patience:
                break
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        train_metrics = _evaluate(model, train_dataset, device, return_scale, batch_size)
        validation_metrics = _evaluate(model, validation_dataset, device, return_scale, batch_size)
        test_metrics = _evaluate(model, test_dataset, device, return_scale, batch_size)
        signature = _training_signature(configuration, feature_names, samples["train"])
        timestamp = _utc_now().strftime("%Y%m%dT%H%M%SZ")
        model_version = f"hsg-mt-{ALGORITHM_VERSION}-{timestamp}-{signature[:10]}"
        resolved_configuration = dict(configuration)
        resolved_configuration.update(
            {
                "sequence_length": sequence_length,
                "horizon_steps": horizon_steps,
                "momentum_lookback": momentum_lookback,
                "return_scale": return_scale,
                "hidden_dim": hidden_dim,
                "gru_layers": gru_layers,
                "dropout": dropout,
                "vertex_temperature": vertex_temperature,
                "harmonic_components": harmonic_components,
                "regime_return_threshold": regime_threshold,
                "prediction_horizon": str(configuration.get("prediction_horizon", f"{horizon_steps}-step")),
                "bar_interval": str(configuration.get("bar_interval", "unspecified")),
                "seed": seed,
            }
        )
        _STATE.model = model
        _STATE.feature_names = list(feature_names)
        _STATE.scaler_mean = scaler_mean
        _STATE.scaler_std = scaler_std
        _STATE.configuration = resolved_configuration
        _STATE.model_version = model_version
        _STATE.base_model_version = model_version
        _STATE.training_history = history
        _STATE.histories = {}
        _STATE.trajectory_state = {}
        _STATE.pending = {}
        _STATE.global_step = 0
        _STATE.online_updates = 0
        _STATE.device = device
        _STATE.regime_threshold = regime_threshold
        online_learning_rate = float(configuration.get("online_learning_rate", 1e-5))
        _STATE.online_optimizer = torch.optim.AdamW(model.parameters(), lr=online_learning_rate, weight_decay=weight_decay)
        class_distribution = {regime: int(np.sum(labels == index)) for index, regime in enumerate(REGIMES)}
        _STATE.metrics = {
            "algorithm_name": ALGORITHM_NAME,
            "algorithm_version": ALGORITHM_VERSION,
            "model_version": model_version,
            "training_signature": signature,
            "feature_count": len(feature_names),
            "feature_names": list(feature_names),
            "sequence_length": sequence_length,
            "horizon_steps": horizon_steps,
            "prediction_horizon": resolved_configuration["prediction_horizon"],
            "bar_interval": resolved_configuration["bar_interval"],
            "regime_return_threshold": regime_threshold,
            "train_samples": len(train_dataset),
            "validation_samples": len(validation_dataset),
            "test_samples": len(test_dataset),
            "class_distribution": class_distribution,
            "epochs_completed": len(history),
            "best_validation_loss": float(best_validation if math.isfinite(best_validation) else history[-1]["train_loss"]),
            "train": train_metrics,
            "validation": validation_metrics,
            "test": test_metrics,
            "online": {
                "updates": 0,
                "mean_loss": None,
                "directional_accuracy": None,
            },
        }
        checkpoint, checkpoint_metadata = _serialize_state({"reason": "PRETRAIN_COMPLETE"})
        _STATE.latest_checkpoint_metadata = checkpoint_metadata
        return {
            "model_version": model_version,
            "metrics": _json_safe(_STATE.metrics),
            "checkpoint": checkpoint,
            "metadata": _json_safe(
                {
                    "algorithm_name": ALGORITHM_NAME,
                    "algorithm_version": ALGORITHM_VERSION,
                    "feature_names": feature_names,
                    "training_signature": signature,
                    "checkpoint_metadata": checkpoint_metadata,
                }
            ),
        }


def save_checkpoint(context: Mapping[str, Any] | None = None) -> dict[str, Any]:
    with _LOCK:
        data, metadata = _serialize_state(context)
        _STATE.latest_checkpoint_metadata = metadata
        return {"data": data, "metadata": _json_safe(metadata)}


def load_checkpoint(checkpoint: bytes, metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if not isinstance(checkpoint, (bytes, bytearray)) or not checkpoint:
        raise ValueError("checkpoint must be non-empty bytes")
    with _LOCK:
        if metadata and metadata.get("sha256"):
            actual = hashlib.sha256(bytes(checkpoint)).hexdigest()
            if actual != str(metadata["sha256"]):
                raise ValueError("Checkpoint SHA-256 mismatch")
        buffer = io.BytesIO(bytes(checkpoint))
        payload = torch.load(buffer, map_location="cpu", weights_only=False)
        if payload.get("algorithm_name") != ALGORITHM_NAME:
            raise ValueError("Checkpoint belongs to a different algorithm")
        if int(payload.get("format_version", 0)) != 1:
            raise ValueError("Unsupported checkpoint format")
        configuration = dict(payload["configuration"])
        device = _choose_device(configuration)
        parameters = payload["model_parameters"]
        model = HSGMTModel(
            feature_dim=int(parameters["feature_dim"]),
            hidden_dim=int(parameters["hidden_dim"]),
            gru_layers=int(parameters["gru_layers"]),
            dropout=float(parameters["dropout"]),
            vertex_temperature=float(parameters["vertex_temperature"]),
            harmonic_components=int(parameters["harmonic_components"]),
        ).to(device)
        model.load_state_dict(payload["model_state"])
        model.eval()
        weight_decay = float(configuration.get("weight_decay", 1e-4))
        online_learning_rate = float(configuration.get("online_learning_rate", 1e-5))
        optimizer = torch.optim.AdamW(model.parameters(), lr=online_learning_rate, weight_decay=weight_decay)
        if payload.get("online_optimizer_state") is not None:
            optimizer.load_state_dict(payload["online_optimizer_state"])
            for optimizer_state in optimizer.state.values():
                for key, value in optimizer_state.items():
                    if isinstance(value, torch.Tensor):
                        optimizer_state[key] = value.to(device)
        sequence_length = int(configuration.get("sequence_length", 32))
        histories: dict[str, deque[tuple[str, list[float]]]] = {}
        for symbol, entries in payload.get("histories", {}).items():
            histories[str(symbol)] = deque(((str(timestamp), [float(value) for value in vector]) for timestamp, vector in entries), maxlen=sequence_length)
        pending = {}
        for prediction_id, record in payload.get("pending", {}).items():
            restored = dict(record)
            restored["sequence"] = np.asarray(record["sequence"], dtype=np.float32)
            pending[str(prediction_id)] = restored
        _STATE.model = model
        _STATE.online_optimizer = optimizer
        _STATE.feature_names = [str(value) for value in payload["feature_names"]]
        _STATE.scaler_mean = np.asarray(payload["scaler_mean"], dtype=np.float32)
        _STATE.scaler_std = np.asarray(payload["scaler_std"], dtype=np.float32)
        _STATE.configuration = configuration
        _STATE.model_version = str(payload["model_version"])
        _STATE.base_model_version = str(payload.get("base_model_version", payload["model_version"]))
        _STATE.metrics = dict(payload.get("metrics", {}))
        _STATE.training_history = list(payload.get("training_history", []))
        _STATE.histories = histories
        _STATE.trajectory_state = dict(payload.get("trajectory_state", {}))
        _STATE.pending = pending
        _STATE.global_step = int(payload.get("global_step", 0))
        _STATE.online_updates = int(payload.get("online_updates", 0))
        _STATE.device = device
        _STATE.regime_threshold = float(payload.get("regime_threshold", configuration.get("regime_return_threshold", 1e-4)))
        _STATE.latest_checkpoint_metadata = dict(metadata or {})
        return _json_safe(
            {
                "model_version": _STATE.model_version,
                "algorithm_name": ALGORITHM_NAME,
                "algorithm_version": ALGORITHM_VERSION,
                "feature_count": len(_STATE.feature_names),
                "global_step": _STATE.global_step,
                "online_updates": _STATE.online_updates,
                "loaded": True,
            }
        )


def _feature_vector(item: Any) -> np.ndarray:
    if not _STATE.feature_names:
        raise RuntimeError("MODEL_NOT_TRAINED")
    if isinstance(item, Mapping):
        normalized = {_normalize_name(str(key)): value for key, value in item.items()}
        values = []
        for feature in _STATE.feature_names:
            key = _normalize_name(feature)
            if key not in normalized:
                raise ValueError(f"Missing required feature '{feature}'")
            value = float(normalized[key])
            if not math.isfinite(value):
                raise ValueError(f"Feature '{feature}' is non-finite")
            values.append(value)
        return np.asarray(values, dtype=np.float32)
    if isinstance(item, np.ndarray):
        vector = item.astype(np.float32, copy=False).reshape(-1)
    elif isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
        vector = np.asarray(item, dtype=np.float32).reshape(-1)
    else:
        raise TypeError("features must be a mapping or numeric sequence")
    if len(vector) != len(_STATE.feature_names):
        raise ValueError(f"Expected {len(_STATE.feature_names)} features, received {len(vector)}")
    if not np.isfinite(vector).all():
        raise ValueError("Feature vector contains non-finite values")
    return vector


def _sequence_from_features(symbol: str, timestamp: datetime, features: Any, context: Mapping[str, Any]) -> tuple[np.ndarray, int]:
    sequence_length = int(_STATE.configuration.get("sequence_length", 32))
    full_sequence = None
    if isinstance(features, Mapping):
        if "history" in features:
            full_sequence = features["history"]
        elif "sequence" in features:
            full_sequence = features["sequence"]
    if full_sequence is None:
        if "feature_history" in context:
            full_sequence = context["feature_history"]
        elif "sequence" in context:
            full_sequence = context["sequence"]
    if full_sequence is not None:
        vectors = [_feature_vector(item) for item in full_sequence]
        if isinstance(features, Mapping) and all(_normalize_name(name) in {_normalize_name(str(key)) for key in features.keys()} for name in _STATE.feature_names):
            current = _feature_vector(features)
            if not vectors or not np.allclose(vectors[-1], current, rtol=0.0, atol=0.0):
                vectors.append(current)
        if not vectors:
            raise ValueError("Feature history is empty")
        raw = np.stack(vectors[-sequence_length:]).astype(np.float32)
        history = deque(maxlen=sequence_length)
        base_time = timestamp.isoformat().replace("+00:00", "Z")
        for index, vector in enumerate(raw):
            history.append((f"{base_time}:{index}", vector.tolist()))
        _STATE.histories[symbol] = history
        return raw, len(raw)
    current = _feature_vector(features)
    history = _STATE.histories.setdefault(symbol, deque(maxlen=sequence_length))
    timestamp_iso = timestamp.isoformat().replace("+00:00", "Z")
    if history and history[-1][0] == timestamp_iso:
        history[-1] = (timestamp_iso, current.tolist())
    else:
        history.append((timestamp_iso, current.tolist()))
    raw = np.stack([np.asarray(vector, dtype=np.float32) for _, vector in history])
    return raw, len(raw)


def _current_momentum(raw_sequence: np.ndarray) -> float:
    names = {_normalize_name(name): index for index, name in enumerate(_STATE.feature_names)}
    preferred = (
        "return5",
        "return_5",
        "5periodreturn",
        "momentum5",
        "momentum",
        "return1",
        "return_1",
    )
    for candidate in preferred:
        index = names.get(_normalize_name(candidate))
        if index is not None:
            value = float(raw_sequence[-1, index])
            if math.isfinite(value):
                return value
    close_index = names.get("close")
    if close_index is None:
        close_index = names.get("adjclose")
    if close_index is not None and len(raw_sequence) > 1:
        lookback = min(int(_STATE.configuration.get("momentum_lookback", 5)), len(raw_sequence) - 1)
        previous = float(raw_sequence[-1 - lookback, close_index])
        current = float(raw_sequence[-1, close_index])
        if previous > 0 and math.isfinite(previous) and math.isfinite(current):
            return current / previous - 1.0
    return 0.0


def _trajectory_metrics(manifold: np.ndarray, spectrum: np.ndarray) -> dict[str, float | int]:
    length = len(manifold)
    if length > 1:
        dots = np.sum(manifold[1:] * manifold[:-1], axis=1)
        steps = np.arccos(np.clip(dots, -1.0 + 1e-9, 1.0 - 1e-9))
        velocity = float(steps[-1])
        mean_velocity = float(np.mean(steps))
        path_length = float(np.sum(steps))
        energy = float(np.mean(steps**2))
    else:
        steps = np.empty(0, dtype=np.float64)
        velocity = 0.0
        mean_velocity = 0.0
        path_length = 0.0
        energy = 0.0
    if length > 2:
        acceleration = float(steps[-1] - steps[-2])
        first = manifold[-2] - manifold[-3]
        second = manifold[-1] - manifold[-2]
        first_norm = np.linalg.norm(first)
        second_norm = np.linalg.norm(second)
        if first_norm > EPSILON and second_norm > EPSILON:
            curvature = float(np.arccos(np.clip(np.dot(first / first_norm, second / second_norm), -1.0, 1.0)))
        else:
            curvature = 0.0
    else:
        acceleration = 0.0
        curvature = 0.0
    last_spectrum = spectrum[-1]
    entropy = float(-np.sum(last_spectrum * np.log(np.clip(last_spectrum, EPSILON, None))))
    concentration = float(np.max(last_spectrum))
    dominant_vertex = int(np.argmax(last_spectrum))
    recent = manifold[-min(length, 8) :]
    if len(recent) >= 3:
        covariance = np.cov(recent.T)
        eigenvalues = np.linalg.eigvalsh(covariance)
        positive = np.clip(eigenvalues, 1e-10, None)
        condition_number = float(np.max(positive) / np.min(positive))
    else:
        condition_number = 1.0
    centroid = recent.mean(axis=0)
    centroid_norm = np.linalg.norm(centroid)
    if centroid_norm > EPSILON:
        centroid = centroid / centroid_norm
        distance_from_centroid = float(np.arccos(np.clip(np.dot(manifold[-1], centroid), -1.0, 1.0)))
    else:
        distance_from_centroid = 0.0
    return {
        "velocity": velocity,
        "mean_velocity": mean_velocity,
        "acceleration": acceleration,
        "curvature": curvature,
        "path_length": path_length,
        "trajectory_energy": energy,
        "spectral_entropy": entropy,
        "spectral_concentration": concentration,
        "dominant_vertex": dominant_vertex,
        "local_condition_number": condition_number,
        "distance_from_recent_centroid": distance_from_centroid,
        "samples_available": int(length),
    }


def _geometric_state(model: HSGMTModel, manifold: np.ndarray, spectrum: np.ndarray, harmonics: np.ndarray) -> dict[str, Any]:
    last = manifold[-1]
    last_spectrum = spectrum[-1]
    dots = model.vertices.detach().cpu().numpy() @ last
    dominant = int(np.argmax(last_spectrum))
    return {
        "manifold": "icosahedral_spherical",
        "coordinates": [float(value) for value in last],
        "dominant_vertex": dominant,
        "nearest_vertex_geodesic_distance": float(np.arccos(np.clip(np.max(dots), -1.0, 1.0))),
        "spectrum": {f"vertex_{index}": float(value) for index, value in enumerate(last_spectrum)},
        "harmonics": {f"h{index + 1}": float(value) for index, value in enumerate(harmonics)},
    }


def _warmup_prediction(symbol: str, timestamp: datetime, available: int) -> dict[str, Any]:
    required = int(_STATE.configuration.get("sequence_length", 32))
    probabilities = {regime: 0.0 for regime in REGIMES}
    probabilities["neutral"] = 1.0
    return {
        "symbol": symbol,
        "timestamp": timestamp.isoformat().replace("+00:00", "Z"),
        "action": "HOLD",
        "confidence": 0.0,
        "expected_return": 0.0,
        "prediction_horizon": str(_STATE.configuration.get("prediction_horizon", f"{_STATE.configuration.get('horizon_steps', 1)}-step")),
        "regime": "neutral",
        "regime_probabilities": probabilities,
        "geometric_state": {},
        "trajectory_metrics": {"samples_available": int(available), "samples_required": int(required)},
        "model_version": _STATE.model_version,
        "metadata": {
            "algorithm_name": ALGORITHM_NAME,
            "algorithm_version": ALGORITHM_VERSION,
            "status": "INSUFFICIENT_SEQUENCE_HISTORY",
        },
    }


def predict(symbol: str, timestamp: Any, features: Any, context: Mapping[str, Any] | None = None) -> dict[str, Any]:
    context = dict(context or {})
    with _LOCK:
        if _STATE.model is None or _STATE.scaler_mean is None or _STATE.scaler_std is None:
            raise RuntimeError("MODEL_NOT_TRAINED")
        symbol = str(symbol).strip()
        if not symbol:
            raise ValueError("symbol cannot be empty")
        parsed_timestamp = _parse_timestamp(timestamp)
        raw_sequence, available = _sequence_from_features(symbol, parsed_timestamp, features, context)
        required = int(_STATE.configuration.get("sequence_length", 32))
        allow_partial = bool(_STATE.configuration.get("allow_partial_sequence", False))
        minimum_partial = int(_STATE.configuration.get("minimum_partial_sequence", max(4, required // 2)))
        if available < required and not (allow_partial and available >= minimum_partial):
            return _warmup_prediction(symbol, parsed_timestamp, available)
        raw_sequence = raw_sequence[-required:]
        scaled = ((raw_sequence - _STATE.scaler_mean) / _STATE.scaler_std).astype(np.float32)
        if not np.isfinite(scaled).all():
            raise ValueError("Scaled feature sequence contains non-finite values")
        tensor = torch.from_numpy(scaled).unsqueeze(0).to(_STATE.device)
        _STATE.model.eval()
        with torch.no_grad():
            outputs = _STATE.model(tensor)
        return_scale = float(_STATE.configuration.get("return_scale", 100.0))
        expected_return = float(outputs["expected_return_scaled"].item() / return_scale)
        return_std = float(math.sqrt(math.exp(float(outputs["log_variance"].item()))) / return_scale)
        probabilities_array = torch.softmax(outputs["regime_logits"], dim=-1).squeeze(0).cpu().numpy()
        probabilities = {regime: float(probabilities_array[index]) for index, regime in enumerate(REGIMES)}
        regime_index = int(np.argmax(probabilities_array))
        regime = REGIMES[regime_index]
        positive_probability = probabilities["bullish"] + probabilities["reversal"]
        negative_probability = probabilities["bearish"] + probabilities["breakdown"]
        uncertainty_reference = float(_STATE.configuration.get("uncertainty_reference", max(_STATE.regime_threshold * 5.0, 0.005)))
        uncertainty_factor = math.exp(-max(return_std, 0.0) / max(uncertainty_reference, 1e-8))
        buy_threshold = float(_STATE.configuration.get("buy_return_threshold", _STATE.regime_threshold))
        sell_threshold = float(_STATE.configuration.get("sell_return_threshold", -_STATE.regime_threshold))
        min_confidence = float(_STATE.configuration.get("min_action_confidence", 0.60))
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_action_confidence must be within [0, 1]")
        if buy_threshold <= 0 or sell_threshold >= 0:
            raise ValueError("buy_return_threshold must be positive and sell_return_threshold must be negative")
        action = "HOLD"
        signal_probability = float(np.max(probabilities_array))
        if expected_return >= buy_threshold and positive_probability >= min_confidence:
            action = "BUY"
            signal_probability = positive_probability
        elif expected_return <= sell_threshold and negative_probability >= min_confidence:
            action = "SELL"
            signal_probability = negative_probability
        confidence = float(np.clip(signal_probability * uncertainty_factor, 0.0, 1.0))
        if action != "HOLD" and confidence < min_confidence:
            action = "HOLD"
        manifold = outputs["manifold"].squeeze(0).cpu().numpy().astype(np.float64)
        spectrum = outputs["spectrum"].squeeze(0).cpu().numpy().astype(np.float64)
        harmonics = outputs["harmonics"].squeeze(0).cpu().numpy().astype(np.float64)
        trajectory_metrics = _trajectory_metrics(manifold, spectrum)
        geometric_state = _geometric_state(_STATE.model, manifold, spectrum, harmonics)
        algorithm_prediction_id = str(uuid.uuid4())
        momentum = _current_momentum(raw_sequence)
        max_pending = int(_STATE.configuration.get("max_pending_predictions", 5000))
        if len(_STATE.pending) >= max_pending:
            oldest = next(iter(_STATE.pending))
            del _STATE.pending[oldest]
        _STATE.pending[algorithm_prediction_id] = {
            "symbol": symbol,
            "timestamp": parsed_timestamp.isoformat().replace("+00:00", "Z"),
            "sequence": scaled.copy(),
            "momentum": float(momentum),
            "predicted_return": expected_return,
            "predicted_action": action,
            "predicted_regime": regime,
            "model_version": _STATE.model_version,
        }
        state = {
            "symbol": symbol,
            "timestamp": parsed_timestamp.isoformat().replace("+00:00", "Z"),
            "model_version": _STATE.model_version,
            "geometric_state": geometric_state,
            "trajectory_metrics": trajectory_metrics,
            "regime": regime,
            "regime_probabilities": probabilities,
            "expected_return": expected_return,
            "return_std": return_std,
            "confidence": confidence,
            "action": action,
        }
        _STATE.trajectory_state[symbol] = _json_safe(state)
        return _json_safe(
            {
                "symbol": symbol,
                "timestamp": parsed_timestamp,
                "action": action,
                "confidence": confidence,
                "expected_return": expected_return,
                "prediction_horizon": str(_STATE.configuration.get("prediction_horizon", f"{_STATE.configuration.get('horizon_steps', 1)}-step")),
                "regime": regime,
                "regime_probabilities": probabilities,
                "geometric_state": geometric_state,
                "trajectory_metrics": trajectory_metrics,
                "model_version": _STATE.model_version,
                "metadata": {
                    "algorithm_name": ALGORITHM_NAME,
                    "algorithm_version": ALGORITHM_VERSION,
                    "algorithm_prediction_id": algorithm_prediction_id,
                    "return_std": return_std,
                    "positive_regime_probability": positive_probability,
                    "negative_regime_probability": negative_probability,
                    "uncertainty_factor": uncertainty_factor,
                    "sequence_samples": int(len(raw_sequence)),
                    "bar_interval": str(_STATE.configuration.get("bar_interval", "unspecified")),
                },
            }
        )


def _target_return_from_observation(observation: Mapping[str, Any]) -> float:
    for key in ("target_return", "observed_return", "realized_return", "future_return"):
        if key in observation:
            value = float(observation[key])
            if not math.isfinite(value):
                raise ValueError(f"{key} must be finite")
            return value
    reference_keys = ("reference_price", "prediction_price", "start_price", "entry_price")
    observed_keys = ("observed_price", "future_price", "current_price", "close")
    reference = next((float(observation[key]) for key in reference_keys if key in observation), None)
    observed = next((float(observation[key]) for key in observed_keys if key in observation), None)
    if reference is None or observed is None:
        raise ValueError("observation must include a matured return or reference and observed prices")
    if not math.isfinite(reference) or not math.isfinite(observed) or reference <= 0 or observed <= 0:
        raise ValueError("reference and observed prices must be finite and positive")
    return observed / reference - 1.0


def _pending_record(observation: Mapping[str, Any], context: Mapping[str, Any]) -> tuple[str | None, dict[str, Any]]:
    candidates = []
    for source in (observation, context):
        for key in ("algorithm_prediction_id", "prediction_id"):
            if key in source:
                candidates.append(str(source[key]))
        metadata = source.get("metadata")
        if isinstance(metadata, Mapping) and "algorithm_prediction_id" in metadata:
            candidates.append(str(metadata["algorithm_prediction_id"]))
    for prediction_id in candidates:
        if prediction_id in _STATE.pending:
            return prediction_id, _STATE.pending[prediction_id]
    symbol = str(observation.get("symbol", context.get("symbol", ""))).strip()
    if symbol:
        for prediction_id, record in _STATE.pending.items():
            if record.get("symbol") == symbol:
                return prediction_id, record
    sequence_source = context.get("prediction_sequence") or context.get("feature_history")
    if sequence_source is not None:
        vectors = [_feature_vector(item) for item in sequence_source]
        required = int(_STATE.configuration.get("sequence_length", 32))
        raw = np.stack(vectors[-required:]).astype(np.float32)
        scaled = ((raw - _STATE.scaler_mean) / _STATE.scaler_std).astype(np.float32)
        return None, {
            "symbol": symbol or "__UNKNOWN__",
            "timestamp": str(context.get("prediction_timestamp", "")),
            "sequence": scaled,
            "momentum": _current_momentum(raw),
            "predicted_return": float(context.get("predicted_return", 0.0)),
            "predicted_action": str(context.get("predicted_action", "HOLD")),
            "predicted_regime": str(context.get("predicted_regime", "neutral")),
            "model_version": str(context.get("model_version", _STATE.model_version)),
        }
    raise KeyError("No matching matured prediction context was found")


def online_update(observation: Mapping[str, Any], context: Mapping[str, Any] | None = None) -> dict[str, Any]:
    context = dict(context or {})
    with _LOCK:
        if _STATE.model is None or _STATE.online_optimizer is None:
            raise RuntimeError("MODEL_NOT_TRAINED")
        if not isinstance(observation, Mapping):
            raise TypeError("observation must be a mapping")
        target_return = _target_return_from_observation(observation)
        prediction_id, record = _pending_record(observation, context)
        sequence = np.asarray(record["sequence"], dtype=np.float32)
        if sequence.ndim != 2 or sequence.shape[1] != len(_STATE.feature_names):
            raise ValueError("Stored prediction sequence has invalid shape")
        momentum = float(record.get("momentum", 0.0))
        regime = int(_regime_labels(np.asarray([target_return], dtype=np.float32), np.asarray([momentum], dtype=np.float32), _STATE.regime_threshold)[0])
        return_scale = float(_STATE.configuration.get("return_scale", 100.0))
        x = torch.from_numpy(sequence).unsqueeze(0).to(_STATE.device)
        y = torch.tensor([target_return * return_scale], dtype=torch.float32, device=_STATE.device)
        regime_tensor = torch.tensor([regime], dtype=torch.long, device=_STATE.device)
        steps = int(_STATE.configuration.get("online_steps", 1))
        if steps < 1:
            raise ValueError("online_steps must be positive")
        class_weights = None
        _STATE.model.train()
        losses = []
        for _ in range(steps):
            _STATE.online_optimizer.zero_grad(set_to_none=True)
            outputs = _STATE.model(x)
            loss, _ = _loss_components(_STATE.model, outputs, y, regime_tensor, class_weights, _STATE.configuration)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite online learning loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(_STATE.model.parameters(), float(_STATE.configuration.get("gradient_clip_norm", 1.0)))
            _STATE.online_optimizer.step()
            losses.append(float(loss.detach().cpu()))
            _STATE.global_step += 1
        _STATE.model.eval()
        _STATE.online_updates += 1
        _STATE.model_version = f"{_STATE.base_model_version}-u{_STATE.online_updates}"
        online_metrics = dict(_STATE.metrics.get("online", {}))
        previous_updates = int(online_metrics.get("updates") or 0)
        previous_mean_loss = online_metrics.get("mean_loss")
        current_loss = float(np.mean(losses))
        if previous_mean_loss is None or previous_updates == 0:
            mean_loss = current_loss
        else:
            mean_loss = (float(previous_mean_loss) * previous_updates + current_loss) / (previous_updates + 1)
        predicted_return = float(record.get("predicted_return", 0.0))
        directional_correct = float(np.sign(predicted_return) == np.sign(target_return))
        previous_directional = online_metrics.get("directional_accuracy")
        if previous_directional is None or previous_updates == 0:
            directional_accuracy = directional_correct
        else:
            directional_accuracy = (float(previous_directional) * previous_updates + directional_correct) / (previous_updates + 1)
        _STATE.metrics["model_version"] = _STATE.model_version
        _STATE.metrics["online"] = {
            "updates": previous_updates + 1,
            "mean_loss": mean_loss,
            "directional_accuracy": directional_accuracy,
            "last_loss": current_loss,
            "last_target_return": target_return,
            "last_target_regime": REGIMES[regime],
        }
        if prediction_id is not None:
            _STATE.pending.pop(prediction_id, None)
        return _json_safe(
            {
                "model_version": _STATE.model_version,
                "previous_model_version": record.get("model_version", _STATE.base_model_version),
                "algorithm_prediction_id": prediction_id,
                "target_return": target_return,
                "target_regime": REGIMES[regime],
                "loss": current_loss,
                "global_step": _STATE.global_step,
                "online_updates": _STATE.online_updates,
                "metrics": _STATE.metrics["online"],
            }
        )


def get_metrics(context: Mapping[str, Any] | None = None) -> dict[str, Any]:
    with _LOCK:
        if _STATE.model is None:
            raise RuntimeError("MODEL_NOT_TRAINED")
        result = dict(_STATE.metrics)
        result["model_version"] = _STATE.model_version
        result["global_step"] = _STATE.global_step
        result["pending_predictions"] = len(_STATE.pending)
        result["tracked_symbols"] = len(_STATE.histories)
        result["latest_checkpoint"] = _STATE.latest_checkpoint_metadata
        if context and bool(context.get("include_training_history", False)):
            result["training_history"] = list(_STATE.training_history)
        return _json_safe(result)


def get_trajectory_state(context: Mapping[str, Any] | None = None) -> dict[str, Any]:
    with _LOCK:
        if _STATE.model is None:
            raise RuntimeError("MODEL_NOT_TRAINED")
        context = dict(context or {})
        symbol = context.get("symbol")
        if symbol is not None:
            value = _STATE.trajectory_state.get(str(symbol))
            return _json_safe(value or {})
        return _json_safe(
            {
                "model_version": _STATE.model_version,
                "symbols": dict(_STATE.trajectory_state),
            }
        )
