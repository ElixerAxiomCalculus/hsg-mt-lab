# HSG-MT algorithm contract

The application discovers `algo.py` dynamically and never imports it outside `app/services/hsg_adapter.py`. Until a valid implementation is supplied, the adapter reports `NOT_INSTALLED`; offline training and experiments stop with `ALGORITHM_NOT_INSTALLED`, and no signal is fabricated.

The module may implement these synchronous or asynchronous callables. The adapter accepts keyword arguments, awaits results where necessary, and validates every boundary value.

## Required capabilities

### `pretrain(dataset, configuration) -> dict`

Returns metadata with a non-empty `model_version`. It may also return `metrics` and checkpoint bytes under `checkpoint`.

### `load_checkpoint(checkpoint: bytes, metadata: dict) -> dict | None`

Restores model state. If checkpoint support is intentionally absent, omit the callable.

### `save_checkpoint(context: dict) -> bytes | dict`

Returns bytes, or `{ "data": bytes, "metadata": {...} }`.

### `predict(symbol, timestamp, features, context) -> dict`

Required output:

```json
{
  "symbol": "RELIANCE.NS",
  "timestamp": "2026-09-09T04:10:00Z",
  "action": "BUY",
  "confidence": 0.78,
  "expected_return": 0.012,
  "prediction_horizon": "15m",
  "regime": "reversal",
  "regime_probabilities": {"reversal": 0.78},
  "geometric_state": {},
  "trajectory_metrics": {},
  "model_version": "hsg-mt-1",
  "metadata": {}
}
```

`action` must be exactly `BUY`, `HOLD`, or `SELL`; confidence must be finite and within `[0, 1]`; timestamps must be timezone-aware; numeric metrics must be finite. Missing fields are never inferred. Invalid output becomes a structured no-trade error.

### `online_update(observation, context) -> dict`

Called only after the stored target horizon has matured and the corresponding market observation exists. Omit to report `ONLINE_LEARNING_UNSUPPORTED`.

### `get_metrics(context) -> dict`

Returns arbitrary JSON-safe scalar or series research metrics. Unreturned metrics are displayed as `NOT_PROVIDED_BY_ALGORITHM`.

### `get_trajectory_state(context) -> dict`

Returns JSON-safe geometric/trajectory state for research inspection.

## Reproducibility

The application records the SHA-256 of `algo.py`, model/checkpoint identifiers, dataset fingerprints, feature/scanner configuration, and all prediction inputs and outputs. Algorithm code must not access future observations supplied outside the prediction timestamp.
