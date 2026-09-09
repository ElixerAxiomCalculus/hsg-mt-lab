# HSG-MT Lab

HSG-MT Lab is a production-structured research platform for testing whether geometric market-state trajectories learned by **HSG-MT (HyperSpectrum Geometry for Market Trajectories)** contain useful predictive information. It runs isolated NSE or BSE experiments today, with a capability-gated NASDAQ path for checkpoints trained on U.S. data.

> **PAPER TRADING / RESEARCH ONLY.** The application has no brokerage integration and cannot place real orders. Yahoo/yfinance data may be delayed, incomplete, or unavailable and is not exchange-grade. Simulated results do not establish future profitability.

## Current scientific boundary

[`backend/algo.py`](backend/algo.py) is the scientific implementation boundary. No mock prediction or production fallback exists. If the file is absent or fails its contract, experiments stop safely without producing BUY/HOLD/SELL output.

The only integration point is `backend/app/services/hsg_adapter.py`. Before adding HSG-MT, read [`backend/ALGO_CONTRACT.md`](backend/ALGO_CONTRACT.md). Implement the documented functions only in `backend/algo.py`; do not import it elsewhere. The adapter dynamically validates capabilities, timestamps, actions, confidence values, finite numbers, and checkpoint/metric responses.

## Architecture

```text
React/Vite static site
        │ REST + dashboard WebSocket
        ▼
FastAPI web service ───────────── MongoDB Atlas / GridFS
                                      ▲
                                      │ leases, state, events, artifacts
Render background worker ─────────────┘
        │
        ├── MarketDataAgent → yfinance stream/fallback boundary
        ├── ScannerAgent → deterministic bearish ranking
        ├── HSGAgent → adapter → backend/algo.py
        ├── RiskAgent → long-only/cash-only controls
        ├── ExecutionAgent → first eligible post-decision quote
        ├── PortfolioAgent → decimal accounting
        ├── LearningAgent → matured-label updates only
        ├── EvaluationAgent → financial/model metrics
        └── ReportAgent → PDF and auditable ZIP archive
```

The browser is only a client. Experiment ownership, heartbeats, market observations, decisions, positions, and event history live in MongoDB. Closing the browser does not stop a worker. FastAPI `BackgroundTasks` are deliberately not used for the autonomous loop.

MongoDB collections are created/indexed for users, experiments, leases, portfolios, positions, orders, trades, predictions, signals, scanner snapshots, bars, quotes, model/training runs and metrics, online updates, audit events, data-quality events, datasets, holidays, the trading universe, reports, and system state. Large checkpoints, PDFs, and archives belong in GridFS.

## Repository layout

```text
.
├── backend/
│   ├── algo.py                 # intentionally empty
│   ├── ALGO_CONTRACT.md
│   ├── app/
│   │   ├── api/v1/             # versioned HTTP and WebSocket surface
│   │   ├── core/               # settings, JWT/passwords, JSON logging
│   │   ├── db/                 # native async PyMongo client and indexes
│   │   ├── repositories/       # experiments, leases, events
│   │   ├── research/           # datasets, chronological splits, training
│   │   ├── schemas/            # typed API/research models
│   │   ├── services/           # HSG adapter, scanner, risk, execution, reports
│   │   └── worker/             # recovery-aware autonomous process
│   └── tests/
├── frontend/                   # React 19, TypeScript, Vite, Query, Recharts
├── config/                     # editable universe and exchange calendar
├── data/                       # manually supplied training CSVs only
├── render.yaml
└── .env.example
```

## Local setup

Prerequisites: Python 3.12+, Node.js 20+, and a MongoDB Atlas database (or a local MongoDB for development).

```powershell
Copy-Item .env.example .env
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
Set-Location frontend
npm install
Set-Location ..
```

Generate an admin password hash without storing the plain password:

```powershell
python -c "from app.core.security import hash_password; import getpass; print(hash_password(getpass.getpass()))"
```

Run that command from `backend`, then place its output in `ADMIN_PASSWORD_HASH`. Use a random `JWT_SECRET` of at least 32 characters. Never place MongoDB credentials, passwords, or JWT secrets in frontend environment variables.

### Environment variables

| Variable | Purpose |
|---|---|
| `MONGODB_URI` | MongoDB Atlas connection string; backend/worker only |
| `MONGODB_DATABASE` | Database name |
| `JWT_SECRET` | HS256 signing secret |
| `ADMIN_USERNAME` | Single-user login name |
| `ADMIN_PASSWORD_HASH` | bcrypt hash, never a plain password |
| `FRONTEND_ORIGIN` | Exact allowed production origin; no wildcard |
| `APP_ENV` | `development`, `test`, or `production` |
| `LOG_LEVEL` | Structured-log threshold |
| `MARKET_TIMEZONE` | Defaults to `Asia/Kolkata` |
| `QUOTE_STALE_SECONDS` | Maximum tradable quote age; default 60 |
| `SCAN_INTERVAL_SECONDS` | Worker scan cadence; default 300 |
| `TRANSACTION_COST_BPS` | Generic research cost model |
| `SLIPPAGE_BPS` | Generic simulated slippage |
| `MAX_POSITIONS` | Default simultaneous positions |
| `MAX_POSITION_PCT` | Default NAV fraction per position |
| `MODEL_CONFIDENCE_THRESHOLD` | Default minimum HSG confidence |
| `WORKER_LEASE_SECONDS` | MongoDB ownership-lease duration |
| `RAW_TICK_PERSISTENCE` | Off by default to control storage growth |
| `VITE_API_URL` | Public API base URL used only by the frontend |

## Dataset setup

Put CSVs manually in the repository-level `data/` directory. The application does **not** download historical training data, call Kaggle, or scrape NSE. Exact filenames do not matter; classification uses normalized schema and content.

Recommended inputs:

1. A broad historical NSE stock-level OHLCV dataset.
2. NIFTY 50 historical index OHLC data.
3. India VIX historical OHLC/change data.

Common capitalization, spaces, punctuation, `Adj Close`, `Prev. Close`, and `% Change` are normalized. Inspection reports the SHA-256, size, rows, columns, inferred type, date range, symbol count, nulls, duplicates, invalid dates/prices, and warnings. Bad records are reported rather than silently discarded.

Use `GET /api/v1/datasets` or the **Datasets** screen to inspect all files. Training creates stable chronological train/validation/test partitions. Rolling features only use rows at or before their own timestamp; scalers, if the future algorithm requires them, must be fitted on the training period only.

## Running locally

Pretrain once from the repository root. This prints every epoch and publishes the active checkpoint to MongoDB Atlas/GridFS:

```powershell
Set-Location backend
python -m app.research.train_cli
```

Optional controls include `--epochs`, `--batch-size`, `--sequence-length`, `--patience`, `--device`, explicit CSV filenames, and `--config path\to\settings.json`.

For future multi-market models, declare the markets represented by the supplied data, for example `--markets NSE BSE NASDAQ`. The New Experiment form keeps NASDAQ disabled until the active checkpoint explicitly includes NASDAQ; older checkpoints are treated as NSE/BSE-only.

Training metadata records which markets the checkpoint supports. Existing checkpoints default to `NSE BSE`. When NASDAQ data is added later, publish a compatible checkpoint with `--markets NSE BSE NASDAQ` (or `--markets NASDAQ` for a NASDAQ-only model). Until then, NASDAQ remains visible but disabled in New Experiment, and the API independently rejects NASDAQ runs.

After a checkpoint is active, open three terminals from the repository root after activating the virtual environment.

```powershell
Set-Location backend
uvicorn app.main:app --reload --port 8000
```

```powershell
Set-Location backend
python -m app.worker.main
```

```powershell
Set-Location frontend
npm run dev
```

The API documentation is at `http://localhost:8000/docs`; the web application defaults to `http://localhost:5173`. The worker requires MongoDB. The API still exposes liveness, algorithm status, market status, and local dataset inspection if MongoDB is unavailable; readiness correctly reports degraded.

## Offline pretraining

Pretraining is intentionally not exposed in the web application. Run it as a controlled terminal job with `python -m app.research.train_cli`. The command discovers stock OHLCV CSVs in `data/`, performs the chronological split, prints train and validation loss for every epoch, stores the checkpoint in GridFS, records its dataset/code fingerprints, and marks it as the active model.

The autonomous worker loads the active checkpoint before prediction and reloads a newly activated checkpoint on a later cycle. Starting an experiment without an active checkpoint fails with `MODEL_CHECKPOINT_NOT_FOUND`; the platform never falls back to an untrained model. Online updates still occur only after labels mature from real later observations.

Prediction actions must be exactly `BUY`, `HOLD`, or `SELL`. Invalid or exceptional responses result in a structured error and no trade. Online updates are submitted only after the stored target horizon has matured and a real observation exists. If the function is absent, the platform reports `ONLINE_LEARNING_UNSUPPORTED`.

## Trading and research validity

The simulator is long-only, cash-only, and whole-share. It prevents negative cash, negative holdings, overselling, margin, leverage, derivatives, and shorting. BUY sizing uses `floor(available allocation / execution price)`. A zero quantity is rejected as `INSUFFICIENT_CAPITAL_FOR_ONE_SHARE`.

Decisions have a timestamp and may only fill from the first valid quote observed strictly afterward. Configured slippage and generic transaction costs are recorded separately. Scanner components and weights, raw HSG output, risk override, order state, quote, fill, portfolio state, algorithm hash, model/checkpoint version, dataset fingerprints, and correlation IDs remain attributable and reproducible.

Market calendars exclude pre-open, evenings, weekends, and configured holidays. NSE/BSE use `config/market_holidays.json`; NASDAQ uses America/New_York regular hours and `config/nasdaq_market_holidays.json`, including configured early closes. Quotes failing quality/staleness checks cannot be traded.

## Tests and static checks

Backend tests use local fixtures and mocked boundaries; they never depend on live Yahoo availability.

```powershell
Set-Location backend
ruff check app tests
mypy app
pytest
```

```powershell
Set-Location frontend
npm run lint
npm run typecheck
npm test
npm run build
```

## Render deployment

`render.yaml` declares three independent services:

- `hsg-mt-api`: FastAPI web service with liveness health check.
- `hsg-mt-worker`: continuously running background worker.
- `hsg-mt-frontend`: Vite static site with SPA rewrites.

Create a MongoDB Atlas cluster, allow Render’s outbound access according to your Atlas network policy, create a least-privilege database user, and configure the variables above in the `hsg-mt-production` environment group. Set `VITE_API_URL` on the static site to the HTTPS API origin and `FRONTEND_ORIGIN` on the API to the exact static-site origin.

An unattended experiment requires **always-on worker compute**. Do not rely on free-service sleeping behavior. The API service and the worker must not be collapsed into one HTTP process.

On SIGTERM, the worker stops new cycles, records a shutdown event/heartbeat, releases owned leases, and closes MongoDB. On restart it queries active states, atomically reacquires expired leases, restores ownership, and records `WORKER_RECOVERED`. MongoDB uniqueness on order idempotency keys prevents the same model decision from executing twice.

## Reports and archives

Completed experiments generate a server-side PDF labeled as paper trading and a ZIP research archive. The archive includes report, immutable configuration, dataset/model manifests, trades, orders, predictions, scanner snapshots, equity, financial/model metrics, quality events, JSONL agent events, and a SHA-256 `manifest.json`. Missing observations are represented explicitly; the report generator never invents model metrics or market outcomes.

The PDF separates realized and unrealized P&L and documents simulated execution, provider limitations, transaction costs, short-run uncertainty, and the absence of any profitability claim.
