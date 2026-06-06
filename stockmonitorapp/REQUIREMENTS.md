# REQUIREMENTS DOCUMENT

**Project:** ML-Powered Stock Prediction & Monitoring System
**Version:** 1.1
**Author:** Vibhav Inna Kedege
**Last Updated:** May 17, 2026
**Status:** Active Development

---

## Table of Contents

1. [Overview](#1-overview)
   - 1.1 [Purpose & Scope](#11-purpose--scope)
   - 1.2 [Goals & Non-Goals](#12-goals--non-goals)
   - 1.3 [Architecture](#13-architecture)
2. [Requirements](#2-requirements)
   - 2.1 [Functional](#21-functional-requirements)
   - 2.2 [Non-Functional](#22-non-functional-requirements)
   - 2.3 [Technical Stack](#23-technical-stack)
3. [User Stories](#3-user-stories)
4. [Success Criteria](#4-success-criteria)
5. [Delivery](#5-delivery)
   - 5.1 [Milestones](#51-milestones)
   - 5.2 [Risks & Mitigations](#52-risks--mitigations)
6. [Appendix](#6-appendix)

---

## 1. Overview

### 1.1 Purpose & Scope

Extend an existing Streamlit stock dashboard into a production-grade MLOps platform that predicts S&P500 stock prices and monitors model health over time.

**MLOps lifecycle covered:**

1. Data ingestion & validation
2. Feature engineering
3. Model training & experiment tracking
4. Model deployment & serving
5. Drift detection & monitoring
6. Automated retraining triggers

**Target audience:**

| Persona | Description |
|---|---|
| **End User** | Seeks stock price predictions and trend analysis via the Streamlit UI |
| **ML Engineer** | Trains, evaluates, and promotes models via MLflow and Prefect pipelines |
| **Operator** | Monitors pipeline health, drift alerts, and service availability |

**Baseline features (v0) — currently live at [stockmonitor-ikv.streamlit.app](https://stockmonitor-ikv.streamlit.app):**

| Feature | Description |
|---|---|
| Stock data fetch | OHLCV data from Yahoo Finance via `yfinance` for 7 configurable tickers |
| Moving averages | 50-day and 200-day MA plotted against closing price |
| Interactive charts | Toggle between static (matplotlib) and interactive (altair) charts |
| Configurable period | Time period selector: 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max |
| Correlation heatmap | Heatmap of closing price correlations across selected tickers |
| Yearly % changes | Year-on-year percentage change table (5, 10, or 20-year lookback) |

**Limitation of v0 deployment:**
Streamlit Cloud spins down inactive apps requiring a manual wake-up, making it unsuitable for a persistent prediction service.

**Target state (v1) — two-phase deployment:**

**Phase 9a (Hybrid):** 
1. ML backend - FastAPI, MLflow, Prefect, Postgres, MinIO deployed on a cloud VM.
2. Streamlit Cloud retained as the UI, calling the VM API over HTTPS.

**Phase 9b (Full VM):** 
1. Streamlit service migrated to the same VM (Streamlit Cloud retired)
2. All services self-contained.

---

### 1.2 Goals & Non-Goals

**Goals:**
- Multi-model comparison with tracked metrics (RMSE, MAE, MAPE, directional accuracy)
- Automated model promotion: best model → MLflow `Production` stage
- REST endpoint for N-day ahead price predictions
- Scheduled drift monitoring with report generation
- All services containerized and runnable with `docker-compose up`

**Non-Goals (v1.0):**
- Real-time or intraday streaming data
- Authentication or authorisation on any service
- Financial advice or trading signal generation
- Assets outside equities (crypto, forex, commodities)
- Automated retraining triggered by drift alert *(planned for v2.0)*
- Streamlit Cloud as permanent deployment target *(retained as UI host in Phase 9a; retired in Phase 9b)*
- Managed cloud platform services (AWS RDS, GCP Vertex AI, Azure ML, etc.) — all services run self-hosted via Docker Compose on the VM

---

### 1.3 Architecture

```
Yahoo Finance API
       │
       ▼
[Prefect: Data Ingestion Flow]
       │
       ▼
[Feature Engineering] ──► data/processed/
       │
       ▼
[Prefect: Training Flow] ──► MLflow (Experiments + Model Registry)
                                        │
                              [Production Model]
                                        │
                                        ▼
                              [FastAPI Prediction Service]
                                        │
                          ┌─────────────┴──────────────┐
                          ▼                            ▼
               [Streamlit Dashboard]    [Prefect: Monitoring Flow]
                                                       │
                                               [Evidently Reports]
```

**Docker Compose services:**

| Service | Purpose | Port |
|---|---|---|
| `postgres` | Backend DB for MLflow + Prefect | 5432 |
| `minio` | Artifact store for MLflow | 9000 |
| `mlflow` | Experiment tracking + model registry | 5000 |
| `prefect` | Pipeline orchestration UI | 4200 |
| `api` | FastAPI prediction service | 8000 |
| `streamlit` | Existing UI (augmented) | 8501 |

---

## 2. Requirements

### 2.1 Functional Requirements

#### Data Ingestion

| ID | Requirement | Status |
|---|---|---|
| FR-DI-001 | Fetch OHLCV data for configurable tickers from Yahoo Finance via `yfinance` | Done |
| FR-DI-002 | Validate fetched data against a schema (column names, dtypes, no nulls in Close/Volume) | Done |
| FR-DI-003 | Save raw data to a data folder `data/raw/{ticker}_{date}.csv` | Done |
| FR-DI-004 | Save a reference window snapshot to `data/reference/` on first run | Done |
| FR-DI-005 | Ingestion flow shall be schedulable via Prefect and runnable manually | Done |

#### Feature Engineering

| ID | Requirement |
|---|---|
| FR-FE-001 | Compute lag features for Close price: 1, 5, 10, 20 days |
| FR-FE-002 | Compute RSI (14-day), MACD (12/26/9), Bollinger Bands (20-day), and ATR (14-day) |
| FR-FE-003 | Add calendar features: day-of-week, month, quarter |
| FR-FE-004 | Feature toggles shall be configurable via `configs/features_config.yaml` |
| FR-FE-005 | Save processed datasets to `data/processed/{ticker}_{date}.csv` |

#### Model Training

| ID | Requirement |
|---|---|
| FR-MT-001 | Train four model types: Linear Regression, XGBoost, LSTM (PyTorch), Prophet |
| FR-MT-002 | Log RMSE, MAE, MAPE, and directional accuracy to MLflow per run |
| FR-MT-003 | Log all hyperparameters from `configs/model_config.yaml` to MLflow |
| FR-MT-004 | Promote the model with the lowest validation RMSE to `Production` in MLflow Model Registry |
| FR-MT-005 | Transition previous Production model to `Archived` before promotion |

#### Prediction API

| ID | Requirement |
|---|---|
| FR-API-001 | Expose `POST /predict` accepting `{"ticker": str, "horizon_days": int}` |
| FR-API-002 | Response shall include predicted prices, model name, model version, and timestamp |
| FR-API-003 | Expose `GET /health` returning service status and loaded model version |
| FR-API-004 | Expose `GET /drift-status` returning the latest drift score and alert flag |
| FR-API-005 | Load the MLflow `Production` model at startup |

#### Drift Monitoring

| ID | Requirement |
|---|---|
| FR-DM-001 | Compare current data window against the reference snapshot using Evidently `DataDriftPreset` |
| FR-DM-002 | Evaluate prediction quality using Evidently `RegressionPerformancePreset` |
| FR-DM-003 | Save a timestamped HTML report to `monitoring/reports/` per run |
| FR-DM-004 | Log drift metrics (drift score, number of drifted features) to MLflow |
| FR-DM-005 | Raise a drift alert if drift score exceeds a configurable threshold |

#### Streamlit Dashboard

| ID | Requirement |
|---|---|
| FR-UI-001 | Existing tabs (Stock Monitor, Correlation, Yearly Changes) remain functionally unchanged |
| FR-UI-002 | New **Predictions** tab displays multi-model forecasts via the FastAPI `/predict` endpoint |
| FR-UI-003 | New **Model Registry** tab displays MLflow experiment runs with key metrics |
| FR-UI-004 | New **Drift Reports** tab renders the latest Evidently HTML report |

---

### 2.2 Non-Functional Requirements

| ID | Category | Requirement |
|---|---|---|
| NFR-001 | Performance | `POST /predict` p95 response time < 2 seconds for horizon ≤ 30 days |
| NFR-002 | Performance | Streamlit dashboard initial load < 5 seconds |
| NFR-003 | Reliability | All Docker Compose services shall define health checks |
| NFR-004 | Reproducibility | Training runs shall be fully reproducible given the same data and config (fixed random seeds) |
| NFR-005 | Data Freshness | Ingestion pipeline supports daily scheduling; data older than 1 day triggers a staleness warning |
| NFR-006 | Portability | The full system runs via `docker-compose up` on both local machines and a cloud VM — no prerequisites beyond Docker |
| NFR-007 | Testability | All pipeline stages and API endpoints shall have automated tests in `tests/` |

---

### 2.3 Technical Stack

**Runtime:** Python 3.10, Docker 24.x+, Docker Compose v2+

| Component | Technology | Version |
|---|---|---|
| Dashboard | Streamlit | 1.38.x |
| REST API | FastAPI + Uvicorn | 0.115.x |
| Data Source | yfinance | 0.2.x |
| ML — Baseline | scikit-learn (Linear Regression) | 1.5.x |
| ML — Gradient Boosting | XGBoost | 2.x |
| ML — Deep Learning | PyTorch (LSTM) | 2.x |
| ML — Time Series | Prophet | 1.1.x |
| Experiment Tracking | MLflow | 2.x |
| Orchestration | Prefect | 3.x |
| Drift Monitoring | Evidently AI | 0.4.x |
| Schema Validation | Pandera | 0.20.x |
| Backend DB | PostgreSQL | 15 |
| Artifact Store | MinIO | latest |

**Configuration:**
- Hyperparameters: `configs/model_config.yaml`
- Feature toggles: `configs/features_config.yaml`
- Service connection strings: `.env` (see `.env.example`)

---

## 3. User Stories

### End User
- **US-01:** As an end user, I want to select a ticker and see predicted closing prices for the next 5–30 days.
- **US-02:** As an end user, I want to see which model produced the prediction and its reported accuracy.
- **US-03:** As an end user, I want existing charts (moving averages, correlation, yearly changes) unchanged.

### ML Engineer
- **US-04:** As an ML engineer, I want to trigger a training run for all models from the Prefect UI.
- **US-05:** As an ML engineer, I want to compare model metrics across runs in MLflow.
- **US-06:** As an ML engineer, I want the best-performing model automatically promoted to `Production`.
- **US-07:** As an ML engineer, I want to configure model hyperparameters via `configs/model_config.yaml` without touching code.

### Operator
- **US-08:** As an operator, I want a drift report showing whether input feature distributions have shifted.
- **US-09:** As an operator, I want a `/health` endpoint to check API service status.
- **US-10:** As an operator, I want all services to start with a single `docker-compose up` command.

---

## 4. Success Criteria

| ID | Criterion | Target |
|---|---|---|
| SC-001 | XGBoost RMSE on held-out test set | < 3% of mean Close price |
| SC-002 | Directional accuracy (up/down) of best model | ≥ 55% |
| SC-003 | `POST /predict` p95 latency | < 2 seconds |
| SC-004 | All services healthy after `docker-compose up` | < 3 minutes |
| SC-005 | Drift report generated for a 30-day window | < 60 seconds end-to-end |
| SC-006 | Test suite pass rate | 100% on `pytest tests/` |
| SC-007 | MLflow logs present for all 4 model types | After one training flow run |

---

## 5. Delivery

### 5.1 Milestones

| Phase | Description | Key Deliverables |
|---|---|---|
| 1 | Infrastructure Setup | `docker-compose.yml`, `.env.example`, updated `requirements.txt` |
| 2 | Feature Engineering | `utils/feature_engineering.py`, `configs/features_config.yaml` |
| 3 | Data Pipeline | `pipelines/data_ingestion.py`, schema validation, `data/` directories |
| 4 | Model Training | 4 model files, `pipelines/training_pipeline.py`, `configs/model_config.yaml` |
| 5 | Inference API | `api/main.py`, `api/schemas.py`, all endpoints |
| 6 | Drift Monitoring | `monitoring/evidently_config.py`, `pipelines/monitoring_pipeline.py` |
| 7 | Dashboard Update | 3 new tabs in `app.py` (Predictions, Model Registry, Drift Reports) |
| 8 | Tests | `tests/test_data_pipeline.py`, `test_models.py`, `test_api.py` |
| 9a | Hybrid Deployment | Provision VM, deploy backend services (FastAPI, MLflow, Prefect, Postgres, MinIO) via Docker Compose, configure CORS on FastAPI, point Streamlit Cloud app at VM API |
| 9b | Full VM Migration | Add Streamlit service to Docker Compose, configure reverse proxy (Nginx/Caddy), set up DNS, retire Streamlit Cloud deployment |

Phases 1–3 are sequential. Phases 4 and 5 may run in parallel after Phase 3. Phase 6 may start in parallel with Phase 5. Phase 7 depends on Phases 5 and 6. Phase 8 runs alongside all phases.

---

### 5.2 Risks & Mitigations

| ID | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R-001 | `yfinance` API changes or rate-limits break ingestion | Medium | High | Pin version; add retry logic with exponential backoff; support CSV fallback |
| R-002 | Data leakage in feature engineering | Medium | High | Strict temporal train/val/test split; lag features reference only past timestamps |
| R-003 | LSTM overfitting on small datasets | High | Medium | Dropout, early stopping, cross-validation; compare against simpler baselines |
| R-004 | MLflow/Prefect backend not ready on first startup | Low | Medium | `depends_on` + health checks in Docker Compose; retry logic in pipeline startup |
| R-005 | MinIO misconfiguration causes model save failures | Low | High | Smoke test artifact upload; document bucket setup in `.env.example` |
| R-006 | Prophet installation conflicts with PyTorch dependencies | Medium | Low | Document known conflicts in `requirements.txt` comments |
| R-007 | Cloud VM free-tier resource limits constrain model training | Medium | Medium | Profile memory usage locally first; offload LSTM training to a separate step or reduce model size if needed |
| R-008 | CORS misconfiguration blocks Streamlit Cloud from calling VM API (Phase 9a) | Medium | High | Configure FastAPI `CORSMiddleware` with explicit Streamlit Cloud origin; test with browser dev tools before Phase 9b |

---

## 6. Appendix

### Supported Tickers (v1.0)
`^GSPC`, `AMZN`, `TSLA`, `NVDA`, `AAPL`, `GOOGL`, `MSFT`

### MLflow Model Registry Stage Transitions
```
None → Staging (after training) → Production (if best RMSE) → Archived (on next promotion)
```

### Drift Alert Threshold
Default: `drift_score > 0.5` (configurable via `.env` as `DRIFT_THRESHOLD`).
Evidently defines drift score as the share of features that have drifted.

### Glossary

| Term | Definition |
|---|---|
| **RMSE** | Root Mean Squared Error — primary model selection metric |
| **MAPE** | Mean Absolute Percentage Error — scale-independent accuracy measure |
| **Directional Accuracy** | % of predictions where predicted direction (up/down) matches actual |
| **Data Drift** | Shift in input feature distribution relative to the reference window |
| **Model Drift** | Degradation in prediction performance over time |
| **Reference Window** | Historical data snapshot used as the baseline for drift comparison |