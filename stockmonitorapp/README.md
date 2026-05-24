# Stock Monitor and ML Prediction

**Status:** v0 live · v1 (MLOps) in development

A stock analysis and prediction platform built on S&P500 data. Currently this project is a Streamlit dashboard for market monitoring and is being extended into a full end-to-end MLOps system with multi-model prediction, experiment tracking, and drift monitoring.

**Live demo (v0):** [stockmonitor-ikv.streamlit.app](https://stockmonitor-ikv.streamlit.app)

---

## Current Features (v0)

- Stock data (OHLCV) fetched from Yahoo Finance via `yfinance`
- 50-day and 200-day moving averages plotted against closing price
- Toggle between static and interactive charts
- Configurable analysis time periods: 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
- Correlation heatmap across selected tickers
- Year-on-year percentage changes table (5, 10, or 20-year lookback)

Supported tickers: `^GSPC` `AMZN` `TSLA` `NVDA` `AAPL` `GOOGL` `MSFT`

---

## Roadmap (v1 — MLOps)

| Phase | Description |
|---|---|
| 1 | Docker Compose infrastructure (MLflow, Prefect, Postgres, MinIO) |
| 2 | Feature engineering (RSI, MACD, Bollinger Bands, lag features) |
| 3 | Prefect data ingestion pipeline with schema validation |
| 4 | Multi-model training — Linear Regression, XGBoost, LSTM, Prophet — logged to MLflow |
| 5 | FastAPI prediction service (`/predict`, `/health`, `/drift-status`) |
| 6 | Evidently AI drift monitoring pipeline with scheduled reports |
| 7 | Streamlit dashboard extended with Predictions, Model Registry, and Drift Report tabs |
| 8 | Automated test suite (`pytest`) |
| 9a | Hybrid deployment — ML backend on cloud VM, Streamlit Cloud calls VM API |
| 9b | Full VM migration — All services on VM, Streamlit Cloud retired |

See [REQUIREMENTS.md](REQUIREMENTS.md) for the full specification.

---

## Project Structure

```
stockmonitorapp/
├── app.py                          # Streamlit dashboard
├── docker-compose.yml              # All services (added in Phase 1)
├── Dockerfile                      # Streamlit container
├── requirements.txt
├── .env.example                    # Service connection strings template
├── configs/                        # Model and feature config (Phase 1-2)
├── data/                           # Raw, processed, and reference datasets (Phase 3)
├── models/                         # ML model implementations (Phase 4)
├── pipelines/                      # Prefect flows (Phases 3, 4, 6)
├── api/                            # FastAPI prediction service (Phase 5)
├── monitoring/                     # Evidently config and reports (Phase 6)
├── utils/                          # Data readers and feature engineering
├── tests/                          # Automated tests (Phase 8)
├── notebooks/                      # EDA and model comparison
└── docs/                           # Architecture diagrams
```

---

## Local Setup (v0 — Streamlit only)

### 1. Create and activate a virtual environment
```sh
python3 -m venv --prompt stocktracker venv
source venv/bin/activate
```

### 2. Install dependencies
```sh
pip install -r requirements.txt
```

### 3. Run the app
```sh
streamlit run app.py
```

### Run with Docker (single container)
```sh
docker build -t stockmonitor -f docker/streamlit/Dockerfile .
docker run -p 8501:8501 stockmonitor
```

---

## Full Stack Setup (v1 — Docker Compose)

> Available after Phase 1 is complete.

```sh
cp .env.example .env
# Edit .env with your configuration
docker-compose up
```

| Service | URL |
|---|---|
| Streamlit dashboard | http://localhost:8501 |
| FastAPI prediction service | http://localhost:8000/docs |
| MLflow experiment tracking | http://localhost:5000 |
| Prefect orchestration UI | http://localhost:4200 |
| MinIO artifact store | http://localhost:9000 |
