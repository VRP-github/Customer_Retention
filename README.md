# 📊 Customer Retention

> End-to-end machine learning system for predicting customer churn, segmenting customers by RFM behavior, and deploying a production-ready prediction API with a Streamlit dashboard for real-time insights.

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.135-green.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.55-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🎯 Business Problem

Acquiring a new customer costs **5–7× more** than retaining an existing one. The goal of this project is to answer two questions with statistical rigour:

1. **Which customers are about to leave?** → Churn prediction model
2. **Which ones are worth saving?** → CLV-weighted retention strategy

The output is a production-ready system that a retention team can query in real-time.

---

## 📂 Repository Structure

```
customer-retention/
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb        # Phase 1: ETL + cleaning
│   ├── 02_rfm_analysis.ipynb         # Phase 2: RFM segmentation
│   ├── 03_churn_prediction.ipynb     # Phase 3: Model training + MLflow
│   └── 04_business_strategy.ipynb    # Phase 4: CLV + ROI simulation
│
├── data/
│   └── online_retail_II.csv          # UCI Online Retail II dataset
│
├── api.py                            # FastAPI inference server
├── app.py                            # Streamlit dashboard
├── get_best_model.py                 # Script to retrieve best model from MLflow
├── requirements.txt                  # Pinned dependencies
├── conftest.py                       # Pytest configuration
│
└── tests/
    └── test_api.py                   # Pytest suite (unit + integration)
```

---

## 🔬 Methodology

### Phase 1 — Data Cleaning
- Source: **Online Retail II UCI** dataset (1M+ transactions)
- Removed rows with missing `Customer ID` (~22.8% of raw data)
- Filtered out returns (`Quantity < 0`) and price anomalies
- Engineered `Revenue = Quantity × Price`; persisted as Parquet

### Phase 2 — RFM Segmentation
| Metric | Definition |
|--------|-----------|
| **Recency** | Days since last purchase |
| **Frequency** | Unique invoice count |
| **Monetary** | Total revenue per customer |

Each metric is scored 1–5 via quantile binning; composite scores yield 5 segments: *Champions*, *Loyal Customers*, *Needs Attention*, *At Risk*, *Lost/Hibernating*.

### Phase 3 — Churn Prediction & Experiment Tracking
- **Label**: `Churned = 1` if no purchase in last 90 days
- **Models trained** (tracked in MLflow): Logistic Regression, Random Forest, **XGBoost** ✅ (best F1)
- **CLV**: BG/NBD + Gamma-Gamma submodel (6-month horizon) via the `lifetimes` library
- Best model auto-registered to MLflow Model Registry

### Phase 4 — Business Strategy & ROI Simulation

| Segment | Strategy | Rationale |
|---------|----------|-----------|
| High CLV + High Churn Risk | **VIP — Save Immediately** (£20 coupon) | Max ROI intervention |
| High CLV + Low Churn Risk | **VIP — Loyalty Programme** | Maintain engagement |
| Low CLV + High Churn Risk | **Do Not Target** | Save marketing budget |
| Low CLV + Low Churn Risk | **Standard Maintenance** | Automated nurture |

**Simulated ROI** (15% coupon success rate assumption):
```
VIPs at risk identified  : ~XXX customers
Cost of coupons          : £XX,XXX
Revenue recovered        : £XXX,XXX
Net campaign profit      : £XXX,XXX
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Virtual environment (recommended)

### Installation & Running

```bash
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\Activate  # on Windows
source .venv/bin/activate  # on macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Start the FastAPI backend
uvicorn api:app --reload --port 8000

# In a second terminal, start the Streamlit frontend
streamlit run app.py
```

The dashboard will be available at `http://localhost:8501` and the API at `http://localhost:8000`.

### Retrieving the Best Model

To find and register the best-performing model from MLflow:

```bash
python get_best_model.py
```

This script searches runs ranked by F1 score and registers the top model.

---

## 🔌 API Reference

### `POST /predict`
```json
{
  "Frequency": 12,
  "Monetary": 3250.75,
  "F_Score": 4,
  "M_Score": 5
}
```
Response:
```json
{
  "prediction": {
    "churn_probability": 0.2341,
    "churn_label": false,
    "risk_tier": "Low"
  },
  "model_name": "Best_Churn_Predictor",
  "model_stage": "latest",
  "api_version": "1.0.0"
}
```

### `POST /predict/batch`
Send `{"customers": [...]}` with up to 500 customer objects.

### `GET /health`
Liveness + readiness probe — suitable for Kubernetes/ECS health checks.

---

## 🧪 Testing

Run the test suite:

```bash
pytest tests/ -v --cov=api --cov-report=term-missing
```

Tests are configured via `conftest.py` and located in the `tests/` directory.

---

## ⚙️ Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MLFLOW_TRACKING_URI` | `sqlite:///mlflow.db` | MLflow backend store |
| `MODEL_NAME` | `Best_Churn_Predictor` | Registered model name |
| `MODEL_STAGE` | `latest` | `Production` / `Staging` / `latest` |
| `API_BASE_URL` | `http://localhost:8000` | Streamlit → API base URL |

---

## 🛣 Future Roadmap

- [ ] Add SHAP explainability endpoint (`/explain`)
- [ ] Swap SQLite for PostgreSQL MLflow backend
- [ ] A/B test tracking for coupon campaign ROI validation
- [ ] Data drift monitoring with Evidently AI
- [ ] Performance optimization for batch predictions
- [ ] Model versioning and rollback capabilities

---

## 📄 License

MIT © 2024