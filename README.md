# 🚨 Real-Time Fraud Detection Pipeline

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![Tests](https://img.shields.io/badge/Tests-17%20passing-green)
![License](https://img.shields.io/badge/License-MIT-yellow)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-teal)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)

An end-to-end real-time fraud detection system that processes streaming credit card transactions, scores them using an ensemble of Isolation Forest and Autoencoder models, serves predictions via FastAPI with sub-10ms latency, and monitors for data drift — all containerized with Docker.

## 🏗️ Architecture

**Data Flow:**
1. Transactions arrive via streaming producer (Kafka or local file simulation)
2. Real-time feature engineering computes 25 features per transaction (geospatial, velocity, behavioral)
3. Ensemble model (Isolation Forest + Autoencoder) scores each transaction
4. FastAPI serves predictions with risk tiers (LOW/MEDIUM/HIGH)
5. Monitoring system tracks drift, latency, and triggers alerts

## ✨ Key Features

- **Real-Time Streaming**: Kafka-compatible pipeline with local file simulation for development
- **Ensemble ML**: Semi-supervised Isolation Forest + PyTorch Autoencoder trained on non-fraud data only
- **25 Engineered Features**: Geospatial (haversine distance), velocity (transaction frequency), behavioral (amount patterns), temporal (hour, day, weekend)
- **Sub-10ms Scoring**: FastAPI serving with 8.2ms average prediction latency
- **Drift Detection**: Population Stability Index (PSI) based feature and prediction drift monitoring
- **Automated Alerting**: Rate-limited alerts with configurable channels (console, file, webhook)
- **Cost-Optimized Thresholds**: FN=$500, FP=$50 cost-based threshold optimization
- **Config-Driven**: All hyperparameters externalized in YAML configs
- **17 Unit Tests**: Feature engineering, model scoring, and API endpoint tests
- **CI/CD Pipeline**: GitHub Actions with linting, testing, and Docker build
- **Docker Ready**: Full stack with Kafka, PostgreSQL, Grafana, and API

## 📊 Model Performance

| Model | ROC-AUC | PR-AUC | F1 | Precision | Recall |
|-------|---------|--------|----|-----------|--------|
| Isolation Forest | 0.906 | 0.538 | 0.622 | 0.737 | 0.539 |
| Autoencoder | 0.868 | 0.257 | 0.419 | - | - |
| **Ensemble** | **0.906** | **0.538** | **0.622** | **0.737** | **0.539** |

*Trained on 8K stratified sample (1.7% fraud rate). Pipeline designed to scale to millions of transactions.*

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **ML/DL** | Scikit-learn, PyTorch, NumPy, Pandas |
| **API** | FastAPI, Pydantic, Uvicorn |
| **Streaming** | Confluent-Kafka, JSON Lines |
| **Infrastructure** | Docker, Docker Compose |
| **Monitoring** | Grafana, Prometheus (metrics), Evidently (drift) |
| **Database** | PostgreSQL |
| **Testing** | Pytest (17 tests), GitHub Actions CI/CD |
| **Config** | YAML, python-dotenv |

## 🚀 Quick Start

### Local Development
git clone https://github.com/ashah5123/realtime-fraud-detection.git
cd realtime-fraud-detection
pip install -r requirements.txt
make train
make serve
make demo
make test

### Docker
docker-compose up -d

## 📁 Project Structure

realtime-fraud-detection/
├── configs/                    # YAML configs (model, kafka, features)
├── data/
│   ├── raw/                    # Original dataset (zip)
│   ├── processed/              # Sampled 8K dataset
│   ├── stream_output/          # Streaming pipeline outputs
│   └── monitoring/             # Monitoring logs and metrics
├── models/artifacts/           # Trained model files
├── notebooks/
│   └── 01_EDA.ipynb            # Exploratory Data Analysis
├── results/                    # Evaluation plots and reports
├── scripts/
│   └── sample_data.py          # Dataset sampling script
├── src/
│   ├── data/                   # Data loading, preprocessing, splitting
│   ├── features/               # Feature engineering (19 features)
│   ├── models/                 # Isolation Forest, Autoencoder, Ensemble
│   ├── serving/                # FastAPI application
│   ├── streaming/              # Kafka producer, consumer, processor
│   └── monitoring/             # Drift detection, metrics, alerting
├── tests/                      # 17 unit tests
├── docker-compose.yml          # Full infrastructure stack
├── Dockerfile                  # API container
├── Makefile                    # Common commands
└── requirements.txt            # Dependencies

## 🔍 Feature Engineering

25 features across 5 categories:

| Category | Features | Description |
|----------|----------|-------------|
| Amount | log_amount, amount_decimal, is_round_amount, amount_to_mean_ratio | Transaction amount patterns |
| Temporal | hour_of_day, day_of_week, is_weekend, is_night | Time-based fraud signals |
| Geospatial | distance_to_merchant, is_high_risk_category | Location anomalies via haversine |
| Velocity | time_since_last_txn, rapid_fire_flag, txn_count_1h, txn_count_24h, avg_amount_24h, amount_zscore | Card usage speed and frequency |
| Demographic | age, age_group, category_fraud_rate | Cardholder and merchant risk profiles |

## 🧪 Testing

pytest tests/ -v --cov=src

17 tests covering:
- Feature engineering (7 tests)
- Ensemble model scoring (6 tests)
- API endpoints (4 tests)

## 📈 Future Improvements
- Graph Neural Networks for relationship-based fraud detection
- Online learning for real-time model adaptation
- A/B testing framework for model comparison in production
- Feature importance with SHAP values
- Extended to full Kafka deployment with Kubernetes

## 📄 License
MIT
