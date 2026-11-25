# Cinema Show Occupancy (Lucknow) - MP028

**Group ID:** G084  
**Problem ID:** MP028  
**Faculty:** Gaurav Tripathi  
**Batch:** 5  
**Repository:** https://github.com/SammanDevDubey/cinema_show_occupancy-  
**Data hash (SHA-256 of original train.csv):** `16eaf6fd83b4e69343539e8248617dbb9e237fb97ec682e1a5140442fc44ed86`

## Members

- S24CSEU0454 - Samman Dev Dubey - GitHub: SammanDevDubey
- S24CSEU0469 - Sanyam Singla - GitHub: <!-- TODO: Sanyam's exact GitHub username -->
- S24CSEU0445 - Saketh Reddy Sanikommu - GitHub: <!-- TODO: Saketh's exact GitHub username -->
- S24CSEU0476 - Satvick Miglani - GitHub: <!-- TODO: Satvick's exact GitHub username -->

## Member Roles

| Member | Enrolment No. | Role | Owns |
|---|---|---|---|
| Samman Dev Dubey | S24CSEU0454 | Data lead | `README.md`, `params.yaml`, DVC, data checks & cleaning, `findings.yaml` (`data_issues`), `tests/test_data_clean.py` |
| Sanyam Singla | S24CSEU0469 | Model lead | `modeling.py`, `train.py`, `predict.py`, MLflow v1 & `@champion`, `findings.yaml` (`tracking`) <!-- TODO: Sanyam --> |
| Saketh Reddy Sanikommu | S24CSEU0445 | Service lead | `app/main.py`, `Dockerfile`, `.dockerignore`, API tests <!-- TODO: Saketh --> |
| Satvick Miglani | S24CSEU0476 | Ops lead | `.github/workflows/ci.yml`, `k8s/`, JSON logging, `monitoring.py`, `retrain.py`, `findings.yaml` (`monitoring`) <!-- TODO: Satvick --> |

## Problem Summary

Regression to predict **`occupancy_pct`** (percentage of cinema seats filled) for single-screen cinemas in Lucknow, evaluated using **MAE** (Mean Absolute Error). The end-to-end MLOps pipeline covers:

1. **Data versioning** with DVC  
2. **Experiment tracking** with MLflow (SQLite backend)  
3. **REST API** with FastAPI  
4. **Containerisation** with Docker  
5. **CI** with GitHub Actions  
6. **Deployment specs** with Kubernetes YAML  
7. **Drift monitoring & retraining** (PSI, per-group MAE, v2 promotion rule)

## Repository Layout

```text
README.md                           group id, problem id, members, setup & commands
requirements.txt                    pinned dependencies (e.g. pandas==2.2.3)
params.yaml                         central settings (paths, mlflow, features, model grid, monitoring)
src/mp028/                          core package (config, findings, data_checks, data_clean, modeling, train, predict, monitoring, retrain)
tests/                              pytest unit and API tests
data/raw/*.dvc                      DVC metadata files for train.csv and later.csv (CSVs are gitignored)
.dvc/config                         DVC configuration
mlflow.db                           MLflow SQLite tracking & registry database <!-- TODO: Sanyam / Satvick -->
app/main.py                         FastAPI service: POST /predict and GET /health <!-- TODO: Saketh -->
models/model.joblib                 champion model loaded by the API <!-- TODO: Sanyam / Satvick -->
Dockerfile                          container image definition <!-- TODO: Saketh -->
.dockerignore                       files excluded from Docker build context <!-- TODO: Saketh -->
.github/workflows/ci.yml            GitHub Actions workflow running pytest on push <!-- TODO: Satvick -->
k8s/deployment.yaml                 Kubernetes Deployment (mp028-api, 3 replicas) <!-- TODO: Satvick -->
k8s/service.yaml                    Kubernetes Service (mp028-svc, port 80 -> 8000) <!-- TODO: Satvick -->
reports/findings.yaml               structured findings filled by each member
predictions/holdout_predictions.csv final holdout predictions (record_id,prediction) <!-- TODO: Sanyam / Satvick -->
```

## Setup

Requires **Python 3.12**. Run all commands from the repository root:

```bash
pip install -r requirements.txt
dvc pull                       # or place the original raw CSVs in data/raw/
```

## Running Each Stage

- **Data checks / clean:**
  ```bash
  python -m mp028.data_checks
  ```
- **Run tests:**
  ```bash
  pytest
  ```
- **Train model (MLflow):** <!-- TODO: Sanyam -->
  ```bash
  python -m mp028.train
  ```
- **Serve prediction API:** <!-- TODO: Saketh -->
  ```bash
  uvicorn app.main:app --port 8000
  ```
- **Docker build & run:** <!-- TODO: Saketh -->
  ```bash
  docker build -t mp028-api:1.0 .
  docker run -p 8000:8000 mp028-api:1.0
  ```
- **Drift monitoring:** <!-- TODO: Satvick -->
  ```bash
  python -m mp028.monitoring
  ```
- **Retrain & promotion check:** <!-- TODO: Satvick -->
  ```bash
  python -m mp028.retrain
  ```
- **Holdout predictions:** <!-- TODO: Sanyam / Satvick -->
  ```bash
  python -m mp028.predict
  ```
