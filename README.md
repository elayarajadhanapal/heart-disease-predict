# Heart Disease Prediction – MLOps Project

An end-to-end MLOps project for heart disease classification using the UCI Heart Disease dataset.

The project demonstrates the complete machine learning lifecycle, including data acquisition, preprocessing, exploratory analysis, model development, experiment tracking, automated testing, CI/CD, containerization, Kubernetes deployment, API serving, logging, and monitoring.

---

## 1. Project Overview

The objective of this project is to develop and deploy a machine learning solution that predicts the presence of heart disease based on patient health attributes.

The project implements:

- Data acquisition
- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Feature engineering
- Machine learning model development
- Hyperparameter tuning
- Cross-validation
- Model evaluation
- MLflow experiment tracking
- FastAPI model serving
- Unit testing
- Code quality checks
- GitHub Actions CI/CD
- Docker containerization
- Kubernetes deployment
- Prometheus monitoring
- Grafana visualization
- API request logging

The complete workflow is designed to be reproducible and suitable for deployment-oriented machine learning development.

---

## 2. Architecture

```text
                         ┌───────────────────────┐
                         │  UCI Heart Disease    │
                         │       Dataset         │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │ Data Acquisition &    │
                         │    Preprocessing      │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │   Model Development   │
                         │                       │
                         │ Logistic Regression   │
                         │ Random Forest         │
                         │ GridSearchCV          │
                         │ Cross Validation      │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │       MLflow          │
                         │ Experiment Tracking   │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │    Trained Model      │
                         │ heart_disease_        │
                         │ pipeline.pkl          │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │       FastAPI         │
                         │    Prediction API     │
                         └───────────┬───────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    │                │                │
                    ▼                ▼                ▼
               ┌─────────┐     ┌────────────┐   ┌────────────┐
               │ Docker  │     │ Kubernetes │   │ Prometheus │
               │         │     │            │   │ Monitoring │
               └─────────┘     └─────┬──────┘   └─────┬──────┘
                                     │                 │
                                     ▼                 ▼
                              ┌────────────┐     ┌────────────┐
                              │ 2 Replicas │     │  Grafana   │
                              │   + Probes │     │ Dashboard  │
                              └────────────┘     └────────────┘
```

---

## 3. Repository

GitHub repository:

https://github.com/elayaradhanapal/heart-disease-predict

---

## 4. Repository Structure

```text
heart-disease-predict/
│
├── .github/
│   └── workflows/
│       └── mlops.yml
│
├── data/
│   └── processed/
│       └── heart_disease_cleaned.csv
│
├── k8s/
│   ├── deployment.yaml
│   └── service.yaml
│
├── models/
│   └── heart_disease_pipeline.pkl
│
├── monitoring/
│   └── prometheus.yml
│
├── screenshots/
│
├── src/
│   ├── app.py
│   ├── download_data.py
│   ├── preprocess.py
│   └── train.py
│
├── tests/
│   └── test_pipeline.py
│
├── Dockerfile
├── docker-compose.monitoring.yml
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## 5. Dataset

The project uses the UCI Heart Disease dataset.

The dataset is acquired programmatically using the `ucimlrepo` Python package.

The automated data acquisition process is implemented in:

```text
src/download_data.py
```

The processed dataset is stored under:

```text
data/processed/heart_disease_cleaned.csv
```

The target variable is:

```text
target
```

The classification problem predicts whether heart disease is present.

---

## 6. Data Preprocessing

The preprocessing pipeline is implemented in:

```text
src/preprocess.py
```

The project separates features into numerical, categorical, and binary groups.

### Numerical Features

```text
age
trestbps
chol
thalach
oldpeak
```

Numerical preprocessing:

- Median imputation
- Standard scaling

### Categorical Features

```text
cp
restecg
slope
ca
thal
```

Categorical preprocessing:

- Most-frequent imputation
- One-hot encoding
- Unknown categories handled using `handle_unknown="ignore"`

### Binary Features

```text
sex
fbs
exang
```

Binary features use most-frequent imputation.

The complete preprocessing pipeline is implemented using scikit-learn's `ColumnTransformer`.

---

## 7. Machine Learning Models

The model training pipeline is implemented in:

```text
src/train.py
```

The project evaluates two classification algorithms:

1. Logistic Regression
2. Random Forest

The data is divided into training and testing datasets using an 80/20 split with stratification.

Model selection and hyperparameter tuning are performed using:

```text
GridSearchCV
```

with:

```text
roc_auc
```

as the primary optimization metric.

Five-fold stratified cross-validation is also used during model evaluation.

---

## 8. Model Evaluation

The following evaluation metrics are calculated:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

Additional evaluation artifacts include:

- Confusion matrix
- ROC curve

The final selected model is saved as:

```text
models/heart_disease_pipeline.pkl
```

The saved artifact contains the preprocessing pipeline together with the trained classifier, allowing the same preprocessing logic to be used during inference.

---

## 9. MLflow Experiment Tracking

MLflow is used for experiment tracking.

The experiment name is:

```text
Heart_Disease_Classification_MLOps
```

The training pipeline records:

- Model parameters
- Cross-validation metrics
- Test metrics
- Model artifacts
- Confusion matrix
- ROC curve
- Trained model

The MLflow tracking URI used by the training pipeline is:

```text
sqlite:///mlflow.db
```

The trained model artifact is also stored under:

```text
models/heart_disease_pipeline.pkl
```

---

## 10. FastAPI Prediction Service

The trained model is exposed through a FastAPI application.

The API implementation is located at:

```text
src/app.py
```

The application runs on port:

```text
8000
```

### Start the API

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
uvicorn src.app:app --host 0.0.0.0 --port 8000
```

The API will be available at:

```text
http://localhost:8000
```

Interactive Swagger documentation:

```text
http://localhost:8000/docs
```

---

## 11. Health Check

The API provides a health endpoint:

```http
GET /health
```

Test it using:

```bash
curl http://localhost:8000/health
```

Example response:

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

The health endpoint is also used by Kubernetes readiness and liveness probes.

---

## 12. Prediction API

The prediction endpoint is:

```http
POST /predict
```

Example request:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 55,
    "sex": 1,
    "cp": 2,
    "trestbps": 140,
    "chol": 240,
    "fbs": 0,
    "restecg": 1,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 1.5,
    "slope": 2,
    "ca": 0,
    "thal": 2
  }'
```

Example response:

```json
{
  "prediction": 1,
  "prediction_label": "Heart Disease Detected",
  "probability_no_disease": 0.4743,
  "probability_disease": 0.5257
}
```

The API returns:

- Predicted class
- Human-readable prediction label
- Probability of no disease
- Probability of disease

---

## 13. Automated Testing

Unit tests are implemented in:

```text
tests/test_pipeline.py
```

The tests cover:

- Processed dataset availability
- Preprocessor transformation
- Model artifact loading
- Model inference

Run the tests using:

```bash
PYTHONPATH=. pytest -q
```

Expected result:

```text
3 passed
```

---

## 14. Code Quality

Flake8 is used for code quality checks.

The CI/CD pipeline performs:

```bash
flake8 src/ tests/ --count --select=E9,F63,F7,F82 --show-source --statistics
```

Additional code-quality checks are also executed with a maximum line length of 127 characters.

---

## 15. CI/CD with GitHub Actions

The project uses GitHub Actions for automated CI/CD.

The workflow performs the following steps:

```text
GitHub Repository
        │
        ▼
Checkout
        │
        ▼
Python Setup
        │
        ▼
Install Dependencies
        │
        ▼
Install Project
        │
        ▼
Flake8
        │
        ▼
Pytest
        │
        ▼
Data Acquisition
        │
        ▼
Model Training
        │
        ▼
Archive Model & MLflow Artifacts
```

The workflow automatically:

1. Checks out the source code.
2. Sets up Python.
3. Installs dependencies.
4. Installs the project in editable mode.
5. Runs Flake8.
6. Runs unit tests.
7. Downloads/acquires the dataset.
8. Executes model training.
9. Archives trained model and MLflow artifacts.

This provides automated validation of the machine learning pipeline whenever changes are pushed to GitHub.

---

## 16. Docker

The application is containerized using Docker.

The Docker configuration is provided in:

```text
Dockerfile
```

The container uses Python 3.10.

### Build the Docker image

```bash
docker build -t heart-disease-api:latest .
```

### Run the container

```bash
docker run --rm -p 8000:8000 heart-disease-api:latest
```

### Test the container

```bash
curl http://localhost:8000/health
```

The FastAPI service is exposed on:

```text
http://localhost:8000
```

Swagger documentation:

```text
http://localhost:8000/docs
```

---

## 17. Kubernetes Deployment

The application is deployed to Kubernetes using the manifests under:

```text
k8s/
```

The deployment configuration is:

```text
k8s/deployment.yaml
```

The service configuration is:

```text
k8s/service.yaml
```

The Kubernetes deployment uses:

- 2 application replicas
- CPU requests and limits
- Memory requests and limits
- Readiness probe
- Liveness probe

### Build and Load Image into Minikube

Build the image:

```bash
docker build -t heart-disease-api:latest .
```

Load it into Minikube:

```bash
minikube image load heart-disease-api:latest
```

### Deploy the Application

Apply the Deployment:

```bash
kubectl apply -f k8s/deployment.yaml
```

Apply the Service:

```bash
kubectl apply -f k8s/service.yaml
```

### Verify Deployment

```bash
kubectl get deployment
```

```bash
kubectl get pods -o wide
```

```bash
kubectl get service
```

### Access the Application

For Minikube:

```bash
minikube service heart-disease-api-service --url
```

---

## 18. Kubernetes Health Probes

The Kubernetes Deployment uses the FastAPI health endpoint:

```text
/health
```

as both the readiness and liveness probe.

The readiness probe determines whether the application is ready to receive traffic.

The liveness probe determines whether the application is healthy and should continue running.

---

## 19. API Request Logging

The FastAPI application includes request logging middleware.

For each request, the application records information such as:

- HTTP method
- Request path
- HTTP status code
- Request duration

Example log format:

```text
GET /health | status=200 | duration_ms=...
```

This provides basic operational visibility into API requests.

---

## 20. Prometheus Monitoring

The FastAPI application exposes Prometheus-compatible metrics through:

```http
GET /metrics
```

Prometheus instrumentation is implemented using:

```text
prometheus-fastapi-instrumentator
```

The metrics endpoint can be tested using:

```bash
curl http://localhost:8000/metrics
```

The Prometheus configuration is located at:

```text
monitoring/prometheus.yml
```

Prometheus is configured to scrape the API every five seconds.

---

## 21. Prometheus Queries

Prometheus is available at:

```text
http://localhost:9090
```

### Check target availability

```promql
up
```

This verifies whether Prometheus can successfully scrape the application.

### View HTTP request metrics

```promql
http_requests_total
```

### View request rate by endpoint

```promql
sum by (handler) (rate(http_requests_total[1m]))
```

These queries provide visibility into API request activity.

---

## 22. Grafana Monitoring

Grafana is used to visualize the Prometheus metrics.

Grafana is available at:

```text
http://localhost:3000
```

The Prometheus data source inside the Docker Compose network is:

```text
http://prometheus:9090
```

Example dashboard metrics include:

- API request count
- API request rate
- Requests by endpoint
- Request duration

The monitoring setup provides visibility into the operational behavior of the prediction API.

---

## 23. Docker Compose Monitoring Stack

The monitoring stack is defined in:

```text
docker-compose.monitoring.yml
```

The stack contains:

```text
FastAPI Application
        │
        ▼
   Prometheus
        │
        ▼
     Grafana
```

Start the monitoring stack:

```bash
docker compose -f docker-compose.monitoring.yml up -d
```

Check the containers:

```bash
docker compose -f docker-compose.monitoring.yml ps
```

Stop the monitoring stack:

```bash
docker compose -f docker-compose.monitoring.yml down
```

---

## 24. Generate API Traffic for Monitoring

To generate API traffic for Prometheus and Grafana:

```bash
for i in {1..20}; do
  curl -s http://localhost:8000/health > /dev/null
done
```

A prediction request can also be generated using the `/predict` endpoint.

After generating traffic, Prometheus can be queried for the resulting HTTP request metrics.

---

## 25. Local Setup

Clone the repository:

```bash
git clone https://github.com/elayaradhanapal/heart-disease-predict.git
```

Change to the project directory:

```bash
cd heart-disease-predict
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment.

### Linux/macOS

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Install the project in editable mode:

```bash
pip install -e .
```

Run the tests:

```bash
PYTHONPATH=. pytest -q
```

---

## 26. End-to-End Workflow

The complete project workflow is:

```text
                 DATA
                  │
                  ▼
        Data Acquisition
                  │
                  ▼
         Data Preprocessing
                  │
                  ▼
          Feature Engineering
                  │
                  ▼
         Model Development
                  │
                  ▼
       Hyperparameter Tuning
                  │
                  ▼
       Cross-Validation
                  │
                  ▼
        Model Evaluation
                  │
                  ▼
             MLflow
                  │
                  ▼
        Model Serialization
                  │
                  ▼
              FastAPI
                  │
          ┌───────┴────────┐
          ▼                ▼
       Docker          Kubernetes
                            │
                            ▼
                       API Service
                            │
                            ▼
                       API Requests
                            │
                ┌───────────┴───────────┐
                ▼                       ▼
             Logging              Prometheus
                                        │
                                        ▼
                                     Grafana
```

---

## 27. Technology Stack

| Category | Technology |
|---|---|
| Programming Language | Python |
| Data Processing | pandas, NumPy |
| Machine Learning | scikit-learn |
| Experiment Tracking | MLflow |
| API Framework | FastAPI |
| API Server | Uvicorn |
| Testing | pytest |
| Code Quality | Flake8 |
| Containerization | Docker |
| Orchestration | Kubernetes |
| Local Kubernetes | Minikube |
| Monitoring | Prometheus |
| Visualization | Grafana |
| CI/CD | GitHub Actions |

---

## 28. Key MLOps Capabilities Demonstrated

### Data Pipeline

Automated data acquisition and preprocessing.

### Reproducible Model Training

The training pipeline uses a defined preprocessing and model pipeline.

### Experiment Tracking

MLflow tracks model parameters, metrics, and artifacts.

### Automated Testing

Unit tests validate the preprocessing and inference pipeline.

### CI/CD

GitHub Actions automatically performs validation, testing, data acquisition, and model training.

### Containerization

The FastAPI service is packaged as a Docker image.

### Production-style Deployment

The application is deployed to Kubernetes with two replicas and health probes.

### Monitoring

Prometheus collects API metrics and Grafana provides visualization.

### Logging

API requests are logged with method, endpoint, status, and request duration.

---

## 29. Evidence and Screenshots

The `screenshots/` directory contains evidence of the implementation, including relevant:

- EDA results
- Model evaluation
- MLflow experiments
- CI/CD execution
- Docker/API execution
- Kubernetes deployment
- Prometheus monitoring
- Grafana monitoring

These screenshots provide supporting evidence for the implemented MLOps workflow.

---

## 30. Conclusion

This project demonstrates an end-to-end MLOps workflow for a machine learning classification problem.

The solution progresses from:

```text
Data Acquisition
       ↓
Data Processing
       ↓
Model Development
       ↓
Experiment Tracking
       ↓
Testing
       ↓
CI/CD
       ↓
Docker
       ↓
Kubernetes
       ↓
API Serving
       ↓
Monitoring
```

The resulting system provides a reproducible machine learning pipeline together with automated validation, containerized deployment, Kubernetes orchestration, API serving, logging, and operational monitoring.
