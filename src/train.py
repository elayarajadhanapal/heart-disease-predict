import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)
import mlflow
import mlflow.sklearn
from src.preprocess import build_preprocessor, ALL_FEATURES, TARGET_COL


def plot_and_save_artifacts(y_true, y_pred, y_prob, model_name, output_dir="screenshots"):
    """Generates and saves Confusion Matrix and ROC Curve plots for MLflow logging."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Confusion Matrix - {model_name}", fontweight="bold")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    cm_path = os.path.join(output_dir, f"cm_{model_name.lower()}.png")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300)
    plt.close()

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, color="#d95f02", lw=2, label=f"ROC curve (AUC = {auc_val:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title(f"ROC Curve - {model_name}", fontweight="bold")
    plt.legend(loc="lower right")
    roc_path = os.path.join(output_dir, f"roc_{model_name.lower()}.png")
    plt.tight_layout()
    plt.savefig(roc_path, dpi=300)
    plt.close()

    return cm_path, roc_path


def train_and_evaluate(data_path="data/processed/heart_disease_cleaned.csv", model_dir="models"):
    os.makedirs(model_dir, exist_ok=True)
    df = pd.read_csv(data_path)
    X = df[ALL_FEATURES]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Heart_Disease_Classification_MLOps")

    models_config = {
        "Logistic_Regression": {
            "estimator": LogisticRegression(max_iter=1000, random_state=42),
            "param_grid": {
                "classifier__C": [0.01, 0.1, 1.0, 10.0],
                "classifier__solver": ["lbfgs", "liblinear"],
            },
        },
        "Random_Forest": {
            "estimator": RandomForestClassifier(random_state=42),
            "param_grid": {
                "classifier__n_estimators": [50, 100, 150],
                "classifier__max_depth": [3, 5, 8, None],
                "classifier__min_samples_split": [2, 5],
            },
        },
    }

    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_overall_auc = -1.0
    best_overall_pipeline = None
    best_model_name = ""
    results_summary = []

    for model_name, config in models_config.items():
        pipeline = Pipeline(steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", config["estimator"]),
        ])

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=config["param_grid"],
            cv=cv_strategy,
            scoring="roc_auc",
            n_jobs=-1,
            refit=True,
        )

        with mlflow.start_run(run_name=model_name):
            grid_search.fit(X_train, y_train)
            best_pipeline = grid_search.best_estimator_

            cv_scores = cross_validate(
                best_pipeline,
                X_train,
                y_train,
                cv=cv_strategy,
                scoring=["accuracy", "precision", "recall", "roc_auc"],
            )

            y_pred = best_pipeline.predict(X_test)
            y_prob = best_pipeline.predict_proba(X_test)[:, 1]
            roc_score = float(roc_auc_score(y_test, y_prob))

            metrics = {
                "cv_accuracy_mean": float(np.mean(cv_scores["test_accuracy"])),
                "cv_precision_mean": float(np.mean(cv_scores["test_precision"])),
                "cv_recall_mean": float(np.mean(cv_scores["test_recall"])),
                "cv_roc_auc_mean": float(np.mean(cv_scores["test_roc_auc"])),
                "test_accuracy": float(accuracy_score(y_test, y_pred)),
                "test_precision": float(precision_score(y_test, y_pred)),
                "test_recall": float(recall_score(y_test, y_pred)),
                "test_f1": float(f1_score(y_test, y_pred)),
                "test_roc_auc": roc_score,
            }

            mlflow.log_params(grid_search.best_params_)
            mlflow.log_param("cv_folds", 5)
            mlflow.log_param("model_type", model_name)
            mlflow.log_metrics(metrics)

            cm_path, roc_path = plot_and_save_artifacts(y_test, y_pred, y_prob, model_name)
            mlflow.log_artifact(cm_path, artifact_path="evaluation_plots")
            mlflow.log_artifact(roc_path, artifact_path="evaluation_plots")

            mlflow.sklearn.log_model(
                sk_model=best_pipeline,
                artifact_path="model",
                input_example=X_train.iloc[:2],
            )

            results_summary.append({"model": model_name, **metrics})

            if roc_score > best_overall_auc:
                best_overall_auc = roc_score
                best_overall_pipeline = best_pipeline
                best_model_name = model_name

    final_model_path = os.path.join(model_dir, "heart_disease_pipeline.pkl")
    joblib.dump(best_overall_pipeline, final_model_path)
    print(f"Champion Model: {best_model_name} (Test ROC-AUC: {best_overall_auc:.4f}) -> Saved to {final_model_path}")
    return pd.DataFrame(results_summary)


if __name__ == "__main__":
    train_and_evaluate()
