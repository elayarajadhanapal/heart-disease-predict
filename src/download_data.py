
import os
import pandas as pd
from ucimlrepo import fetch_ucirepo


RAW_DATA_PATH = "data/raw/heart_disease_raw.csv"
PROCESSED_DATA_PATH = "data/processed/heart_disease_cleaned.csv"


def download_and_prepare_data():
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    print("Downloading UCI Heart Disease dataset...")

    heart_disease = fetch_ucirepo(id=45)

    X = heart_disease.data.features.copy()
    y = heart_disease.data.targets.copy()

    df = pd.concat([X, y], axis=1)

    # Preserve the acquired dataset before cleaning.
    df.to_csv(RAW_DATA_PATH, index=False)

    print(f"Raw dataset saved to: {RAW_DATA_PATH}")
    print(f"Raw dataset shape: {df.shape}")

    if "num" in df.columns:
        df = df.rename(columns={"num": "target"})

    if "target" in df.columns:
        df["target"] = (df["target"] > 0).astype(int)

    df = df.replace("?", pd.NA)

    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df.to_csv(PROCESSED_DATA_PATH, index=False)

    print(f"Processed dataset saved to: {PROCESSED_DATA_PATH}")
    print(f"Processed dataset shape: {df.shape}")

    return df


if __name__ == "__main__":
    download_and_prepare_data()

import matplotlib.pyplot as plt
import seaborn as sns


def run_eda(
    data_path="data/processed/heart_disease_cleaned.csv",
    output_dir="screenshots"
):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(data_path)

    sns.set_theme(style="whitegrid")

    # 1. Class balance
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="target")
    plt.title("Heart Disease Class Distribution")
    plt.xlabel("Target")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "01_class_balance.png"))
    plt.close()

    # 2. Feature histograms
    numeric_columns = [
        "age",
        "trestbps",
        "chol",
        "thalach",
        "oldpeak"
    ]

    df[numeric_columns].hist(figsize=(12, 8), bins=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "02_feature_histograms.png"))
    plt.close()

    # 3. Correlation heatmap
    plt.figure(figsize=(12, 8))
    correlation = df.corr(numeric_only=True)
    sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "03_correlation_heatmap.png"))
    plt.close()

    # 4. Clinical interaction
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="age", y="thalach", hue="target")
    plt.title("Age vs Maximum Heart Rate")
    plt.xlabel("Age")
    plt.ylabel("Maximum Heart Rate")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "04_clinical_interaction.png"))
    plt.close()

    print("EDA completed successfully.")
    print(f"EDA figures saved to: {output_dir}")
