from prefect import flow, task
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import mlflow
import boto3
import tempfile
import os

S3_BUCKET = "mlflow"
S3_PREFIX = "emnist/"
TRAIN_FILE = "emnist-balanced-train.csv"
TEST_FILE = "emnist-balanced-test.csv"

@task
def download_csvs(tmpdir: str):
    print("🔽 Descargando CSVs desde MinIO...")
    s3 = boto3.client(
        "s3",
        endpoint_url=os.environ["MLFLOW_S3_ENDPOINT_URL"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"]
    )
    train_path = os.path.join(tmpdir, TRAIN_FILE)
    test_path = os.path.join(tmpdir, TEST_FILE)

    s3.download_file(S3_BUCKET, S3_PREFIX + TRAIN_FILE, train_path)
    s3.download_file(S3_BUCKET, S3_PREFIX + TEST_FILE, test_path)

    return train_path, test_path

@task
def load_data(train_path: str, test_path: str):
    df_train = pd.read_csv(train_path, header=None)
    df_test = pd.read_csv(test_path, header=None)

    X_train = df_train.iloc[:, 1:].values
    y_train = df_train.iloc[:, 0].values

    X_test = df_test.iloc[:, 1:].values
    y_test = df_test.iloc[:, 0].values

    return X_train, y_train, X_test, y_test

@task
def train_model(X_train, y_train):
    print("🧠 Entrenando modelo...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

@task
def evaluate_model(model, X_test, y_test):
    print("📈 Evaluando modelo...")
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True)
    return acc, report

@task
def log_to_mlflow(model, acc, report):
    print("📝 Logueando en MLflow...")
    with mlflow.start_run():
        mlflow.log_metric("accuracy", acc)
        for label, scores in report.items():
            if isinstance(scores, dict):
                for metric_name, value in scores.items():
                    mlflow.log_metric(f"{label}_{metric_name}", value)
        mlflow.sklearn.log_model(model, "model")

@flow(name="EMNIST-CSV-Pipeline")
def emnist_pipeline():
    with tempfile.TemporaryDirectory() as tmpdir:
        train_path, test_path = download_csvs(tmpdir)
        X_train, y_train, X_test, y_test = load_data(train_path, test_path)
        model = train_model(X_train, y_train)
        acc, report = evaluate_model(model, X_test, y_test)
        log_to_mlflow(model, acc, report)

if __name__ == "__main__":
    emnist_pipeline()
