from airflow.decorators import dag, task
from airflow.utils.dates import days_ago
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import mlflow
import boto3
import os
import tempfile
import pickle

S3_BUCKET = "mlflow"
S3_PREFIX = "emnist/"
TRAIN_FILE = "emnist-balanced-train.csv"
TEST_FILE = "emnist-balanced-test.csv"

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))

def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=os.environ["MLFLOW_S3_ENDPOINT_URL"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"]
    )

@dag(
    dag_id="emnist_airflow_pipeline",
    schedule_interval=None,
    start_date=days_ago(1),
    catchup=False,
    tags=["emnist"],
)
def emnist_pipeline():
    
    @task
    def download_csvs():
        s3 = get_s3_client()
        tmpdir = tempfile.mkdtemp()
        train_path = os.path.join(tmpdir, TRAIN_FILE)
        test_path = os.path.join(tmpdir, TEST_FILE)

        s3.download_file(S3_BUCKET, S3_PREFIX + TRAIN_FILE, train_path)
        s3.download_file(S3_BUCKET, S3_PREFIX + TEST_FILE, test_path)
        return {"train_path": train_path, "test_path": test_path}

    @task
    def load_data(paths_dict: dict):
        train_path = paths_dict["train_path"]
        test_path = paths_dict["test_path"]
        df_train = pd.read_csv(train_path, header=None)
        df_test = pd.read_csv(test_path, header=None)

        X_train = df_train.iloc[:, 1:].values.tolist()
        y_train = df_train.iloc[:, 0].values.tolist()
        X_test = df_test.iloc[:, 1:].values.tolist()
        y_test = df_test.iloc[:, 0].values.tolist()

        dataloader_key="preprocessed/emnist_data.pkl"
        s3 = get_s3_client()
        s3.put_object(
            Bucket="mlflow",
            Key=dataloader_key,
            Body=pickle.dumps({
                "X_train": X_train,
                "y_train": y_train,
                "X_test": X_test,
                "y_test": y_test,
            }),
        )
        return dataloader_key

    @task
    def train_model(data_key: str):
        s3 = get_s3_client()

        # Descargar el objeto desde MinIO como bytes
        response = s3.get_object(Bucket="mlflow", Key=data_key)
        data_bytes = response['Body'].read()

        # Cargar el dict de datos con pickle
        data = pickle.loads(data_bytes)

        X_train = data["X_train"]
        y_train = data["y_train"]

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Serializar modelo
        model_bytes = pickle.dumps(model)

        # Guardar modelo serializado en MinIO
        model_key = "models/rf_model.pkl"
        s3.put_object(Bucket="mlflow", Key=model_key, Body=model_bytes)

        return model_key

    @task
    def evaluate_model(model_key: str, data_key: str):
        s3 = get_s3_client()

        # Descargar modelo
        model_response = s3.get_object(Bucket="mlflow", Key=model_key)
        model_bytes = model_response['Body'].read()
        model = pickle.loads(model_bytes)

        # Descargar datos
        data_response = s3.get_object(Bucket="mlflow", Key=data_key)
        data_bytes = data_response['Body'].read()
        data = pickle.loads(data_bytes)

        X_test = data["X_test"]
        y_test = data["y_test"]

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        report = classification_report(y_test, preds, output_dict=True)
        return {"model_key": model_key, "accuracy": acc, "report": report}

    @task
    def log_to_mlflow(results_dict: dict):
        s3 = get_s3_client()

        # Descargar modelo
        model_response = s3.get_object(Bucket="mlflow", Key=results_dict["model_key"])
        model_bytes = model_response['Body'].read()
        model = pickle.loads(model_bytes)
        acc = results_dict["accuracy"]
        report = results_dict["report"]

        with mlflow.start_run():
            mlflow.log_metric("accuracy", acc)
            for label, scores in report.items():
                if isinstance(scores, dict):
                    for metric_name, value in scores.items():
                        mlflow.log_metric(f"{label}_{metric_name}", value)
            mlflow.sklearn.log_model(model, "model")

    paths_dict = download_csvs()

    data_key = load_data(paths_dict)

    model_key = train_model(data_key)

    results_dict = evaluate_model(model_key, data_key)

    log_to_mlflow(results_dict)

dag = emnist_pipeline()
