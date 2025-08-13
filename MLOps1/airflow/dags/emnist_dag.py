from airflow.decorators import dag, task
from airflow.utils.dates import days_ago
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import mlflow
from mlflow.tracking import MlflowClient
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

        model = RandomForestClassifier(n_estimators=100)
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

        try:
            # Descargar modelo
            model_response = s3.get_object(Bucket="mlflow", Key=results_dict["model_key"])
            model_bytes = model_response['Body'].read()
            model = pickle.loads(model_bytes)
            acc = results_dict["accuracy"]
            report = results_dict["report"]

            with mlflow.start_run() as run:
                mlflow_run_id = run.info.run_id
                mlflow.log_metric("accuracy", acc)
                # Loguear métricas globales del classification_report
                if "macro avg" in report:
                    for metric_name, value in report["macro avg"].items():
                        mlflow.log_metric(f"macro_avg_{metric_name}", value)

                if "weighted avg" in report:
                    for metric_name, value in report["weighted avg"].items():
                        mlflow.log_metric(f"weighted_avg_{metric_name}", value)
                
                # A veces 'accuracy' también está como clave en report
                if "accuracy" in report and isinstance(report["accuracy"], (int, float)):
                    mlflow.log_metric("report_accuracy", report["accuracy"])

                mlflow.sklearn.log_model(model, "model")
            mlflow.end_run(status="FINISHED")
            return mlflow_run_id
        except Exception as e:
            mlflow.end_run(status="FAILED")
            raise

    @task
    def promote_best_model(mlflow_run_id: str, registered_model_name: str = "EMNIST-Model"):
        client = MlflowClient()

        existing_models = [m.name for m in client.search_registered_models()]
        if registered_model_name not in existing_models:
            client.create_registered_model(registered_model_name)

        # Registrar la nueva versión del modelo en Model Registry de MLFlow
        model_version = client.create_model_version(
            name=registered_model_name,
            source=f"runs:/{mlflow_run_id}/model",
            run_id=mlflow_run_id
        )

        # Obtener todas las versiones del modelo
        versions = client.search_model_versions(f"name='{registered_model_name}'")

        # Buscar mejor versión por accuracy (u otra métrica)
        best_version = None
        best_acc = -float("inf")

        # Buscar la mejor versión por accuracy
        for v in versions:
            run_data = client.get_run(v.run_id).data
            acc_v = run_data.metrics.get("accuracy", 0)
            if acc_v > best_acc:
                best_acc = acc_v
                best_version = v

        # Promocionar la mejor versión y archivar las demás
        if best_version:
            for v in versions:
                if v.version == best_version.version:
                    # Poner tag stage = Production
                    client.set_model_version_tag(
                        name=registered_model_name,
                        version=v.version,
                        key="stage",
                        value="Production"
                    )
                else:
                    # Poner tag stage = Archived
                    client.set_model_version_tag(
                        name=registered_model_name,
                        version=v.version,
                        key="stage",
                        value="Archived"
                    )

    paths_dict = download_csvs()

    data_key = load_data(paths_dict)

    model_key = train_model(data_key)

    results_dict = evaluate_model(model_key, data_key)

    mlflow_run_id = log_to_mlflow(results_dict)

    promote_best_model(mlflow_run_id)

dag = emnist_pipeline()
