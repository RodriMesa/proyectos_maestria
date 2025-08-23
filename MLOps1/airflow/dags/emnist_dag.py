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
TRAIN_FILE = "emnist-digits-train.csv"
TEST_FILE = "emnist-digits-test.csv"

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))


def get_s3_client():
    """
    Quick access to s3 bucket.
    """
    return boto3.client(
        "s3",
        endpoint_url=os.environ["MLFLOW_S3_ENDPOINT_URL"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )


@dag(
    dag_id="emnist_airflow_pipeline",
    schedule_interval=None,
    start_date=days_ago(1),
    catchup=False,
    tags=["emnist"],
    is_paused_upon_creation=False,
)
def emnist_pipeline():

    # Step 1: Search raw data files and download them into the s3 bucket
    @task
    def download_csvs():
        s3 = get_s3_client()
        tmpdir = tempfile.mkdtemp()
        train_path = os.path.join(tmpdir, TRAIN_FILE)
        test_path = os.path.join(tmpdir, TEST_FILE)

        s3.download_file(S3_BUCKET, S3_PREFIX + TRAIN_FILE, train_path)
        s3.download_file(S3_BUCKET, S3_PREFIX + TEST_FILE, test_path)

        # XComs prevent passing the files to the next block, so we store the info on
        # s3 and pass the path to the next task.
        return {"train_path": train_path, "test_path": test_path}

    # Step 2: Read the datasets as dataframes and save them as .pkl files in the bucket
    @task
    def load_data(paths_dict: dict):
        # Read data
        train_path = paths_dict["train_path"]
        test_path = paths_dict["test_path"]
        df_train = pd.read_csv(train_path, header=None)
        df_test = pd.read_csv(test_path, header=None)

        # Transform into dataframe
        X_train = df_train.iloc[:, 1:].values.tolist()
        y_train = df_train.iloc[:, 0].values.tolist()
        X_test = df_test.iloc[:, 1:].values.tolist()
        y_test = df_test.iloc[:, 0].values.tolist()

        # Store the dataframes to process them in another tasks. The dataloader key is
        # essential for letting the other tasks know where is the info
        dataloader_key = "preprocessed/emnist_data.pkl"
        s3 = get_s3_client()
        s3.put_object(
            Bucket="mlflow",
            Key=dataloader_key,
            Body=pickle.dumps(
                {
                    "X_train": X_train,
                    "y_train": y_train,
                    "X_test": X_test,
                    "y_test": y_test,
                }
            ),
        )
        return dataloader_key

    # Step 3: Use the train dataframes to train a Random Forest Classifier to identify the numbers written.
    @task
    def train_model(data_key: str):
        s3 = get_s3_client()

        # Download dataframe from MinIO as bytes
        response = s3.get_object(Bucket="mlflow", Key=data_key)
        data_bytes = response["Body"].read()

        # Load pickle dict
        data = pickle.loads(data_bytes)

        X_train = data["X_train"]
        y_train = data["y_train"]

        # Train the classifier. Do not use seeds to have a variety of models & select the best.
        model = RandomForestClassifier(n_estimators=50)
        model.fit(X_train, y_train)

        # Serialize model
        model_bytes = pickle.dumps(model)

        # Store model in the s3 bucket
        model_key = "models/rf_model.pkl"
        s3.put_object(Bucket="mlflow", Key=model_key, Body=model_bytes)

        # Per XCom limitation, return the route to the model instead of the instance itself
        return model_key

    # Step 4: Use the validation set to obtain metrics for the model
    @task
    def evaluate_model(model_key: str, data_key: str):
        s3 = get_s3_client()

        # Download model
        model_response = s3.get_object(Bucket="mlflow", Key=model_key)
        model_bytes = model_response["Body"].read()
        model = pickle.loads(model_bytes)

        # Instance validation sets
        data_response = s3.get_object(Bucket="mlflow", Key=data_key)
        data_bytes = data_response["Body"].read()
        data = pickle.loads(data_bytes)

        X_test = data["X_test"]
        y_test = data["y_test"]

        # Make predictions and save the model's metrics
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        report = classification_report(y_test, preds, output_dict=True)
        return {"model_key": model_key, "accuracy": acc, "report": report}

    # Step 5: Log the experiment to MLFlow
    @task
    def log_to_mlflow(results_dict: dict):
        s3 = get_s3_client()

        try:
            # Download model
            model_response = s3.get_object(Bucket="mlflow", Key=results_dict["model_key"])
            model_bytes = model_response["Body"].read()
            model = pickle.loads(model_bytes)
            acc = results_dict["accuracy"]
            report = results_dict["report"]

            # Start logging metrics for the experiment
            with mlflow.start_run() as run:
                mlflow_run_id = run.info.run_id

                # Log the model's accuracy
                mlflow.log_metric("accuracy", acc)

                # Log global metrics from classification_report
                if "macro avg" in report:
                    for metric_name, value in report["macro avg"].items():
                        mlflow.log_metric(f"macro_avg_{metric_name}", value)

                # Log average metrics from classification_report
                if "weighted avg" in report:
                    for metric_name, value in report["weighted avg"].items():
                        mlflow.log_metric(f"weighted_avg_{metric_name}", value)

                # Log accuracy classification_report (acuracy should be duplicated)
                if "accuracy" in report and isinstance(report["accuracy"], (int, float)):
                    mlflow.log_metric("report_accuracy", report["accuracy"])

                # Save a version of this model
                mlflow.sklearn.log_model(model, "model")
            mlflow.end_run(status="FINISHED")

            # Return the experiment id to manage the model on the following task
            return mlflow_run_id
        except Exception as e:
            mlflow.end_run(status="FAILED")
            raise

    # Step 6: Compare this experiment with other versions of the model to detect the better performing one.
    # Tag the best model in 'Production' stage and 'Archive' the rest of them
    @task
    def promote_best_model(mlflow_run_id: str, registered_model_name: str = "EMNIST-Model"):
        client = MlflowClient()

        # Check the model is registered in MLFlow's Model Registry
        existing_models = [m.name for m in client.search_registered_models()]
        if registered_model_name not in existing_models:
            client.create_registered_model(registered_model_name)

        # Register a new version of the model
        client.create_model_version(
            name=registered_model_name, source=f"runs:/{mlflow_run_id}/model", run_id=mlflow_run_id
        )

        # Get all versions
        versions = client.search_model_versions(f"name='{registered_model_name}'")

        # Search the best version by accuracy
        best_version = None
        best_acc = -float("inf")
        for v in versions:
            run_data = client.get_run(v.run_id).data
            acc_v = run_data.metrics.get("accuracy", 0)
            if acc_v > best_acc:
                best_acc = acc_v
                best_version = v

        # Promote the best version & achive the rest
        if best_version:
            for v in versions:
                if v.version == best_version.version:
                    # Tag stage = Production
                    client.set_model_version_tag(
                        name=registered_model_name, version=v.version, key="stage", value="Production"
                    )
                else:
                    # Tag stage = Archived
                    client.set_model_version_tag(
                        name=registered_model_name, version=v.version, key="stage", value="Archived"
                    )

    # ============ DAG Pipeline Downstream =================
    paths_dict = download_csvs()
    data_key = load_data(paths_dict)
    model_key = train_model(data_key)
    results_dict = evaluate_model(model_key, data_key)
    mlflow_run_id = log_to_mlflow(results_dict)
    promote_best_model(mlflow_run_id)


# Expose this pipeline in Airflow
dag = emnist_pipeline()
