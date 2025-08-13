from mlflow.tracking import MlflowClient
import subprocess
import os

client = MlflowClient()
registered_model_name = os.getenv("REGISTERED_MODEL_NAME", "EMNIST-Model")
port = os.getenv("MLFLOW_SERVE_MODEL_PORT", 1234)

# Obtener todas las versiones del modelo
versions = client.search_model_versions(f"name='{registered_model_name}'")

# Filtrar por tag stage=Production
production_versions = []
for v in versions:
    mv = client.get_model_version(registered_model_name, v.version)
    if mv.tags.get("stage") == "Production":
        production_versions.append(v)

if not production_versions:
    raise Exception("No hay versiones en producción")

# Servir modelo en producción
prod_version = max(production_versions, key=lambda x: int(x.version))
model_uri = f"models:/{registered_model_name}/{prod_version.version}"

print(f"Sirviendo modelo {model_uri}")

subprocess.run([
    "mlflow", "models", "serve",
    "-m", model_uri,
    "-p", port,
    "--host", "0.0.0.0",
    "--no-conda"
])
