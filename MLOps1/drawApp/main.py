from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import numpy as np
import cv2
import base64
import io
import requests
from PIL import Image
import uvicorn
import os

MODEL_URL = os.environ.get("MODEL_URL")

app = FastAPI()

# Montamos carpeta 'static' para archivos estáticos (js, css)
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

# Cargar mapping para decodificar la respuesta del modelo
mapping_file = "mapping.txt"
idx_to_char = {}

with open(mapping_file, "r") as f:
    for line in f:
        idx, code = line.strip().split()
        idx_to_char[int(idx)] = chr(int(code))  # convertimos Unicode a carácter


# Página principal con el canvas
@app.get("/", response_class=HTMLResponse)
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Endpoint que recibe la imagen del canvas (base64)
@app.post("/predict")
async def predict(data: dict):
    try:
        # Extraemos base64
        img_data = data.get("image")
        header, base64_data = img_data.split(",", 1)
        
        # Decodificamos la imagen
        img_bytes = base64.b64decode(base64_data)
        img = Image.open(io.BytesIO(img_bytes)).convert("L")  # escala grises

        # Procesamos para modelo EMNIST: resize 28x28, invertimos blanco-negro
        img = img.resize((28, 28))
        img_np = np.array(img)
        img_np = 255 - img_np  # invertir colores: fondo negro, trazo blanco
        img_np = img_np / 255.0  # normalizar

        # Convertimos a lista para enviar JSON
        input_data = img_np.flatten().tolist()
        payload = {
            "instances": [input_data]  # Lista de instancias, cada una es un array
        }

        # Enviamos al modelo expuesto (suponiendo API REST que espera JSON { "input": [...] })
        response = requests.post(MODEL_URL, json=payload)

        if response.status_code == 200:
            pred = response.json().get("predictions")[0]
            pred_char = idx_to_char.get(pred, "?")
            print(f"El valor leído es {response.json()}")
            print(f"El valor decodificado es {pred_char}")
        else:
            pred_char = "Error en modelo"

        return JSONResponse({"prediction": pred_char})

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)