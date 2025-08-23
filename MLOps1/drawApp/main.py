from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import numpy as np
import cv2
import base64
import io
import requests
from PIL import Image, ImageFilter
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

# Decodificar la clase que detecta el modelo al caracter original detectado
with open(mapping_file, "r") as f:
    for line in f:
        idx, code = line.strip().split()
        idx_to_char[int(idx)] = chr(int(code))  # convertimos Unicode a carácter


# Proprocesar la imagen leída del canvas para asemejarse a las imágenes de entrenamiento del modelo
def transform_image_to_emnist_format(img_data):
    # 1) Decodificar base64 → PIL (L)
    header, base64_data = img_data.split(",", 1)
    img_bytes = base64.b64decode(base64_data)
    img = Image.open(io.BytesIO(img_bytes)).convert("L")

    # 2) Suavizado leve (opcional)
    img = img.filter(ImageFilter.GaussianBlur(radius=0.6))
    img_np = np.array(img, dtype=np.uint8)

    # 3) Asegurar colores: fondo negro (0), trazo blanco (255)
    #    Si las esquinas son claras, invertimos.
    if np.mean([img_np[0, 0], img_np[0, -1], img_np[-1, 0], img_np[-1, -1]]) > 127:
        img_np = 255 - img_np

    # 4) Binarizar (umbral simple robusto)
    thresh = int(np.clip(img_np.mean() * 0.9, 60, 180))
    img_np = (img_np > thresh).astype(np.uint8) * 255

    # 5) Recortar bbox y centrar en 28×28 preservando aspecto (lado mayor≈20)
    ys, xs = np.where(img_np > 0)
    if len(xs) == 0 or len(ys) == 0:
        canvas = np.zeros((28, 28), dtype=np.uint8)
    else:
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        crop = img_np[y0 : y1 + 1, x0 : x1 + 1]

        h, w = crop.shape
        scale = 20 / max(h, w)
        new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        crop = Image.fromarray(crop).resize((new_w, new_h), Image.BILINEAR)

        canvas = np.zeros((28, 28), dtype=np.uint8)
        y_off, x_off = (28 - new_h) // 2, (28 - new_w) // 2
        canvas[y_off : y_off + new_h, x_off : x_off + new_w] = np.array(crop)

    # 6) Convención EMNIST: rotar -90° y espejo horizontal
    canvas = np.fliplr(np.rot90(canvas, 3))

    # 7) (Opcional) Normalizar a [0,1] si tu modelo lo espera así:
    # canvas_norm = canvas.astype(np.float32) / 255.0
    # inputs = [canvas_norm.flatten().tolist()]
    # Si tu modelo entrenó con 0..255, envía uint8:
    inputs = [canvas.flatten().tolist()]

    # ---- Crear preview (dataURL) 28x28 escalado a 280px “pixelado” ----
    im28 = Image.fromarray(canvas)  # 28x28
    im_big = im28.resize((280, 280), Image.NEAREST)
    buf = io.BytesIO()
    im_big.save(buf, format="PNG")
    preview_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    preview_dataurl = f"data:image/png;base64,{preview_b64}"

    return {"inputs": inputs}, preview_dataurl


# Página principal con el canvas
@app.get("/", response_class=HTMLResponse)
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# Endpoint que recibe la imagen del canvas (base64), la procesa con el modelo y devuelve la predicción
@app.post("/predict")
async def predict(data: dict):
    try:
        img_data = data.get("image")

        # Preprocesar la imagen
        payload, preview = transform_image_to_emnist_format(img_data=img_data)

        # Llamar al endpoint del modelo para clasificación
        response = requests.post(MODEL_URL, json=payload)
        if response.status_code == 200:
            pred = response.json().get("predictions")[0]
            pred_char = idx_to_char.get(pred, "?")
            print(f"El valor leído es {response.json()}")
            print(f"El valor decodificado es {pred_char}")
        else:
            pred_char = "Error en modelo"

        return JSONResponse({"prediction": pred_char, "preview": preview})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
