const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const clearBtn = document.getElementById("clear-btn");
const predictionDiv = document.getElementById("prediction");
const previewEl = document.getElementById("preview");

let drawing = false;
let lastX = 0;
let lastY = 0;

// ===== Configuración de dibujo =====
ctx.lineWidth = 20;
ctx.lineCap = "round";
ctx.strokeStyle = "white";
ctx.fillStyle = "black";

// Fondo negro inicial
ctx.fillRect(0, 0, canvas.width, canvas.height);

// ===== Helpers =====
function getCoords(e) {
  if (e.touches && e.touches.length) {
    const rect = canvas.getBoundingClientRect();
    return [e.touches[0].clientX - rect.left, e.touches[0].clientY - rect.top];
  }
  return [e.offsetX, e.offsetY];
}

function startPosition(e) {
  drawing = true;
  [lastX, lastY] = getCoords(e);
}

function endPosition() {
  drawing = false;
}

function draw(e) {
  if (!drawing) return;
  const [x, y] = getCoords(e);
  ctx.beginPath();
  ctx.moveTo(lastX, lastY);
  ctx.lineTo(x, y);
  ctx.stroke();
  [lastX, lastY] = [x, y];
}

// ===== Eventos de mouse =====
canvas.addEventListener("mousedown", startPosition);
canvas.addEventListener("mouseup", endPosition);
canvas.addEventListener("mouseout", endPosition);
canvas.addEventListener("mousemove", draw);

// ===== Eventos touch =====
canvas.addEventListener("touchstart", (e) => {
  e.preventDefault();
  startPosition(e);
});
canvas.addEventListener("touchend", (e) => {
  e.preventDefault();
  endPosition();
});
canvas.addEventListener("touchcancel", (e) => {
  e.preventDefault();
  endPosition();
});
canvas.addEventListener("touchmove", (e) => {
  e.preventDefault();
  draw(e);
});

// ===== Limpiar =====
clearBtn.addEventListener("click", () => {
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  predictionDiv.textContent = "Predicción: -";
  if (previewEl) previewEl.removeAttribute("src");
});

// ===== Polling de predicción =====
const POLL_MS = 1000;

setInterval(() => {
  try {
    const dataUrl = canvas.toDataURL("image/png");
    fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: dataUrl }),
    })
      .then((resp) => resp.json())
      .then((data) => {
        if (typeof data.prediction !== "undefined") {
          predictionDiv.textContent = `Predicción: ${data.prediction}`;
        }
        if (data.preview && previewEl) {
          previewEl.src = data.preview; // imagen 28x28 (escalada a 280) que entra al modelo
        }
        if (data.error) {
          predictionDiv.textContent = `Error: ${data.error}`;
        }
      })
      .catch((err) => {
        predictionDiv.textContent = `Error: ${err}`;
      });
  } catch (e) {
    predictionDiv.textContent = `Error: ${e}`;
  }
}, POLL_MS);
