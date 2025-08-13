const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const clearBtn = document.getElementById("clear-btn");
const predictionDiv = document.getElementById("prediction");

let drawing = false;
let lastX = 0;
let lastY = 0;

// Configuración inicial
ctx.lineWidth = 15;
ctx.lineCap = "round";
ctx.strokeStyle = "white";
ctx.fillStyle = "black";

// Fondo negro
ctx.fillRect(0, 0, canvas.width, canvas.height);

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

function getCoords(e) {
  if (e.touches) {
    const rect = canvas.getBoundingClientRect();
    return [
      e.touches[0].clientX - rect.left,
      e.touches[0].clientY - rect.top,
    ];
  } else {
    return [e.offsetX, e.offsetY];
  }
}

clearBtn.addEventListener("click", () => {
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  predictionDiv.textContent = "Predicción: -";
});

// Polling cada 1 segundo
setInterval(() => {
  const dataUrl = canvas.toDataURL("image/png");
  fetch("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image: dataUrl }),
  })
    .then((resp) => resp.json())
    .then((data) => {
      if (data.prediction !== undefined) {
        predictionDiv.textContent = `Predicción: ${data.prediction}`;
      }
      if (data.error) {
        predictionDiv.textContent = `Error: ${data.error}`;
      }
    })
    .catch((err) => {
      predictionDiv.textContent = `Error: ${err}`;
    });
}, 1000);

canvas.addEventListener("mousedown", startPosition);
canvas.addEventListener("mouseup", endPosition);
canvas.addEventListener("mouseout", endPosition);
canvas.addEventListener("mousemove", draw);

// Para pantallas táctiles
canvas.addEventListener("touchstart", startPosition);
canvas.addEventListener("touchend", endPosition);
canvas.addEventListener("touchcancel", endPosition);
canvas.addEventListener("touchmove", draw);
