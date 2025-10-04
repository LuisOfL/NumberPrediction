from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from tensorflow import keras
import cv2
import numpy as np
import pandas as pd
import base64
import io
from PIL import Image
import os

app = FastAPI()

# Configurar CORS para Render
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar modelo (NO entrenar en Render)
try:
    print("🧠 Cargando modelo pre-entrenado...")
    modelo = keras.models.load_model('modelo_digitos.h5')
    print("✅ Modelo cargado exitosamente!")
except Exception as e:
    print(f"❌ Error cargando modelo: {e}")
    # Crear modelo dummy para evitar crash (solo para desarrollo)
    from tensorflow.keras import layers
    modelo = keras.Sequential([
        layers.Dense(10, activation='softmax', input_shape=(784,))
    ])
    modelo.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

def procesar_imagen(imagen_base64):
    try:
        if ',' in imagen_base64:
            imagen_base64 = imagen_base64.split(',')[1]
        imagen_bytes = base64.b64decode(imagen_base64)
        imagen_pil = Image.open(io.BytesIO(imagen_bytes))
        imagen_cv = np.array(imagen_pil)
        if len(imagen_cv.shape) == 3:
            imagen_cv = cv2.cvtColor(imagen_cv, cv2.COLOR_RGB2GRAY)
        return imagen_cv
    except Exception as e:
        print(f"Error procesando imagen: {e}")
        return None

def predecir_digito(imagen):
    try:
        imagen = cv2.resize(imagen, (28, 28))
        imagen_flat = imagen.flatten() / 255.0
        predicciones = modelo.predict(np.array([imagen_flat]), verbose=0)
        return int(np.argmax(predicciones[0]))
    except Exception as e:
        print(f"Error en predicción: {e}")
        return 0  # Valor por defecto si hay error

HTML_PAGINA = """
<!DOCTYPE html>
<html>
<head>
    <title>Reconocedor de Dígitos</title>
    <style>
        body { font-family: Arial; text-align: center; padding: 20px; background: #1a1a1a; color: white; margin: 0; }
        .container { max-width: 600px; margin: 0 auto; background: #2d2d2d; padding: 20px; border-radius: 10px; }
        canvas { border: 2px solid #555; background: black; cursor: crosshair; margin: 10px 0; touch-action: none; }
        button { padding: 10px 20px; margin: 5px; font-size: 16px; cursor: pointer; background: #007bff; color: white; border: none; border-radius: 5px; }
        button:hover { background: #0056b3; }
        .resultado { font-size: 72px; font-weight: bold; color: #28a745; margin: 20px 0; }
        .status { background: #3a3a3a; padding: 10px; border-radius: 5px; margin: 10px 0; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎨 Reconocedor de Dígitos</h1>
        <div class="status" id="status">✅ Servidor funcionando</div>
        <canvas id="canvas" width="280" height="280"></canvas>
        <div>
            <button onclick="predecir()">🔍 Predecir</button>
            <button onclick="limpiar()">🗑️ Limpiar</button>
        </div>
        <div class="resultado" id="prediccion">?</div>
    </div>

    <script>
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        let dibujando = false;
        let ultimoX = 0, ultimoY = 0;
        
        function inicializarCanvas() {
            ctx.fillStyle = 'black';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            ctx.strokeStyle = 'white';
            ctx.lineWidth = 12;
            ctx.lineJoin = 'round';
            ctx.lineCap = 'round';
            document.getElementById('prediccion').textContent = '?';
        }
        
        canvas.addEventListener('mousedown', (e) => {
            dibujando = true;
            [ultimoX, ultimoY] = [e.offsetX, e.offsetY];
        });
        
        canvas.addEventListener('mousemove', (e) => {
            if (!dibujando) return;
            ctx.beginPath();
            ctx.moveTo(ultimoX, ultimoY);
            ctx.lineTo(e.offsetX, e.offsetY);
            ctx.stroke();
            [ultimoX, ultimoY] = [e.offsetX, e.offsetY];
        });
        
        canvas.addEventListener('mouseup', () => dibujando = false);
        canvas.addEventListener('mouseout', () => dibujando = false);
        
        function limpiar() { inicializarCanvas(); }
        
        async function predecir() {
            document.getElementById('prediccion').textContent = '...';
            try {
                const respuesta = await fetch('/predecir', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({imagen: canvas.toDataURL('image/png')})
                });
                
                if (!respuesta.ok) {
                    throw new Error('Error del servidor');
                }
                
                const resultado = await respuesta.json();
                document.getElementById('prediccion').textContent = resultado.prediccion;
            } catch (error) {
                document.getElementById('prediccion').textContent = '!';
                console.error('Error:', error);
            }
        }
        
        inicializarCanvas();
    </script>
</body>
</html>
"""

@app.get("/")
async def pagina_principal():
    return HTMLResponse(HTML_PAGINA)

@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "Servidor funcionando"}

@app.post("/predecir")
async def predecir(data: dict):
    try:
        imagen = procesar_imagen(data.get("imagen", ""))
        if imagen is None:
            raise HTTPException(status_code=400, detail="Error procesando imagen")
        
        prediccion = predecir_digito(imagen)
        return {"prediccion": prediccion}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# No uses __main__ en Render - ellos manejan el servidor
