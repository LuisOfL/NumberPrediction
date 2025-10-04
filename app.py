from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from tensorflow import keras
from tensorflow.keras import layers
import cv2
import numpy as np
import pandas as pd
import base64
import io
from PIL import Image
import uvicorn

app = FastAPI()

# Entrenar o cargar red neuronal
def cargar_modelo():
    try:
        print(" Cargando red neuronal...")
        modelo = keras.models.load_model('modelo_digitos.h5')
        print("✅ Red neuronal cargada!")
        return modelo
    except:
        print(" Entrenando nueva red neuronal...")
        df = pd.read_csv('dataset/train.csv')
        X_train = df.iloc[:, 1:].values / 255.0
        y_train = df.iloc[:, 0].values
        
        modelo = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(784,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])
        
        modelo.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy', 
            metrics=['accuracy']
        )
        
        modelo.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.1, verbose=1)
        modelo.save('modelo_digitos.h5')
        print(" Red neuronal entrenada y guardada!")
        return modelo

modelo = cargar_modelo()

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
        print(f"Error: {e}")
        return None

def predecir_digito(imagen):
    imagen = cv2.resize(imagen, (28, 28))
    imagen_flat = imagen.flatten() / 255.0
    predicciones = modelo.predict(np.array([imagen_flat]), verbose=0)
    return int(np.argmax(predicciones[0]))

HTML_PAGINA = """
<!DOCTYPE html>
<html>
<head>
    <title>Reconocedor de Dígitos</title>
    <style>
        body { font-family: Arial; text-align: center; padding: 20px; background: #1a1a1a; color: white; }
        .container { max-width: 600px; margin: 0 auto; background: #2d2d2d; padding: 20px; border-radius: 10px; }
        canvas { border: 2px solid #555; background: black; cursor: crosshair; margin: 10px 0; touch-action: none; }
        button { padding: 10px 20px; margin: 5px; font-size: 16px; cursor: pointer; background: #007bff; color: white; border: none; border-radius: 5px; }
        button:hover { background: #0056b3; }
        .resultado { font-size: 72px; font-weight: bold; color: #28a745; margin: 20px 0; }
    </style>
</head>
<body>
    <div class="container">
        <h1> Dibuja un Dígito (0-9)</h1>
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
                const resultado = await respuesta.json();
                document.getElementById('prediccion').textContent = resultado.prediccion;
            } catch {
                document.getElementById('prediccion').textContent = '!';
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

@app.post("/predecir")
async def predecir(data: dict):
    try:
        imagen = procesar_imagen(data.get("imagen", ""))
        if imagen is None:
            raise HTTPException(status_code=400, detail="Error procesando imagen")
        return {"prediccion": predecir_digito(imagen)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("🚀 Servidor con RED NEURONAL en http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)