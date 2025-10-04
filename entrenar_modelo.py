# entrenar_modelo.py
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np

print("📦 Cargando datos...")
df = pd.read_csv('dataset/train.csv')
X_train = df.iloc[:, 1:].values / 255.0
y_train = df.iloc[:, 0].values

print("🧠 Creando red neuronal...")
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

print("🏋️ Entrenando... (esto tomará unos minutos)")
modelo.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.1, verbose=1)

modelo.save('modelo_digitos.h5')
print("✅ Modelo guardado como 'modelo_digitos.h5'")