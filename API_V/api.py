from flask import Flask, request, jsonify
import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from textPreprocessor import TextPreprocessor

# Cargar el pipeline entrenado
pipeline = joblib.load('API_V/pipeline.joblib')

# Crear la app Flask
app = Flask(__name__)

# Endpoint 1: Predicción
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Recibir los datos en formato JSON
    df = pd.DataFrame(data)  # Convertir los datos a DataFrame

    # Asegurarse de que la estructura sea correcta
    if 'Textos_espanol' not in df.columns:
        return jsonify({"error": "La columna 'Textos_espanol' no está en los datos"}), 400

    try:
        # Realizar predicciones
        predictions = pipeline.predict(df['Textos_espanol'])
        # Devolver las predicciones en el mismo orden
        return jsonify(predictions.tolist())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Endpoint 2: Reentrenamiento del modelo
@app.route('/retrain', methods=['POST'])
def retrain():
    data = request.get_json()  # Recibir los datos en formato JSON
    df = pd.DataFrame(data)  # Convertir los datos a DataFrame

    # Asegurarse de que las columnas requeridas estén en los datos
    if 'Textos_espanol' not in df.columns or 'sdg' not in df.columns:
        return jsonify({"error": "Faltan las columnas 'Textos_espanol' y/o 'sdg'"}), 400

    # Separar características y etiquetas
    X_data = df['Textos_espanol']
    y_data = df['sdg']

    # Reentrenar el modelo
    pipeline.fit(X_data, y_data)

    # Evaluar el modelo
    y_pred = pipeline.predict(X_data)
    rmse = mean_squared_error(y_data, y_pred, squared=False)
    mae = mean_absolute_error(y_data, y_pred)
    r2 = r2_score(y_data, y_pred)

    # Guardar el modelo reentrenado
    joblib.dump(pipeline, 'pipeline.joblib')

    # Devolver las métricas de desempeño
    return jsonify({"rmse": rmse, "mae": mae, "r2": r2})

# Ejecutar la app
if __name__ == '__main__':
    app.run(debug=True)
