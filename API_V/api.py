from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from textPreprocessor import TextPreprocessor

# Cargar el pipeline entrenado
pipeline = joblib.load('./pipeline.joblib')

# Crear la app Flask
app = Flask(__name__)

# Página principal con el formulario para la interfaz
@app.route('/')
def home():
    return render_template('index.html')

# Endpoint 1: Predicción
@app.route('/predict_ui', methods=['POST'])
def predict_ui():
    try:
        # Recibir los textos ingresados en la interfaz
        text = request.form['text']
        textos = [t.strip() for t in text.split(',')]  # Convertir el texto en una lista

        # Crear un DataFrame con los textos
        df = pd.DataFrame({"Textos_espanol": textos})

        # Realizar predicciones
        predictions = pipeline.predict(df['Textos_espanol'])

        # Renderizar la página con las predicciones
        return render_template('result.html', predictions=predictions)
    except KeyError as e:
        return f"Error: no se encontró la clave {e.args[0]} en el formulario", 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500



# Endpoint 2: Reentrenamiento del modelo
@app.route('/retrain', methods=['POST'])
def retrain():
    # Cargar los datos enviados en la solicitud
    data = request.get_json()  # Recibir los datos en formato JSON
    df_nuevos_datos = pd.DataFrame(data)  # Convertir los datos a DataFrame

    # Asegurarse de que las columnas requeridas estén en los datos
    if 'Textos_espanol' not in df_nuevos_datos.columns or 'sdg' not in df_nuevos_datos.columns:
        return jsonify({"error": "Faltan las columnas 'Textos_espanol' y/o 'sdg'"}), 400

    try:
        # Intentar cargar los datos existentes almacenados en el archivo Excel
        datos_existentes = pd.read_excel('API_V/ODScat_345.xlsx')
    except FileNotFoundError:
        # Si no existe el archivo, crear uno nuevo con los datos recibidos
        datos_existentes = pd.DataFrame(columns=['Textos_espanol', 'sdg'])

    # Combinar los datos nuevos con los existentes
    datos_completos = pd.concat([datos_existentes, df_nuevos_datos], ignore_index=True)

    # Guardar los datos combinados en el archivo Excel (sobrescribir el archivo con los datos actualizados)
    datos_completos.to_excel('API_V/ODScat_345.xlsx', index=False)

    # Separar las características y las etiquetas
    X_data = datos_completos['Textos_espanol']
    y_data = datos_completos['sdg']

    # Reentrenar el modelo con el conjunto completo de datos (antiguos + nuevos)
    pipeline.fit(X_data, y_data)

    # Evaluar el modelo con el conjunto completo de datos
    y_pred = pipeline.predict(X_data)
    rmse = mean_squared_error(y_data, y_pred, squared=False)
    mae = mean_absolute_error(y_data, y_pred)
    r2 = r2_score(y_data, y_pred)

    # Guardar el modelo reentrenado
    joblib.dump(pipeline, 'pipeline.joblib')

    # Devolver las métricas de desempeño del reentrenamiento
    return jsonify({"rmse": rmse, "mae": mae, "r2": r2})

@app.route('/predict_xlsx', methods=['POST'])
def predict_xlsx():
    if 'file' not in request.files:
        return jsonify({"error": "No se ha subido ningún archivo"}), 400
    
    file = request.files['file']

    # Verificar que el archivo sea XLSX
    if file.filename == '' or not file.filename.endswith('.xlsx'):
        return jsonify({"error": "Por favor sube un archivo XLSX válido"}), 400

    try:
        # Leer el archivo XLSX
        df = pd.read_excel(file)

        # Verificar que la columna 'Textos_espanol' exista
        if 'Textos_espanol' not in df.columns:
            return jsonify({"error": "El archivo XLSX debe contener una columna 'Textos_espanol'"}), 400
        
        # Obtener los textos a clasificar
        textos = df['Textos_espanol'].tolist()

        # Crear un DataFrame con los textos
        df_textos = pd.DataFrame({"Textos_espanol": textos})

        # Realizar predicciones
        predictions = pipeline.predict(df_textos['Textos_espanol']).tolist()  # Convertir a lista

        # Renderizar la nueva plantilla con textos y sus predicciones
        return render_template('result_archivo.html', textos=textos, predictions=predictions)
    except Exception as e:
        return jsonify({"error": str(e)}), 500




# Ejecutar la app
if __name__ == '__main__':
    app.run(debug=True)
