from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from textPreprocessor import TextPreprocessor

# Cargar el pipeline entrenado
pipeline = joblib.load('./pipeline.joblib')

# Crear la app Flask
app = Flask(__name__)

# Página principal con el formulario para la interfaz
@app.route('/')
def home():
    return render_template('index.html')

# Endpoint 1: Predicción con probabilidades
@app.route('/predict_ui', methods=['POST'])
def predict_ui():
    try:
        # Recibir los textos ingresados en la interfaz
        text = request.form['text']
        textos = [t.strip() for t in text.split(',')]  # Convertir el texto en una lista

        # Crear un DataFrame con los textos
        df = pd.DataFrame({"Textos_espanol": textos})

        # Realizar predicciones usando la columna 'Textos_espanol'
        predictions = pipeline.predict(df['Textos_espanol'])

        # Asegurarse de que el modelo soporte 'predict_proba'
        if hasattr(pipeline, 'predict_proba'):
            probabilities = pipeline.predict_proba(df['Textos_espanol'])
        else:
            return jsonify({"error": "El modelo no soporta predict_proba"}), 400

        # Asignar los SDG correspondientes a las probabilidades
        sdg_labels = [3, 4, 5]
        results = []
        for i in range(len(predictions)):
            prob_with_labels = [{"sdg": sdg_labels[j], "probability": probabilities[i][j]} for j in range(len(sdg_labels))]
            results.append({
                "text": textos[i],  # Agregar el texto original
                "prediction": predictions[i],
                "probabilities": prob_with_labels  # Probabilidades con los SDG asociados
            })

        # Renderizar la página con las predicciones y probabilidades
        return render_template('result.html', predictions=results)

    except KeyError as e:
        return f"Error: no se encontró la clave {e.args[0]} en el formulario", 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Endpoint 1: Predicción con probabilidades para archivos
@app.route('/predict_xlsx', methods=['POST'])
def predict_xlsx():
    try:
        # Obtener el archivo subido
        file = request.files['file']
        
        # Leer el archivo (puede ser CSV o XLSX)
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.filename.endswith('.xlsx'):
            df = pd.read_excel(file)
        else:
            return "Formato no soportado", 400

        # Verificar que la columna 'Textos_espanol' exista
        if 'Textos_espanol' not in df.columns:
            return jsonify({"error": "El archivo debe contener una columna 'Textos_espanol'"}), 400

        # Asegurarse de que 'Textos_espanol' sea un DataFrame antes de usar el pipeline
        df_textos = pd.DataFrame(df['Textos_espanol'])  # Crear un DataFrame si es necesario

        # Realizar predicciones y calcular probabilidades
        predictions = pipeline.predict(df_textos['Textos_espanol'])

        if hasattr(pipeline, 'predict_proba'):
            probabilities = pipeline.predict_proba(df_textos['Textos_espanol'])
        else:
            return jsonify({"error": "El modelo no soporta predict_proba"}), 400

        # Asignar los SDG correspondientes a las probabilidades
        sdg_labels = [3, 4, 5]
        results = []
        for i in range(len(predictions)):
            prob_with_labels = [{"sdg": sdg_labels[j], "probability": probabilities[i][j]} for j in range(len(sdg_labels))]
            results.append({
                "text": df['Textos_espanol'].iloc[i],  # El texto original
                "prediction": predictions[i],
                "probabilities": prob_with_labels
            })

        # Renderizar la página result_archivo.html con las predicciones y probabilidades
        return render_template('result_archivo.html', predictions=results, textos=df['Textos_espanol'])

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Endpoint 2: Reentrenamiento del modelo
@app.route('/retrain', methods=['POST'])
def retrain():
    try:
        # Obtener el archivo subido
        file = request.files['file']
        
        # Leer el archivo (puede ser CSV o XLSX)
        if file.filename.endswith('.csv'):
            df_nuevos_datos = pd.read_csv(file)
        elif file.filename.endswith('.xlsx'):
            df_nuevos_datos = pd.read_excel(file)
        else:
            return "Formato no soportado", 400

        # Verificar que el archivo contiene las columnas adecuadas
        if 'Textos_espanol' not in df_nuevos_datos.columns or 'sdg' not in df_nuevos_datos.columns:
            return jsonify({"error": "Faltan las columnas 'Textos_espanol' y/o 'sdg'"}), 400

        # Intentar cargar los datos existentes almacenados en el archivo Excel
        try:
            datos_existentes = pd.read_excel('API_V/ODScat_345.xlsx')
        except FileNotFoundError:
            # Si no existe el archivo, crear uno nuevo con los datos recibidos
            datos_existentes = pd.DataFrame(columns=['Textos_espanol', 'sdg'])

        # Combinar los datos nuevos con los existentes
        datos_completos = pd.concat([datos_existentes, df_nuevos_datos], ignore_index=True)

        # Guardar los datos combinados en el archivo Excel
        datos_completos.to_excel('API_V/ODScat_345.xlsx', index=False)

        # Separar las características y etiquetas
        X_data = datos_completos['Textos_espanol']
        y_data = datos_completos['sdg']

        # Reentrenar el modelo
        pipeline.fit(X_data, y_data)

        # Evaluar el modelo
        y_pred = pipeline.predict(X_data)
        accuracy = accuracy_score(y_data, y_pred)
        precision = precision_score(y_data, y_pred, average='weighted')
        recall = recall_score(y_data, y_pred, average='weighted')
        f1 = f1_score(y_data, y_pred, average='weighted')

        # Guardar el modelo reentrenado
        joblib.dump(pipeline, 'pipeline.joblib')

        # Renderizar la plantilla con las métricas
        return render_template('result_retrain.html', accuracy=accuracy, precision=precision, recall=recall, f1_score=f1)

    except Exception as e:
        return jsonify({"error": str(e)}), 500





# Ejecutar la app
if __name__ == '__main__':
    app.run(debug=True)
