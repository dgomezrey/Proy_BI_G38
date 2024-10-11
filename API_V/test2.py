import requests

# URL del endpoint de reentrenamiento
url = 'http://127.0.0.1:5000/retrain'

# Datos de prueba (asegúrate de que coincidan con las columnas esperadas)
data = {
    "Textos_espanol": ["Desnutricion en niños", "La educación debe ser accesible para todos.", "La igualdad de género es clave para un futuro sostenible."],
    "sdg": [3, 4, 5]
}




# Enviar la solicitud POST
response = requests.post(url, json=data)

# Verificar el estado de la respuesta y los resultados
if response.status_code == 200:
    print("Reentrenamiento exitoso!")
    print(response.json())  # Mostrar las métricas de evaluación
else:
    print(f"Error {response.status_code}: {response.text}")
