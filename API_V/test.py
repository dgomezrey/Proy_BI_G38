import requests

url = 'http://127.0.0.1:5000/predict'
data = {
    "Textos_espanol": ["El cambio climático es un gran reto.", "La educación es un derecho."]
}

response = requests.post(url, json=data)

# Verifica el estado de la respuesta
print(f"Status Code: {response.status_code}")
if response.status_code == 200:
    print(response.json())
else:
    print(f"Error: {response.text}")
