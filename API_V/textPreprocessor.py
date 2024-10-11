
# Importación de librerías
import pandas as pd
import numpy as np
import re
import unicodedata
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import nltk
from langdetect import detect
from googletrans import Translator
from num2words import num2words
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer, WordNetLemmatizer
import ftfy
import joblib
from sklearn.base import BaseEstimator, TransformerMixin

# Descargar recursos de nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# Paso 1: Función para detectar idioma y traducir si es necesario
def translate_text(text):
    translator = Translator()
    detected_lang = detect(text)
    if detected_lang != 'es':
        try:
            translation = translator.translate(text, src=detected_lang, dest='es')
            return translation.text
        except Exception:
            return text
    return text

# Paso 2: Función para limpieza de texto
def clean_text(text):
    # Convertir a minúsculas
    text = text.lower()
    # Eliminar caracteres no ASCII
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    # Eliminar signos de puntuación
    text = re.sub(r'[^\w\s]', '', text)
    # Eliminar stopwords
    stop_words = set(stopwords.words('spanish'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Paso 3: Corrección de errores de codificación
def fix_encoding(text):
    return ftfy.fix_text(text)

# Paso 4: Conversión de números a palabras
def convert_numbers(text):
    return ' '.join([num2words(word, lang='es') if word.isdigit() else word for word in text.split()])

# Paso 5: Tokenización de palabras
def tokenize_text(text):
    return word_tokenize(text)

# Paso 6: Aplicar stemming y lematización
def stem_and_lemmatize(tokens):
    stemmer = SnowballStemmer('spanish')
    lemmatizer = WordNetLemmatizer()
    stems = [stemmer.stem(token) for token in tokens]
    lemmas = [lemmatizer.lemmatize(token, pos='v') for token in tokens]
    # Concatenar stems y lemmas para mayor cobertura
    return stems + lemmas

# Paso 7: Preprocesamiento completo del texto
def preprocess_text(text):
    # Traducción de texto si es necesario
    text = translate_text(text)
    # Limpieza básica
    text = clean_text(text)
    # Corrección de codificación
    text = fix_encoding(text)
    # Conversión de números a palabras
    text = convert_numbers(text)
    # Tokenización
    tokens = tokenize_text(text)
    # Stemming y lematización
    processed_tokens = stem_and_lemmatize(tokens)
    # Unir tokens procesados en una cadena
    return ' '.join(processed_tokens)

# Clase personalizada para el preprocesamiento de texto
class TextPreprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Aplicar el preprocesamiento personalizado a cada texto
        return X.apply(preprocess_text)





