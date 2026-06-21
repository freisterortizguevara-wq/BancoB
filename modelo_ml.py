# modelo_ml.py - Clasificador de intenciones BBVA (versión final corregida)
import pandas as pd
import re
import unicodedata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib
import os

def normalizar_texto(texto):
    texto = texto.lower()
    texto = unicodedata.normalize('NFKD', texto).encode('ASCII', 'ignore').decode('utf-8')
    texto = re.sub(r'[^a-z\s]', '', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

# Lista manual de stopwords en español
stopwords_es = ['el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas', 'y', 'o', 'de', 'del', 'a', 'ante', 'bajo', 'cabe', 'con', 'contra', 'para', 'por', 'segun', 'sin', 'so', 'sobre', 'tras', 'que', 'cual', 'quien', 'donde', 'cuando', 'como', 'muy', 'mas', 'menos', 'si', 'no', 'ya', 'lo', 'le', 'les', 'me', 'te', 'se', 'nos', 'os', 'mi', 'tu', 'su', 'nuestro', 'vuestro', 'este', 'ese', 'aquel', 'esto', 'eso', 'aquello', 'ser', 'estar', 'tener', 'hacer', 'poder', 'decir', 'ir', 'ver', 'dar', 'saber', 'querer', 'llegar', 'pasar', 'deber', 'poner', 'parecer', 'quedar', 'creer', 'hablar', 'llevar', 'dejar', 'seguir', 'encontrar', 'llamar']

# Datos de entrenamiento (balanceados)
textos = []
intenciones = []

# Saldo (20 ejemplos)
saldo = [
    "cual es mi saldo", "quiero saber mi saldo", "cuanto tengo en mi cuenta",
    "dime mi saldo disponible", "ver saldo", "consultar saldo", "saldo actual",
    "que saldo tengo", "mostrar mi saldo", "necesito mi saldo", "saldo de mi cuenta",
    "cuanto dinero tengo", "ver mi saldo", "consulta de saldo", "saldo bancario",
    "mi saldo", "sado", "salddo", "consultar mi dinero", "cuanto tengo"
]
textos.extend(saldo)
intenciones.extend(["saldo"]*len(saldo))

# Apelación (20 ejemplos)
apelacion = [
    "quiero apelar un cargo", "reversion de pago", "no reconozco este cargo",
    "apelar transaccion", "me cobraron dos veces", "cargo no reconocido",
    "reversion por favor", "quiero disputar un pago", "apelar cargo",
    "reclamar transaccion", "cargo indebido", "error en mi estado de cuenta",
    "me hicieron un cobro que no es mio", "quiero reportar un cargo",
    "desconozco esta transaccion", "apelacion", "reclamo", "queja",
    "cobro duplicado", "cobro incorrecto"
]
textos.extend(apelacion)
intenciones.extend(["apelacion"]*len(apelacion))

# Préstamo (20 ejemplos)
prestamo = [
    "quiero un prestamo", "solicitar credito", "prestamo personal",
    "necesito dinero prestado", "credito rapido", "solicitar prestamo",
    "cuanto me prestan", "requisitos para prestamo", "tasas de prestamo",
    "quiero pedir un credito", "prestamo bancario", "solicitud de credito",
    "me interesa un prestamo", "financiamiento", "credito de nomina",
    "prestamo", "credito", "necesito dinero", "me prestan", "solicitar un credito"
]
textos.extend(prestamo)
intenciones.extend(["prestamo"]*len(prestamo))

# General (20 ejemplos)
general = [
    "gracias", "hola buenos dias", "que horario tienen", "donde esta tu oficina",
    "telefono de contacto", "correo electronico", "gracias por la ayuda", "adios",
    "buenas tardes", "como estas", "que tal", "buen dia", "informacion", "ayuda",
    "soporte", "hola", "buenas", "saludos", "ok", "vale"
]
textos.extend(general)
intenciones.extend(["general"]*len(general))

# Agregar respuestas "sí" y "no" como general (para que no clasifique mal)
textos.extend(["si", "sí", "no", "tal vez", "adelante", "continuar"])
intenciones.extend(["general"]*6)

df = pd.DataFrame({"texto": textos, "intencion": intenciones})
df["texto_norm"] = df["texto"].apply(normalizar_texto)

# Pipeline sin stop_words conflictivo, usamos nuestra lista manual
pipeline = Pipeline([
    ("vectorizer", TfidfVectorizer(ngram_range=(1, 3), max_features=3000, stop_words=stopwords_es)),
    ("clasificador", MultinomialNB(alpha=0.3))
])

pipeline.fit(df["texto_norm"], df["intencion"])

os.makedirs("modelo", exist_ok=True)
joblib.dump(pipeline, "modelo/clasificador_intenciones.pkl")
print(f"✅ Modelo entrenado con {len(df)} ejemplos. Guardado en 'modelo/clasificador_intenciones.pkl'.")

# Prueba
if __name__ == "__main__":
    modelo = joblib.load("modelo/clasificador_intenciones.pkl")
    pruebas = ["cual es mi sado", "reclamo", "prestamo", "gracias", "si", "no", "quiero continuar"]
    for p in pruebas:
        pred = modelo.predict([normalizar_texto(p)])[0]
        print(f"'{p}' -> {pred}")