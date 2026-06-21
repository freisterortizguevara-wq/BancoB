# ===================================================================
# MODELO DE CLASIFICACIÓN DE INTENCIONES PARA BBVA
# ===================================================================
# Este programa aprende a reconocer qué quiere hacer un usuario
# según lo que escribe (saldo, apelación, préstamo, o conversación general)
# ===================================================================

# ==================== IMPORTACIONES ====================
# Las importaciones son como "traer herramientas" de una caja de herramientas

import pandas as pd  
# ¿Qué hace? Pandas es una herramienta para manejar tablas de datos
# ¿Para qué? Para crear una tabla con textos y sus intenciones
# Ejemplo: crea una tabla como Excel con columnas "texto" e "intencion"

import re  
# ¿Qué hace? Re = Regular Expressions (Expresiones Regulares)
# ¿Para qué? Para buscar patrones en texto (como números, símbolos, espacios)
# Ejemplo: eliminar todos los números de un texto: re.sub(r'\d+', '', texto)

import unicodedata  
# ¿Qué hace? Maneja caracteres especiales y tildes
# ¿Para qué? Para convertir "camión" en "camion" (quitar tildes)
# Ejemplo: "áéíóú" se convierte en "aeiou"

from sklearn.feature_extraction.text import TfidfVectorizer  
# ¿Qué hace? Convierte texto en números (las computadoras solo entienden números)
# ¿Para qué? Para que el modelo pueda procesar el texto
# Ejemplo: "hola mundo" -> [0.2, 0.5, 0.1, 0.8] (un montón de números)

from sklearn.naive_bayes import MultinomialNB  
# ¿Qué hace? Es un algoritmo de aprendizaje automático para clasificar
# ¿Para qué? Para predecir a qué categoría pertenece un texto
# NB = Naive Bayes (Naive = Ingenuo, Bayes = por el teorema de Bayes)
# Ejemplo: aprende que "saldo" y "tengo" = categoría "saldo"

from sklearn.pipeline import Pipeline  
# ¿Qué hace? Encadena varios procesos en uno solo
# ¿Para qué? Para hacer todo el proceso (vectorizar + clasificar) de una vez
# Ejemplo: texto -> [convertir a números] -> [clasificar] -> resultado

import joblib  
# ¿Qué hace? Guarda y carga modelos entrenados
# ¿Para qué? Para no tener que reentrenar cada vez que usamos el programa
# Ejemplo: guarda el modelo en un archivo .pkl y lo carga después

import os  
# ¿Qué hace? OS = Operating System (Sistema Operativo)
# ¿Para qué? Para crear carpetas, verificar rutas de archivos
# Ejemplo: os.makedirs("carpeta") crea una carpeta llamada "carpeta"


# ==================== FUNCIÓN DE LIMPIEZA DE TEXTO ====================
# Esta función limpia el texto para que el modelo lo entienda mejor

def normalizar_texto(texto):
    """
    LIMPIA EL TEXTO:
    - Convierte a minúsculas
    - Elimina tildes
    - Elimina números y signos de puntuación
    - Elimina espacios extras
    
    EJEMPLO:
    Entrada: "¡Hola! ¿Cómo estás? 123"
    Salida:  "hola como estas"
    """
    
    # PASO 1: Convertir TODO a minúsculas
    texto = texto.lower()
    # "HOLA MUNDO" -> "hola mundo"
    # "Hola" -> "hola"
    
    # PASO 2: Eliminar tildes y caracteres especiales
    # NFKD = Normalization Form KD (descompone caracteres)
    # encode('ASCII', 'ignore') = solo deja caracteres ingleses, ignora el resto
    # decode('utf-8') = vuelve a convertir a texto
    texto = unicodedata.normalize('NFKD', texto).encode('ASCII', 'ignore').decode('utf-8')
    # "camión" -> "camion"
    # "corazón" -> "corazon"
    # "¿Cómo?" -> "Como?" (sin signos aún)
    
    # PASO 3: Eliminar todo lo que NO sea letras minúsculas o espacios
    # [^a-z\s] = cualquier cosa que NO sea (^) letras de la a a la z o espacios
    # re.sub() = reemplaza lo que encuentre con '' (nada)
    texto = re.sub(r'[^a-z\s]', '', texto)
    # "hola123 mundo456" -> "hola mundo"
    # "¿hola? ¡mundo!" -> "hola mundo"
    
    # PASO 4: Eliminar espacios múltiples y espacios al inicio/final
    # \s+ = uno o más espacios
    # strip() = elimina espacios al inicio y final
    texto = re.sub(r'\s+', ' ', texto).strip()
    # "hola    mundo" -> "hola mundo"
    # "  hola mundo  " -> "hola mundo"
    
    return texto  # Devuelve el texto limpio


# ==================== STOPWORDS (PALABRAS QUE SE IGNORAN) ====================
# Las stopwords son palabras tan comunes que no ayudan a clasificar
# Es como cuando buscas en Google: ignoran "el", "la", "y"

stopwords_es = [
    # ARTÍCULOS (el, la, los, las, un, una, unos, unas)
    'el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas',
    
    # CONJUNCIONES (y, o, de, del)
    'y', 'o', 'de', 'del',
    
    # PREPOSICIONES (a, ante, bajo, cabe, con, contra, para, por, según, sin, so, sobre, tras)
    'a', 'ante', 'bajo', 'cabe', 'con', 'contra', 'para', 'por', 'segun', 'sin', 'so', 'sobre', 'tras',
    
    # PRONOMBRES Y CONECTORES (que, cual, quien, donde, cuando, como)
    'que', 'cual', 'quien', 'donde', 'cuando', 'como',
    
    # ADVERBIOS (muy, mas, menos, si, no, ya)
    'muy', 'mas', 'menos', 'si', 'no', 'ya',
    
    # PRONOMBRES PERSONALES (lo, le, les, me, te, se, nos, os, mi, tu, su, nuestro, vuestro)
    'lo', 'le', 'les', 'me', 'te', 'se', 'nos', 'os', 'mi', 'tu', 'su', 'nuestro', 'vuestro',
    
    # DEMOSTRATIVOS (este, ese, aquel, esto, eso, aquello)
    'este', 'ese', 'aquel', 'esto', 'eso', 'aquello',
    
    # VERBOS COMUNES (ser, estar, tener, hacer, poder, decir, ir, ver, dar, saber, querer)
    'ser', 'estar', 'tener', 'hacer', 'poder', 'decir', 'ir', 'ver', 'dar', 'saber', 'querer',
    
    # MÁS VERBOS COMUNES (llegar, pasar, deber, poner, parecer, quedar, creer, hablar, llevar, dejar, seguir, encontrar, llamar)
    'llegar', 'pasar', 'deber', 'poner', 'parecer', 'quedar', 'creer', 'hablar', 'llevar', 'dejar', 'seguir', 'encontrar', 'llamar'
]

# EJEMPLO de cómo funcionan las stopwords:
# Texto original: "yo quiero saber mi saldo por favor"
# Después de stopwords: "quiero saber saldo" (se eliminaron "yo", "mi", "por", "favor")


# ==================== DATOS DE ENTRENAMIENTO ====================
# Son los ejemplos que el modelo usará para aprender
# Es como mostrarle al niño muchas tarjetas con ejemplos

textos = []        # Lista vacía para guardar los textos de ejemplo
intenciones = []   # Lista vacía para guardar las categorías

# ---------- INTENCIÓN 1: SALDO (consultar dinero disponible) ----------
# 20 ejemplos de cómo los usuarios piden su saldo
saldo = [
    "cual es mi saldo",           # Forma más común
    "quiero saber mi saldo",      # Pidiendo saber
    "cuanto tengo en mi cuenta",  # Preguntando cantidad
    "dime mi saldo disponible",   # Saldo disponible
    "ver saldo",                  # Versión corta
    "consultar saldo",            # Consultar
    "saldo actual",               # Saldo del momento
    "que saldo tengo",            # Preguntando
    "mostrar mi saldo",           # Mostrar
    "necesito mi saldo",          # Necesidad
    "saldo de mi cuenta",         # De mi cuenta
    "cuanto dinero tengo",        # Dinero total
    "ver mi saldo",               # Ver (repetido)
    "consulta de saldo",          # Consulta
    "saldo bancario",             # Saldo del banco
    "mi saldo",                   # Mío
    "sado",                       # Error tipográfico (falta la L)
    "salddo",                     # Error tipográfico (L doble)
    "consultar mi dinero",        # Consultar dinero
    "cuanto tengo"                # Versión corta
]
textos.extend(saldo)  # Agrega los 20 textos a la lista principal
# textos = textos + saldo (une las dos listas)

intenciones.extend(["saldo"] * len(saldo))  
# Crea una lista de 20 veces "saldo" y la agrega
# ["saldo", "saldo", "saldo", ...] (20 veces)

# ---------- INTENCIÓN 2: APELACIÓN (reclamar cargos no reconocidos) ----------
# 20 ejemplos de cómo los usuarios reclaman cargos
apelacion = [
    "quiero apelar un cargo",      # Apelar específicamente
    "reversion de pago",           # Revertir pago
    "no reconozco este cargo",     # No reconocimiento
    "apelar transaccion",          # Apelar transacción
    "me cobraron dos veces",       # Doble cobro
    "cargo no reconocido",         # Cargo desconocido
    "reversion por favor",         # Reversión
    "quiero disputar un pago",     # Disputar
    "apelar cargo",                # Apelar
    "reclamar transaccion",        # Reclamar
    "cargo indebido",              # Cargo incorrecto
    "error en mi estado de cuenta", # Error en cuenta
    "me hicieron un cobro que no es mio", # Cobro ajeno
    "quiero reportar un cargo",    # Reportar
    "desconozco esta transaccion", # Desconocido
    "apelacion",                   # Palabra clave
    "reclamo",                     # Reclamo
    "queja",                       # Queja
    "cobro duplicado",             # Duplicado
    "cobro incorrecto"             # Incorrecto
]
textos.extend(apelacion)
intenciones.extend(["apelacion"] * len(apelacion))

# ---------- INTENCIÓN 3: PRÉSTAMO (solicitar crédito) ----------
# 20 ejemplos de cómo los usuarios piden préstamos
prestamo = [
    "quiero un prestamo",          # Pedir préstamo
    "solicitar credito",           # Solicitar crédito
    "prestamo personal",           # Préstamo personal
    "necesito dinero prestado",    # Necesidad de dinero
    "credito rapido",              # Crédito rápido
    "solicitar prestamo",          # Solicitar
    "cuanto me prestan",           # Preguntar cantidad
    "requisitos para prestamo",    # Requisitos
    "tasas de prestamo",           # Tasas de interés
    "quiero pedir un credito",     # Pedir crédito
    "prestamo bancario",           # Préstamo del banco
    "solicitud de credito",        # Solicitud
    "me interesa un prestamo",     # Interés
    "financiamiento",              # Financiamiento
    "credito de nomina",           # Crédito de nómina
    "prestamo",                    # Palabra clave
    "credito",                     # Crédito
    "necesito dinero",             # Necesidad
    "me prestan",                  # Me prestan
    "solicitar un credito"         # Solicitar
]
textos.extend(prestamo)
intenciones.extend(["prestamo"] * len(prestamo))

# ---------- INTENCIÓN 4: GENERAL (saludos, preguntas comunes, despedidas) ----------
# 20 ejemplos de conversación general
general = [
    "gracias",                     # Agradecimiento
    "hola buenos dias",            # Saludo
    "que horario tienen",          # Horario
    "donde esta tu oficina",       # Ubicación
    "telefono de contacto",        # Teléfono
    "correo electronico",          # Correo
    "gracias por la ayuda",        # Agradecimiento
    "adios",                       # Despedida
    "buenas tardes",               # Saludo
    "como estas",                  # Pregunta personal
    "que tal",                     # Saludo informal
    "buen dia",                    # Buen día
    "informacion",                 # Pedir información
    "ayuda",                       # Pedir ayuda
    "soporte",                     # Soporte
    "hola",                        # Saludo corto
    "buenas",                      # Saludo
    "saludos",                     # Saludo
    "ok",                          # Afirmación
    "vale"                         # Afirmación (España)
]
textos.extend(general)
intenciones.extend(["general"] * len(general))

# Agregar respuestas cortas como "sí" y "no" a la categoría general
# Esto evita que el modelo clasifique "sí" como préstamo o saldo
textos.extend(["si", "sí", "no", "tal vez", "adelante", "continuar"])
intenciones.extend(["general"] * 6)  # Agrega 6 veces "general"


# ==================== CREAR DATAFRAME ====================
# DataFrame = tabla de datos (como Excel pero en Python)

df = pd.DataFrame({"texto": textos, "intencion": intenciones})
# Crea una tabla con dos columnas:
# - Columna "texto": contiene todos los textos de ejemplo
# - Columna "intencion": contiene la categoría de cada texto

df["texto_norm"] = df["texto"].apply(normalizar_texto)
# Crea una nueva columna llamada "texto_norm"
# .apply(normalizar_texto) = aplica la función de limpieza a CADA texto
# Ahora tenemos:
#   texto                    intencion    texto_norm
#   "cual es mi saldo"       saldo        "cual es mi saldo"
#   "¡Hola!"                 general      "hola"

print(f"Total de ejemplos de entrenamiento: {len(df)}")
# Muestra cuántos ejemplos hay (deberían ser 86: 20+20+20+20+6)

print("\nEjemplos de datos preparados:")
print(df.head(10))
# Muestra las primeras 10 filas para verificar


# ==================== CREAR PIPELINE DEL MODELO ====================
# Pipeline = tubería que hace varios pasos automáticamente

pipeline = Pipeline([
    # PASO 1: Vectorizador (convierte texto a números)
    ("vectorizer", TfidfVectorizer(
        ngram_range=(1, 3),      
        # ngram_range=(1, 3) = considera palabras individuales (1) 
        # y grupos de 2 o 3 palabras
        # Ejemplo: "no reconozco cargo" genera:
        # - Palabras solas: "no", "reconozco", "cargo"
        # - Pares: "no reconozco", "reconozco cargo"  
        # - Triples: "no reconozco cargo"
        
        max_features=3000,        
        # max_features=3000 = solo usa las 3000 palabras más comunes
        # Esto hace el modelo más rápido y usa menos memoria
        # Si hay 10,000 palabras diferentes, solo elige las 3000 más frecuentes
        
        stop_words=stopwords_es   
        # stop_words = ignora las palabras de nuestra lista
        # Palabras como "el", "la", "y" no aportan significado
    )),
    
    # PASO 2: Clasificador (aprende patrones y clasifica)
    ("clasificador", MultinomialNB(
        alpha=0.3  
        # alpha = suavizado de Laplace
        # Evita que palabras nuevas tengan probabilidad cero
        # Si alpha=0 y aparece palabra nueva → probabilidad 0 → error
        # alpha=0.3 da una pequeña probabilidad a palabras nuevas
    ))
])


# ==================== ENTRENAR EL MODELO ====================
# Entrenar = el modelo aprende de los ejemplos

pipeline.fit(df["texto_norm"], df["intencion"])
# fit() = entrenar el modelo
# - Recibe los textos limpios (df["texto_norm"])
# - Recibe las etiquetas (df["intencion"])
# - El modelo aprende qué palabras se asocian con cada intención

print("\n✅ Modelo entrenado exitosamente!")


# ==================== GUARDAR EL MODELO ====================
# Guardar el modelo para usarlo después sin reentrenar

os.makedirs("modelo", exist_ok=True)
# Crea una carpeta llamada "modelo" si no existe
# exist_ok=True = no da error si ya existe

joblib.dump(pipeline, "modelo/clasificador_intenciones.pkl")
# Guarda el modelo entrenado en un archivo
# - pipeline = el modelo que guardamos
# - "modelo/clasificador_intenciones.pkl" = ruta y nombre del archivo
# .pkl = pickle (formato de guardado de Python)

print(f"💾 Modelo guardado en 'modelo/clasificador_intenciones.pkl'")


# ==================== PRUEBA DEL MODELO ====================
# Esta parte solo se ejecuta si corremos este script directamente

if __name__ == "__main__":
    # __name__ es una variable especial de Python
    # "__main__" significa "este archivo se está ejecutando directamente"
    # Si importamos este archivo desde otro, NO se ejecutan las pruebas
    
    # Cargar el modelo guardado desde el disco
    modelo = joblib.load("modelo/clasificador_intenciones.pkl")
    # modelo ahora contiene el pipeline entrenado
    
    # Lista de frases para probar el modelo
    pruebas = [
        "cual es mi sado",      # Error tipográfico (sado en lugar de saldo)
        "reclamo",               # Debería detectar apelación
        "prestamo",              # Debería detectar préstamo
        "gracias",               # Debería detectar general
        "si",                    # Debería detectar general
        "no",                    # Debería detectar general
        "quiero continuar"       # Frase ambigua (general)
    ]
    
    print("\n" + "="*50)
    print("PRUEBAS DEL MODELO:")
    print("="*50)
    
    # Probar cada frase una por una
    for prueba in pruebas:
        # PASO 1: Limpiar el texto de prueba
        prueba_norm = normalizar_texto(prueba)
        # "cual es mi sado" -> "cual es mi sado" (limpio)
        
        # PASO 2: Predecir la intención
        # .predict() recibe una lista de textos (por eso [prueba_norm])
        # [0] toma el primer resultado (porque devuelve lista)
        prediccion = modelo.predict([prueba_norm])[0]
        
        # PASO 3: Mostrar resultado
        print(f"📝 '{prueba}' -> {prediccion}")
    
    # Prueba adicional que muestra probabilidades
    print("\n" + "="*50)
    print("PRUEBA CON PROBABILIDADES:")
    print("="*50)
    
    frase_prueba = "no reconozco este cargo"
    frase_norm = normalizar_texto(frase_prueba)
    
    # Predecir la categoría (la más probable)
    prediccion = modelo.predict([frase_norm])[0]
    
    # Predecir probabilidades de TODAS las categorías
    probabilidades = modelo.predict_proba([frase_norm])[0]
    # predict_proba devuelve algo como: [0.85, 0.05, 0.05, 0.05]
    # 85% apelacion, 5% cada una de las otras
    
    print(f"Frase: '{frase_prueba}'")
    print(f"Predicción: {prediccion}")
    print("\nProbabilidades por clase:")
    
    # Mostrar probabilidad para cada intención
    clases = modelo.classes_  # ['apelacion', 'general', 'prestamo', 'saldo']
    for i, clase in enumerate(clases):
        # i = índice (0, 1, 2, 3)
        # clase = nombre de la categoría
        # probabilidades[i] = probabilidad de esa categoría
        print(f"  {clase}: {probabilidades[i]*100:.2f}%")
        # :.2f = muestra 2 decimales
        # 0.8543 -> 85.43%


# ==================== RESUMEN DE EJECUCIÓN ====================
# Cuando ejecutes este programa, pasará por estas etapas:
#
# 1. Importa todas las herramientas necesarias
# 2. Define la función normalizar_texto()
# 3. Define la lista stopwords_es
# 4. Carga 86 ejemplos de entrenamiento (20 por categoría + 6 extra)
# 5. Crea una tabla (DataFrame) con los datos
# 6. Limpia todos los textos
# 7. Crea el Pipeline (vectorizador + clasificador)
# 8. Entrena el modelo
# 9. Guarda el modelo en un archivo
# 10. Prueba el modelo con 7 frases nuevas
# 11. Muestra los resultados con probabilidades