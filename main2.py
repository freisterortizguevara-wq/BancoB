# ===================================================================
# main.py - BBVA Colombia Asistente Inteligente con Neon PostgreSQL
# ===================================================================
# ESTE PROGRAMA ES EL ASISTENTE VIRTUAL COMPLETO QUE:
# 1. Recibe mensajes de usuarios (por web o WhatsApp)
# 2. Clasifica qué quiere el usuario usando Machine Learning
# 3. Genera respuestas automáticas
# 4. Guarda conversaciones en base de datos
# 5. Muestra un dashboard con estadísticas
# ===================================================================

# ==================== IMPORTACIONES ====================
# Cada importación trae herramientas específicas

from fastapi import FastAPI, Request, Form, UploadFile, File, Query
# FastAPI = Framework para crear APIs web
# Request = Para recibir datos de peticiones HTTP
# Form = Para recibir datos de formularios
# UploadFile = Para recibir archivos (PDFs)
# File = Para archivos en peticiones
# Query = Para parámetros en la URL

from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
# HTMLResponse = Envía páginas web (código HTML)
# JSONResponse = Envía datos en formato JSON (como diccionarios)
# StreamingResponse = Envía archivos grandes en partes (para descargas)

import random
# random = Para elegir respuestas aleatorias (variedad)

import unicodedata
# unicodedata = Para eliminar tildes y normalizar texto

import re
# re = Expresiones regulares (para buscar patrones en texto)

from datetime import datetime
# datetime = Para obtener fecha y hora actual

import os
# os = Para acceder al sistema operativo (variables de entorno, carpetas)

import pandas as pd
# pandas = Para manejar tablas de datos (como Excel)

from io import BytesIO, StringIO
# BytesIO = Para manejar archivos binarios en memoria (PDFs)
# StringIO = Para manejar texto en memoria (CSVs)

import joblib
# joblib = Para cargar el modelo de Machine Learning entrenado

import httpx
# httpx = Para hacer peticiones HTTP (enviar mensajes a WhatsApp)

import psycopg2
# psycopg2 = Para conectar a PostgreSQL (base de datos)

from psycopg2.extras import RealDictCursor
# RealDictCursor = Para que las consultas devuelvan diccionarios (más fáciles de usar)

from dotenv import load_dotenv
# dotenv = Para cargar variables secretas desde archivo .env

# ==================== CONFIGURACIÓN INICIAL ====================

load_dotenv()
# Carga las variables del archivo .env (tokens, contraseñas, etc.)
# El archivo .env debe estar en la misma carpeta

# Crear la aplicación FastAPI
app = FastAPI(title="BBVA Colombia Asistente Inteligente")
# app es el objeto principal que manejará todas las rutas (endpoints)

# ==================== CONFIGURACIÓN DE WHATSAPP ====================
# Estas variables se leen desde el archivo .env o usan valores por defecto

WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN", "TU_TOKEN_AQUI")
# Token de autenticación de Meta (Facebook) para WhatsApp Business API

PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID", "TU_PHONE_NUMBER_ID")
# ID del número de teléfono de BBVA en WhatsApp Business

VERIFY_TOKEN = os.getenv("VERIFY_TOKEN", "bbva_colombia_2024")
# Token para verificar el webhook de WhatsApp (seguridad)

# ==================== MEMORIA EN RAM ====================
# MEMORIA es un diccionario que guarda el estado de cada conversación
# Es como una "libreta de apuntes" temporal

MEMORIA = {}
# Estructura: MEMORIA[número_telefono] = {
#     "ultima_intencion": "prestamo",
#     "esperando_respuesta": True,
#     "historial": ["mensaje1", "mensaje2"]
# }

def obtener_memoria(telefono):
    """
    Obtiene la memoria de un usuario por su número de teléfono.
    Si no existe, crea una nueva entrada con valores por defecto.
    
    Ejemplo:
        obtener_memoria("573053667704")
        Devuelve: {"ultima_intencion": "general", "esperando_respuesta": False, "historial": []}
    """
    if telefono not in MEMORIA:
        # Crear memoria nueva para este usuario
        MEMORIA[telefono] = {
            "ultima_intencion": "general",  # Última intención detectada
            "esperando_respuesta": False,   # ¿Estamos esperando su respuesta?
            "historial": []                 # Historial de mensajes
        }
    return MEMORIA[telefono]

def actualizar_memoria(telefono, intencion, esperando=False):
    """
    Actualiza la memoria de un usuario con nueva información.
    
    Ejemplo:
        actualizar_memoria("573053667704", "prestamo", esperando=True)
        Guarda que el usuario pidió préstamo y espera confirmación
    """
    MEMORIA[telefono] = {
        "ultima_intencion": intencion,
        "esperando_respuesta": esperando,
        "historial": MEMORIA.get(telefono, {}).get("historial", [])
    }

# ==================== CONEXIÓN A BASE DE DATOS NEON POSTGRESQL ====================

def get_db():
    """
    Crea y devuelve una conexión a la base de datos PostgreSQL.
    La URL de conexión se lee de la variable de entorno DATABASE_URL.
    """
    DATABASE_URL = os.getenv("DATABASE_URL")
    if not DATABASE_URL:
        raise Exception("❌ DATABASE_URL no configurada")
    return psycopg2.connect(DATABASE_URL)
    # psycopg2.connect() abre la conexión a la base de datos

def init_db():
    """
    Inicializa la base de datos creando la tabla 'conversaciones' si no existe.
    Esta función se ejecuta automáticamente al iniciar el programa.
    """
    try:
        conn = get_db()           # Obtener conexión
        cur = conn.cursor()       # Crear cursor (para ejecutar SQL)
        
        # Comando SQL para crear la tabla
        cur.execute("""
            CREATE TABLE IF NOT EXISTS conversaciones (
                id SERIAL PRIMARY KEY,           # ID único auto-incremental
                telefono TEXT,                    # Número de teléfono del usuario
                mensaje TEXT,                     # Mensaje del usuario
                intencion TEXT,                   # Intención detectada (saldo, préstamo, etc)
                respuesta TEXT,                   # Respuesta del bot
                escalado INTEGER DEFAULT 0,       # 1 si necesita atención humana, 0 si no
                canal TEXT DEFAULT 'web',         # 'web' o 'whatsapp'
                pdf_texto TEXT,                   # Texto extraído del PDF (si hay)
                pdf_nombre TEXT,                  # Nombre original del archivo PDF
                pdf_archivo BYTEA,                # Contenido binario del PDF
                fecha TIMESTAMP DEFAULT NOW()     # Fecha y hora de la conversación
            )
        """)
        
        conn.commit()   # Guardar los cambios
        cur.close()     # Cerrar el cursor
        conn.close()    # Cerrar la conexión
        print("✅ Base de datos Neon inicializada")
        
    except Exception as e:
        print(f"❌ Error inicializando Neon: {e}")

# Ejecutar la inicialización de la base de datos
init_db()

# ==================== MODELO DE MACHINE LEARNING ====================

MODELO_PATH = "modelo/clasificador_intenciones.pkl"
# Ruta donde está guardado el modelo entrenado

def cargar_modelo():
    """
    Carga el modelo de Machine Learning desde el archivo .pkl
    Si no existe, devuelve None (se usarán reglas simples)
    """
    try:
        if os.path.exists(MODELO_PATH):
            m = joblib.load(MODELO_PATH)  # Cargar modelo
            print("✅ Modelo ML cargado")
            return m
        print("⚠️ Modelo no encontrado, usando reglas")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def normalizar_texto_ml(texto):
    """
    Limpia el texto de la misma forma que se usó para entrenar el modelo.
    Pasos:
    1. Convertir a minúsculas
    2. Eliminar tildes
    3. Eliminar signos de puntuación y números
    4. Eliminar espacios extras
    
    Ejemplo: "¡Hola! ¿Cómo estás? 123" -> "hola como estas"
    """
    texto = texto.lower()                                          # "HOLA" -> "hola"
    texto = unicodedata.normalize('NFKD', texto).encode('ASCII', 'ignore').decode('utf-8')  # "camión" -> "camion"
    texto = re.sub(r'[^a-z\s]', '', texto)                         # "hola123" -> "hola"
    return re.sub(r'\s+', ' ', texto).strip()                      # "hola  mundo" -> "hola mundo"

def clasificar_por_reglas(texto, tiene_pdf=False):
    """
    Clasifica la intención usando reglas simples (sin Machine Learning).
    Esto se usa como fallback cuando no hay modelo ML.
    
    Parámetros:
        texto: el mensaje del usuario
        tiene_pdf: True si el usuario adjuntó un PDF
    
    Devuelve: una de estas categorías:
        "confirmacion", "negacion", "saldo", "prestamo", 
        "apelacion", "asesor", "tarjeta", "general"
    """
    t = texto.lower().strip()  # Normalizar texto
    
    # Si tiene PDF, siempre es apelación (reclamo con documento)
    if tiene_pdf:
        return "apelacion"
    
    # Palabras de confirmación (el usuario dice SÍ a algo)
    if t in ["si", "sí", "quiero", "adelante", "continuar", "aceptar", "ok", "dale", "claro"]:
        return "confirmacion"
    
    # Palabras de negación (el usuario dice NO)
    if t in ["no", "nop", "cancelar", "rechazar", "mejor no"]:
        return "negacion"
    
    # Palabras relacionadas con SALDO
    if any(p in t for p in ["saldo", "cuanto tengo", "dinero", "balance", "plata", "mis ahorros"]):
        return "saldo"
    
    # Palabras relacionadas con PRÉSTAMO
    if any(p in t for p in ["prestamo", "credito", "financiamiento", "pedir dinero", "prestado"]):
        return "prestamo"
    
    # Palabras relacionadas con APELACIÓN (reclamos)
    if any(p in t for p in ["apelar", "reclamar", "cargo", "cobro", "reversion", "disputar", "queja", "factura"]):
        return "apelacion"
    
    # Palabras relacionadas con ASESOR HUMANO
    if any(p in t for p in ["asesor", "humano", "ejecutivo", "persona", "hablar con"]):
        return "asesor"
    
    # Palabras relacionadas con TARJETA DE CRÉDITO
    if any(p in t for p in ["tarjeta", "credito", "visa", "mastercard"]):
        return "tarjeta"
    
    # Si no coincide con nada, es conversación GENERAL
    return "general"

# Cargar el modelo ML (si existe)
modelo_ml = cargar_modelo()

def clasificar_intencion(texto, telefono=None, tiene_pdf=False):
    """
    Clasifica la intención del usuario usando:
    1. Prioridad: si tiene PDF -> apelación
    2. Prioridad: si estamos esperando respuesta -> confirmación/rechazo
    3. Modelo ML (si está disponible)
    4. Reglas simples (fallback)
    
    Parámetros:
        texto: mensaje del usuario
        telefono: número (opcional, para memoria)
        tiene_pdf: si adjuntó PDF
    
    Devuelve: categoría de intención
    """
    texto_limpio = texto.lower().strip()
    
    # CASO 1: Usuario adjuntó un PDF -> es una apelación/reclamo
    if tiene_pdf:
        return "apelacion"
    
    # CASO 2: Verificar si estamos esperando una respuesta SÍ/NO de este usuario
    if telefono:
        memoria = obtener_memoria(telefono)
        if memoria["esperando_respuesta"]:
            # El usuario dijo SÍ a lo que preguntamos
            if texto_limpio in ["si", "sí", "quiero", "adelante", "continuar", "aceptar", "ok", "dale", "claro"]:
                return memoria["ultima_intencion"] + "_aceptado"
            # El usuario dijo NO a lo que preguntamos
            if texto_limpio in ["no", "nop", "cancelar", "rechazar", "mejor no"]:
                return "rechazo"
    
    # CASO 3: Usar el modelo de Machine Learning (si está disponible)
    if modelo_ml:
        try:
            texto_normalizado = normalizar_texto_ml(texto)
            prediccion = modelo_ml.predict([texto_normalizado])[0]
            return prediccion
        except:
            # Si hay error con el modelo, continuamos con reglas
            pass
    
    # CASO 4: Fallback a reglas simples
    return clasificar_por_reglas(texto, tiene_pdf)

# ==================== RESPUESTAS DEL BOT ====================
# Diccionario con las respuestas que dará el bot según la intención
# Cada intención tiene MÚLTIPLES respuestas posibles (se elige aleatoria)

RESPUESTAS = {
    # Respuestas para consulta de SALDO
    "saldo": [
        "💰 ¡Claro! Tu saldo actual es *$5.000.000 COP*. ¿Necesitas ver los movimientos recientes?",
        "🏦 Tu cuenta tiene un saldo de *$5.000.000 COP*. Recuerda que puedes transferir desde la app BBVA.",
        "💵 Consulta realizada: Tu saldo disponible es *$5.000.000 COP*."
    ],
    
    # Respuestas para solicitud de PRÉSTAMO
    "prestamo": [
        "🏦 BBVA Colombia te ofrece créditos desde *$1.000.000 COP*. ¿Te interesa?",
        "✅ Estás preaprobado hasta *$10.000.000 COP*. ¿Continuamos con la simulación?",
        "📊 Nuestros créditos tienen aprobación en 24 horas. ¿Quieres más información?"
    ],
    
    # Respuestas cuando el usuario ACEPTA el préstamo (dice SÍ)
    "prestamo_aceptado": [
        "✅ ¡Excelente! Un asesor te contactará en 24 horas para formalizar tu crédito.",
        "🏦 Hemos registrado tu solicitud. En breve recibirás la aprobación.",
        "📄 ¡Felicidades! Te enviaremos los documentos digitales por WhatsApp."
    ],
    
    # Respuestas para APELACIONES (reclamos con PDF)
    "apelacion": [
        "📄 Hemos recibido tu documento. Un ejecutivo revisará tu caso en 24 horas.",
        "🔍 Revisaremos tu caso prioritariamente. Te notificaremos por WhatsApp.",
        "✅ Apelación registrada. Espera comunicación de nuestro equipo."
    ],
    
    # Respuestas para pedir ASESOR HUMANO
    "asesor": [
        "👨‍💼 Te conecto con un asesor BBVA. Te llamarán en los próximos 15 minutos.",
        "📞 Un ejecutivo te contactará en breve. ¿Prefieres WhatsApp o llamada?",
        "🕒 Te llamaremos antes de 1 hora. Gracias por tu paciencia."
    ],
    
    # Respuestas para consultas de TARJETA DE CRÉDITO
    "tarjeta": [
        "💳 Tu tarjeta BBVA tiene disponible *$3.000.000 COP*. Tu próximo pago es el 15 de cada mes.",
        "💰 Tu tarjeta de crédito tiene un cupo de *$3.000.000 COP*. ¿Necesitas diferir una compra?",
        "🛡️ Recuerda que tienes seguro de protección de compras con tu tarjeta BBVA."
    ],
    
    # Respuestas cuando el usuario RECHAZA una oferta (dice NO)
    "rechazo": [
        "Entendido. ¿En qué más puedo ayudarte?",
        "No hay problema. Estoy aquí para lo que necesites.",
        "👍 Perfecto. Si cambias de opinión, solo dímelo."
    ],
    
    # Respuestas para conversación GENERAL (saludos, agradecimientos, etc.)
    "general": [
        "🙋 Hola, soy el asistente BBVA Colombia. ¿En qué te ayudo? Puedo consultar saldos, préstamos o conectar con un asesor.",
        "🏦 ¡Bienvenido! ¿Necesitas información sobre tu cuenta, tarjeta o préstamos?",
        "💬 Hola. ¿En qué puedo asistirte hoy?"
    ]
}

def generar_respuesta(intencion, telefono=None):
    """
    Genera una respuesta apropiada según la intención detectada.
    
    Parámetros:
        intencion: categoría detectada (saldo, prestamo, etc.)
        telefono: número de teléfono (opcional, para actualizar memoria)
    
    Devuelve: el texto de la respuesta
    """
    esperar = False  # ¿Debemos esperar una respuesta del usuario después?
    
    # Elegir respuesta según la intención
    if intencion == "prestamo":
        esperar = True  # Después de ofrecer préstamo, esperamos SÍ/NO
        respuesta = random.choice(RESPUESTAS["prestamo"])
        
    elif intencion == "prestamo_aceptado":
        respuesta = random.choice(RESPUESTAS["prestamo_aceptado"])
        
    elif intencion == "rechazo":
        respuesta = random.choice(RESPUESTAS["rechazo"])
        
    elif intencion == "apelacion":
        respuesta = random.choice(RESPUESTAS["apelacion"])
        
    elif intencion == "saldo":
        respuesta = random.choice(RESPUESTAS["saldo"])
        
    elif intencion == "asesor":
        respuesta = random.choice(RESPUESTAS["asesor"])
        
    elif intencion == "tarjeta":
        respuesta = random.choice(RESPUESTAS["tarjeta"])
        
    else:
        respuesta = random.choice(RESPUESTAS["general"])
    
    # Actualizar la memoria del usuario
    if telefono:
        actualizar_memoria(telefono, intencion, esperar)
    
    return respuesta

def extraer_texto_pdf(archivo_bytes):
    """
    Extrae el texto de un archivo PDF (para análisis posterior).
    
    Parámetros:
        archivo_bytes: el contenido binario del PDF
    
    Devuelve: texto extraído o mensaje de error
    """
    try:
        import pdfplumber  # Librería para leer PDFs
        import io
        
        # Abrir el PDF desde los bytes
        with pdfplumber.open(io.BytesIO(archivo_bytes)) as pdf:
            texto = ""
            # Recorrer cada página del PDF
            for pagina in pdf.pages:
                parte = pagina.extract_text()
                if parte:
                    texto += parte + "\n"
        
        # Devolver solo los primeros 5000 caracteres (para no saturar)
        return texto[:5000] if texto.strip() else "Sin texto extraíble del documento."
        
    except Exception as e:
        print(f"Error PDF: {e}")
        return f"Error al procesar el PDF: {str(e)}"

# ==================== FUNCIONES DE BASE DE DATOS ====================

def guardar_conversacion(telefono, mensaje, intencion, respuesta, escalado, pdf_texto, pdf_nombre, canal, pdf_archivo=None):
    """
    Guarda una conversación completa en la base de datos.
    
    Parámetros:
        telefono: número del usuario
        mensaje: lo que escribió
        intencion: categoría detectada
        respuesta: lo que respondió el bot
        escalado: 1 si necesita humano, 0 si no
        pdf_texto: texto extraído del PDF
        pdf_nombre: nombre del archivo PDF
        canal: 'web' o 'whatsapp'
        pdf_archivo: contenido binario del PDF (opcional)
    """
    try:
        conn = get_db()
        cur = conn.cursor()
        
        # Insertar registro en la tabla
        cur.execute("""
            INSERT INTO conversaciones 
            (telefono, mensaje, intencion, respuesta, escalado, pdf_texto, pdf_nombre, canal, fecha, pdf_archivo)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW(), %s)
        """, (telefono, mensaje[:500], intencion, respuesta[:500], escalado, 
              (pdf_texto or "")[:5000], pdf_nombre, canal, pdf_archivo))
        
        conn.commit()
        cur.close()
        conn.close()
        print(f"✅ Guardado: {intencion} - PDF: {pdf_nombre if pdf_nombre else 'Sin PDF'}")
        
    except Exception as e:
        print(f"❌ Error guardando: {e}")

def obtener_conversaciones(limite=100, canal=""):
    """
    Obtiene las últimas conversaciones de la base de datos.
    
    Parámetros:
        limite: número máximo de registros a devolver
        canal: filtrar por 'web' o 'whatsapp' (vacío = todos)
    
    Devuelve: lista de diccionarios con las conversaciones
    """
    try:
        conn = get_db()
        cur = conn.cursor(cursor_factory=RealDictCursor)  # Devuelve diccionarios
        
        if canal:
            # Filtrar por canal específico
            cur.execute("""
                SELECT id, telefono, mensaje, intencion, respuesta, escalado, canal, pdf_nombre, fecha 
                FROM conversaciones 
                WHERE canal=%s 
                ORDER BY id DESC LIMIT %s
            """, (canal, limite))
        else:
            # Todos los canales
            cur.execute("""
                SELECT id, telefono, mensaje, intencion, respuesta, escalado, canal, pdf_nombre, fecha 
                FROM conversaciones 
                ORDER BY id DESC LIMIT %s
            """, (limite,))
        
        rows = cur.fetchall()
        cur.close()
        conn.close()
        
        # Convertir fechas a string para JSON
        resultado = []
        for row in rows:
            item = dict(row)
            if item.get('fecha'):
                item['fecha'] = item['fecha'].isoformat() if hasattr(item['fecha'], 'isoformat') else str(item['fecha'])
            resultado.append(item)
        return resultado
        
    except Exception as e:
        print(f"❌ Error obteniendo conversaciones: {e}")
        return []

def obtener_estadisticas():
    """
    Calcula estadísticas de las conversaciones para el dashboard.
    
    Devuelve diccionario con:
        total: número total de conversaciones
        escalados: cuántas requirieron humano
        tasa_escalado: porcentaje de escalados
        intenciones: conteo por cada intención
        canales: conteo por cada canal
        pdfs: cuántas conversaciones tienen PDF
    """
    try:
        conn = get_db()
        cur = conn.cursor()
        
        # Total de conversaciones
        cur.execute("SELECT COUNT(*) FROM conversaciones")
        total = cur.fetchone()[0]
        
        # Conversaciones escaladas (requieren humano)
        cur.execute("SELECT COUNT(*) FROM conversaciones WHERE escalado=1")
        escalados = cur.fetchone()[0]
        
        # Conteo por intención
        cur.execute("SELECT intencion, COUNT(*) FROM conversaciones WHERE intencion IS NOT NULL GROUP BY intencion")
        intenciones = dict(cur.fetchall())
        
        # Conteo por canal
        cur.execute("SELECT canal, COUNT(*) FROM conversaciones GROUP BY canal")
        canales = dict(cur.fetchall())
        
        # Conversaciones con PDF
        cur.execute("SELECT COUNT(*) FROM conversaciones WHERE pdf_nombre IS NOT NULL")
        pdfs = cur.fetchone()[0]
        
        cur.close()
        conn.close()
        
        return {
            "total": total,
            "escalados": escalados,
            "tasa_escalado": round(escalados/total*100, 1) if total else 0,
            "intenciones": intenciones,
            "canales": canales,
            "pdfs": pdfs
        }
        
    except Exception as e:
        print(f"❌ Error en estadisticas: {e}")
        return {"total": 0, "escalados": 0, "tasa_escalado": 0, "intenciones": {}, "canales": {}, "pdfs": 0}

def obtener_tendencia():
    """
    Obtiene la tendencia de conversaciones de los últimos 7 días.
    
    Devuelve: (fechas, conteos) donde fechas son strings y conteos son números
    """
    try:
        conn = get_db()
        cur = conn.cursor()
        
        cur.execute("""
            SELECT DATE(fecha) as fecha, COUNT(*) 
            FROM conversaciones 
            WHERE fecha IS NOT NULL 
            GROUP BY DATE(fecha) 
            ORDER BY DATE(fecha) DESC 
            LIMIT 7
        """)
        
        rows = cur.fetchall()
        cur.close()
        conn.close()
        
        rows = list(reversed(rows))  # Ordenar de más antiguo a más reciente
        return [str(r[0]) for r in rows], [r[1] for r in rows]
        
    except Exception as e:
        print(f"❌ Error en tendencia: {e}")
        return [], []

# ==================== ENDPOINTS DE LA API ====================

@app.get("/api/descargar-pdf/{conversacion_id}")
async def descargar_pdf(conversacion_id: int):
    """
    ENDPOINT: GET /api/descargar-pdf/123
    Permite descargar el PDF original de una conversación.
    
    Parámetro:
        conversacion_id: ID de la conversación en la base de datos
    """
    try:
        conn = get_db()
        cur = conn.cursor()
        
        # Buscar el PDF por ID
        cur.execute("SELECT pdf_nombre, pdf_archivo FROM conversaciones WHERE id = %s", (conversacion_id,))
        row = cur.fetchone()
        cur.close()
        conn.close()
        
        if not row or not row[0]:
            return JSONResponse({"error": "No hay PDF asociado"}, status_code=404)
        
        pdf_nombre, pdf_archivo = row
        
        if not pdf_archivo:
            return JSONResponse({"error": "El archivo PDF original no está disponible"}, status_code=404)
        
        # Enviar el archivo para descarga
        return StreamingResponse(
            BytesIO(pdf_archivo), 
            media_type="application/pdf", 
            headers={"Content-Disposition": f"attachment; filename={pdf_nombre}"}
        )
        
    except Exception as e:
        print(f"❌ Error descargando PDF: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

# ==================== WHATSAPP WEBHOOK ====================

async def enviar_whatsapp(telefono: str, texto: str):
    """
    Envía un mensaje de WhatsApp usando la API de Meta (Facebook).
    
    Parámetros:
        telefono: número de destino (con código de país)
        texto: mensaje a enviar
    """
    if WHATSAPP_TOKEN == "TU_TOKEN_AQUI":
        # Modo simulación (sin token real)
        print(f"📱 Simulando envío a {telefono}")
        return True
    
    url = f"https://graph.facebook.com/v18.0/{PHONE_NUMBER_ID}/messages"
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url, 
                headers={
                    "Authorization": f"Bearer {WHATSAPP_TOKEN}",
                    "Content-Type": "application/json"
                },
                json={
                    "messaging_product": "whatsapp",
                    "to": telefono,
                    "type": "text",
                    "text": {"body": texto}
                }
            )
            return response.status_code == 200
    except:
        return False

@app.get("/webhook")
async def verificar_webhook(request: Request):
    """
    ENDPOINT: GET /webhook
    Meta/Facebook llama a este endpoint para VERIFICAR que el webhook es válido.
    
    WhatsApp envía un token de verificación, debemos responder con el mismo token.
    """
    params = request.query_params
    verify_token = params.get("hub.verify_token")
    
    if verify_token == VERIFY_TOKEN:
        # Responder con el challenge token
        return int(params.get("hub.challenge"))
    
    return JSONResponse({"error": "Token inválido"}, status_code=403)

@app.post("/webhook")
async def recibir_whatsapp(request: Request):
    """
    ENDPOINT: POST /webhook
    Recibe mensajes de WhatsApp cuando un usuario escribe al bot.
    
    Meta envía un JSON con la información del mensaje.
    """
    body = await request.json()
    
    try:
        # Extraer datos del JSON que envía WhatsApp
        entry = body["entry"][0]["changes"][0]["value"]
        
        if "messages" not in entry:
            return JSONResponse({"status": "ok"})
        
        msg = entry["messages"][0]
        
        # Solo procesar mensajes de texto (ignorar imágenes, audios, etc)
        if msg.get("type") != "text":
            return JSONResponse({"status": "ok"})
        
        telefono = msg["from"]           # Número del usuario
        texto = msg["text"]["body"]      # Mensaje escrito
        
        # Clasificar intención
        intencion = clasificar_intencion(texto, telefono)
        
        # Generar respuesta
        respuesta = generar_respuesta(intencion, telefono)
        
        # Determinar si necesita escalar a humano
        escalado = 1 if intencion in ["asesor", "apelacion"] else 0
        
        # Guardar en base de datos
        guardar_conversacion(telefono, texto, intencion, respuesta, escalado, None, None, "whatsapp", None)
        
        # Enviar respuesta por WhatsApp
        await enviar_whatsapp(telefono, respuesta)
        
    except Exception as e:
        print(f"Error en webhook: {e}")
        pass
    
    return JSONResponse({"status": "ok"})

# ==================== ENDPOINTS DE LA API (Dashboard) ====================

@app.get("/api/conversaciones")
async def api_conversaciones(canal: str = Query(default="")):
    """
    ENDPOINT: GET /api/conversaciones?canal=web
    Devuelve las últimas conversaciones en formato JSON.
    
    Parámetro opcional:
        canal: filtrar por 'web' o 'whatsapp'
    """
    return JSONResponse(obtener_conversaciones(100, canal))

@app.get("/api/estadisticas")
async def api_estadisticas():
    """
    ENDPOINT: GET /api/estadisticas
    Devuelve estadísticas completas para el dashboard.
    """
    stats = obtener_estadisticas()
    fechas, conteos = obtener_tendencia()
    stats["tendencia_fechas"] = fechas
    stats["tendencia_conteos"] = conteos
    return JSONResponse(stats)

@app.post("/api/simular")
async def simular(
    telefono: str = Form(...), 
    mensaje: str = Form(...), 
    pdf: UploadFile = File(None)
):
    """
    ENDPOINT: POST /api/simular
    Simula una conversación desde el chat web.
    
    Parámetros:
        telefono: número del usuario (simulado)
        mensaje: texto del mensaje
        pdf: archivo PDF opcional (para apelaciones)
    """
    pdf_texto = ""
    pdf_nombre = ""
    pdf_archivo = None
    tiene_pdf = False
    
    # Procesar PDF si fue adjuntado
    if pdf and pdf.filename:
        pdf_nombre = pdf.filename
        tiene_pdf = True
        try:
            contenido = await pdf.read()
            pdf_archivo = contenido
            pdf_texto = extraer_texto_pdf(contenido)
        except Exception as e:
            print(f"Error PDF: {e}")
            pdf_texto = f"Error al leer el PDF: {str(e)}"
    
    # Clasificar intención
    intencion = clasificar_intencion(mensaje, telefono, tiene_pdf)
    
    # Generar respuesta
    respuesta = generar_respuesta(intencion, telefono)
    
    # Determinar si necesita escalar a humano
    escalado = 1 if intencion in ["asesor", "apelacion"] else 0
    
    # Guardar en base de datos
    guardar_conversacion(telefono, mensaje, intencion, respuesta, escalado, pdf_texto, pdf_nombre, "web", pdf_archivo)
    
    return JSONResponse({
        "respuesta": respuesta,
        "intencion": intencion,
        "escalado": escalado,
        "pdf_leido": bool(pdf_texto),
        "pdf_nombre": pdf_nombre
    })

@app.delete("/api/conversaciones")
async def eliminar_todas():
    """
    ENDPOINT: DELETE /api/conversaciones
    Elimina TODAS las conversaciones de la base de datos.
    (Usar con cuidado - solo para pruebas)
    """
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("DELETE FROM conversaciones")
        conn.commit()
        cur.close()
        conn.close()
        return JSONResponse({"message": "Eliminadas todas las conversaciones"})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/api/exportar/excel")
async def exportar_excel():
    """
    ENDPOINT: GET /api/exportar/excel
    Exporta todas las conversaciones a un archivo Excel (.xlsx)
    """
    conn = get_db()
    df = pd.read_sql_query("""
        SELECT id, telefono, mensaje, intencion, respuesta, escalado, canal, pdf_nombre, fecha 
        FROM conversaciones ORDER BY id DESC
    """, conn)
    conn.close()
    
    # Crear archivo Excel en memoria
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Conversaciones BBVA')
    output.seek(0)
    
    return StreamingResponse(
        output, 
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=conversaciones_bbva.xlsx"}
    )

@app.get("/api/exportar/csv")
async def exportar_csv():
    """
    ENDPOINT: GET /api/exportar/csv
    Exporta todas las conversaciones a un archivo CSV
    """
    conn = get_db()
    df = pd.read_sql_query("""
        SELECT id, telefono, mensaje, intencion, respuesta, escalado, canal, pdf_nombre, fecha 
        FROM conversaciones ORDER BY id DESC
    """, conn)
    conn.close()
    
    # Crear archivo CSV en memoria
    buffer = StringIO()
    df.to_csv(buffer, index=False, encoding='utf-8-sig')
    buffer.seek(0)
    
    return StreamingResponse(
        iter([buffer.getvalue()]), 
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=conversaciones_bbva.csv"}
    )

@app.get("/health")
async def health():
    """
    ENDPOINT: GET /health
    Verifica que el servicio esté funcionando.
    """
    return JSONResponse({"status": "ok"})

# ==================== HTML (INTERFAZ WEB) ====================
# El HTML es el código de la página web que ven los usuarios
# Contiene CSS para estilos, JavaScript para interactividad, etc.

HTML = """<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BBVA Colombia | Asistente Inteligente</title>
    <!-- Librerías externas -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-datalabels@2/dist/chartjs-plugin-datalabels.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:opsz,wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    ... (el resto del HTML con estilos y JavaScript)
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def home():
    """
    ENDPOINT: GET /
    Página principal del asistente (interfaz web)
    """
    return HTML

# ==================== INICIO DEL SERVIDOR ====================

if __name__ == "__main__":
    """
    Esta parte se ejecuta SOLO si corremos este archivo directamente
    (no si lo importamos desde otro archivo)
    """
    import uvicorn  # Servidor ASGI para ejecutar FastAPI
    
    print("=" * 50)
    print("🏦 BBVA Colombia Asistente Inteligente")
    print("=" * 50)
    print(f"📊 Dashboard: http://localhost:8000")
    print(f"📱 Webhook: http://localhost:8000/webhook")
    print("=" * 50)
    
    # Iniciar el servidor
    uvicorn.run(app, host="127.0.0.1", port=8000)