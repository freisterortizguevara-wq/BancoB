# main.py - BBVA Colombia Asistente Inteligente Enterprise ULTIMATE
from fastapi import FastAPI, Request, Form, UploadFile, File, Query
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
import random
import unicodedata
import re
from datetime import datetime, timedelta
import os
import pandas as pd
from io import BytesIO, StringIO
import joblib
import httpx
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
import logging
from logging.handlers import RotatingFileHandler
from collections import Counter
import json

# Cargar variables de entorno
load_dotenv()

app = FastAPI(title="BBVA Colombia Asistente Inteligente ULTIMATE")

# ==================== CONFIGURACIÓN LOGGING ====================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bbva_assistant")
handler = RotatingFileHandler('bbva_assistant.log', maxBytes=10000000, backupCount=5)
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

# ==================== CONFIG WHATSAPP ====================
WHATSAPP_TOKEN  = os.getenv("WHATSAPP_TOKEN", "TU_TOKEN_AQUI")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID", "TU_PHONE_NUMBER_ID")
VERIFY_TOKEN    = os.getenv("VERIFY_TOKEN", "bbva_colombia_2024")

# ==================== MEMORIA AVANZADA ====================
MEMORIA = {}

def obtener_memoria(telefono):
    if telefono not in MEMORIA:
        MEMORIA[telefono] = {
            "ultima_intencion": "general", 
            "esperando_respuesta": False, 
            "historial": [],
            "contexto": {},
            "contador": 0
        }
    return MEMORIA[telefono]

def actualizar_memoria(telefono, intencion, esperando=False, contexto=None):
    if telefono not in MEMORIA:
        MEMORIA[telefono] = {"ultima_intencion": intencion, "esperando_respuesta": esperando, "historial": [], "contexto": {}, "contador": 0}
    else:
        MEMORIA[telefono]["ultima_intencion"] = intencion
        MEMORIA[telefono]["esperando_respuesta"] = esperando
        MEMORIA[telefono]["contador"] = MEMORIA[telefono].get("contador", 0) + 1
        if contexto:
            MEMORIA[telefono]["contexto"].update(contexto)
    return MEMORIA[telefono]

# ==================== CONEXIÓN A NEON POSTGRESQL ====================
def get_db():
    DATABASE_URL = os.getenv("DATABASE_URL")
    if not DATABASE_URL:
        raise Exception("❌ DATABASE_URL no configurada")
    return psycopg2.connect(DATABASE_URL)

def init_db():
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS conversaciones (
                id SERIAL PRIMARY KEY,
                telefono TEXT,
                mensaje TEXT,
                intencion TEXT,
                respuesta TEXT,
                escalado INTEGER DEFAULT 0,
                canal TEXT DEFAULT 'web',
                pdf_texto TEXT,
                pdf_nombre TEXT,
                pdf_archivo BYTEA,
                fecha TIMESTAMP DEFAULT NOW()
            )
        """)
        conn.commit()
        cur.close()
        conn.close()
        print("✅ Base de datos Neon inicializada")
    except Exception as e:
        print(f"❌ Error inicializando Neon: {e}")

init_db()

# ==================== MODELO ML ====================
MODELO_PATH = "modelo/clasificador_intenciones.pkl"

def cargar_modelo():
    try:
        if os.path.exists(MODELO_PATH):
            m = joblib.load(MODELO_PATH)
            print("✅ Modelo ML cargado")
            return m
        print("⚠️ Modelo no encontrado, usando reglas")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def normalizar_texto_ml(texto):
    texto = texto.lower()
    texto = unicodedata.normalize('NFKD', texto).encode('ASCII', 'ignore').decode('utf-8')
    texto = re.sub(r'[^a-z\s]', '', texto)
    return re.sub(r'\s+', ' ', texto).strip()

def clasificar_por_reglas(texto, tiene_pdf=False):
    t = texto.lower().strip()
    
    if tiene_pdf:
        return "apelacion"
    
    if t in ["si", "sí", "quiero", "adelante", "continuar", "aceptar", "ok", "dale", "claro"]:
        return "confirmacion"
    if t in ["no", "nop", "cancelar", "rechazar", "mejor no"]:
        return "negacion"
    if any(p in t for p in ["saldo", "cuanto tengo", "dinero", "balance", "plata", "mis ahorros", "cuenta"]):
        return "saldo"
    if any(p in t for p in ["prestamo", "credito", "financiamiento", "pedir dinero", "prestado"]):
        return "prestamo"
    if any(p in t for p in ["apelar", "reclamar", "cargo", "cobro", "reversion", "disputar", "queja", "factura"]):
        return "apelacion"
    if any(p in t for p in ["asesor", "humano", "ejecutivo", "persona", "hablar con"]):
        return "asesor"
    if any(p in t for p in ["tarjeta", "credito", "visa", "mastercard"]):
        return "tarjeta"
    if any(p in t for p in ["inversion", "invertir", "rentabilidad", "fiducia", "fondo"]):
        return "inversion"
    if any(p in t for p in ["seguro", "proteccion", "cobertura", "vida", "vehiculo"]):
        return "seguro"
    if any(p in t for p in ["transferencia", "enviar", "recibir", "pago", "factura"]):
        return "transferencia"
    return "general"

modelo_ml = cargar_modelo()

def clasificar_intencion(texto, telefono=None, tiene_pdf=False):
    texto_limpio = texto.lower().strip()
    
    if tiene_pdf:
        return "apelacion"
    
    if telefono:
        memoria = obtener_memoria(telefono)
        if memoria["esperando_respuesta"]:
            if texto_limpio in ["si", "sí", "quiero", "adelante", "continuar", "aceptar", "ok", "dale", "claro"]:
                return memoria["ultima_intencion"] + "_aceptado"
            if texto_limpio in ["no", "nop", "cancelar", "rechazar", "mejor no"]:
                return "rechazo"
    
    if modelo_ml:
        try:
            pred = modelo_ml.predict([normalizar_texto_ml(texto)])[0]
            return pred
        except:
            pass
    
    return clasificar_por_reglas(texto, tiene_pdf)

def clasificar_emocion(texto):
    t = texto.lower()
    if any(p in t for p in ["gracias", "excelente", "perfecto", "bueno", "genial", "feliz"]):
        return "positivo"
    if any(p in t for p in ["mal", "problema", "error", "no funciona", "triste", "enojado", "frustrado"]):
        return "negativo"
    if any(p in t for p in ["ayuda", "urgente", "rapido", "importante", "necesito", "ya"]):
        return "urgente"
    return "neutral"

# ==================== RESPUESTAS ====================
RESPUESTAS = {
    "saldo": [
        "💰 ¡Claro! Tu saldo actual es *$5.000.000 COP*. ¿Necesitas ver los movimientos recientes?",
        "🏦 Tu cuenta tiene un saldo de *$5.000.000 COP*. Recuerda que puedes transferir desde la app BBVA."
    ],
    "prestamo": [
        "🏦 BBVA Colombia te ofrece créditos desde *$1.000.000 COP*. ¿Te interesa?",
        "✅ Estás preaprobado hasta *$10.000.000 COP*. ¿Continuamos con la simulación?"
    ],
    "prestamo_aceptado": [
        "✅ ¡Excelente! Un asesor te contactará en 24 horas para formalizar tu crédito.",
        "🏦 Hemos registrado tu solicitud. En breve recibirás la aprobación."
    ],
    "apelacion": [
        "📄 Hemos recibido tu documento. Un ejecutivo revisará tu caso en 24 horas.",
        "✅ Apelación registrada. Espera comunicación de nuestro equipo."
    ],
    "asesor": [
        "👨‍💼 Te conecto con un asesor BBVA. Te llamarán en los próximos 15 minutos.",
        "📞 Un ejecutivo te contactará en breve. ¿Prefieres WhatsApp o llamada?"
    ],
    "tarjeta": [
        "💳 Tu tarjeta BBVA tiene disponible *$3.000.000 COP*. Tu próximo pago es el 15 de cada mes.",
        "💰 Tu tarjeta de crédito tiene un cupo de *$3.000.000 COP*."
    ],
    "inversion": [
        "📈 BBVA tiene opciones de inversión desde *$500.000 COP* con rentabilidad del *8% EA*.",
        "📊 Tenemos planes de inversión a 1, 3 y 5 años. ¿Cuál te interesa?"
    ],
    "seguro": [
        "🛡️ BBVA ofrece seguros de vida, hogar y vehículo. ¿Cuál te interesa?",
        "📋 Nuestros seguros tienen cobertura desde *$50.000 COP* mensuales."
    ],
    "transferencia": [
        "💸 BBVA te permite transferencias inmediatas a cualquier banco en Colombia.",
        "📱 Puedes hacer transferencias desde la app móvil de BBVA sin costo."
    ],
    "rechazo": [
        "Entendido. ¿En qué más puedo ayudarte?",
        "No hay problema. Estoy aquí para lo que necesites."
    ],
    "general": [
        "🙋 Hola, soy el asistente BBVA Colombia. ¿En qué te ayudo?",
        "🏦 ¡Bienvenido! ¿Necesitas información sobre tu cuenta, tarjeta o préstamos?"
    ]
}

def generar_respuesta(intencion, telefono=None):
    esperar = False
    if intencion == "prestamo":
        esperar = True
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
    elif intencion == "inversion":
        respuesta = random.choice(RESPUESTAS["inversion"])
    elif intencion == "seguro":
        respuesta = random.choice(RESPUESTAS["seguro"])
    elif intencion == "transferencia":
        respuesta = random.choice(RESPUESTAS["transferencia"])
    else:
        respuesta = random.choice(RESPUESTAS["general"])
    
    if telefono:
        actualizar_memoria(telefono, intencion, esperar)
    
    return respuesta

def extraer_texto_pdf(archivo_bytes):
    try:
        import pdfplumber
        import io
        with pdfplumber.open(io.BytesIO(archivo_bytes)) as pdf:
            texto = ""
            for pagina in pdf.pages:
                parte = pagina.extract_text()
                if parte:
                    texto += parte + "\n"
        return texto[:5000] if texto.strip() else "Sin texto extraíble del documento."
    except Exception as e:
        print(f"Error PDF: {e}")
        return f"Error al procesar el PDF: {str(e)}"

# ==================== DB HELPERS MEJORADOS ====================
def guardar_conversacion(telefono, mensaje, intencion, respuesta, escalado, pdf_texto, pdf_nombre, canal, pdf_archivo=None):
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO conversaciones (telefono, mensaje, intencion, respuesta, escalado, pdf_texto, pdf_nombre, canal, fecha, pdf_archivo)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW(), %s)
        """, (telefono, mensaje[:500], intencion, respuesta[:500], escalado, 
              (pdf_texto or "")[:5000], pdf_nombre, canal, pdf_archivo))
        conn.commit()
        cur.close()
        conn.close()
        return True
    except Exception as e:
        print(f"❌ Error guardando: {e}")
        return False

def obtener_conversaciones(limite=100, canal=""):
    try:
        conn = get_db()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        if canal:
            cur.execute("SELECT id, telefono, mensaje, intencion, respuesta, escalado, canal, pdf_nombre, fecha FROM conversaciones WHERE canal=%s ORDER BY id DESC LIMIT %s", (canal, limite))
        else:
            cur.execute("SELECT id, telefono, mensaje, intencion, respuesta, escalado, canal, pdf_nombre, fecha FROM conversaciones ORDER BY id DESC LIMIT %s", (limite,))
        rows = cur.fetchall()
        cur.close()
        conn.close()
        
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

def obtener_estadisticas_reales():
    """Obtiene estadísticas reales de la base de datos"""
    try:
        conn = get_db()
        cur = conn.cursor()
        
        # Totales
        cur.execute("SELECT COUNT(*) FROM conversaciones")
        total = cur.fetchone()[0] or 0
        
        # Escalados
        cur.execute("SELECT COUNT(*) FROM conversaciones WHERE escalado=1")
        escalados = cur.fetchone()[0] or 0
        
        # Intenciones
        cur.execute("SELECT intencion, COUNT(*) FROM conversaciones WHERE intencion IS NOT NULL GROUP BY intencion")
        intenciones = dict(cur.fetchall())
        
        # Canales
        cur.execute("SELECT canal, COUNT(*) FROM conversaciones GROUP BY canal")
        canales = dict(cur.fetchall())
        
        # PDFs
        cur.execute("SELECT COUNT(*) FROM conversaciones WHERE pdf_nombre IS NOT NULL")
        pdfs = cur.fetchone()[0] or 0
        
        # Tendencia últimos 7 días
        cur.execute("""
            SELECT DATE(fecha) as fecha, COUNT(*) 
            FROM conversaciones 
            WHERE fecha >= NOW() - INTERVAL '7 days'
            GROUP BY DATE(fecha) 
            ORDER BY DATE(fecha) ASC
        """)
        tendencia = cur.fetchall()
        
        # Tendencia últimos 30 días
        cur.execute("""
            SELECT DATE(fecha) as fecha, COUNT(*) 
            FROM conversaciones 
            WHERE fecha >= NOW() - INTERVAL '30 days'
            GROUP BY DATE(fecha) 
            ORDER BY DATE(fecha) ASC
        """)
        tendencia_30 = cur.fetchall()
        
        # Actividad por hora
        cur.execute("""
            SELECT EXTRACT(HOUR FROM fecha) as hora, COUNT(*) 
            FROM conversaciones 
            WHERE fecha >= NOW() - INTERVAL '30 days'
            GROUP BY hora 
            ORDER BY hora ASC
        """)
        horas = cur.fetchall()
        
        # Usuarios únicos
        cur.execute("SELECT COUNT(DISTINCT telefono) FROM conversaciones")
        usuarios_unicos = cur.fetchone()[0] or 0
        
        # Top usuarios
        cur.execute("""
            SELECT telefono, COUNT(*) 
            FROM conversaciones 
            GROUP BY telefono 
            ORDER BY COUNT(*) DESC 
            LIMIT 5
        """)
        top_usuarios = cur.fetchall()
        
        # Días de la semana
        cur.execute("""
            SELECT EXTRACT(DOW FROM fecha) as dia, COUNT(*) 
            FROM conversaciones 
            GROUP BY dia 
            ORDER BY dia ASC
        """)
        dias_semana = cur.fetchall()
        
        cur.close()
        conn.close()
        
        return {
            "total": total,
            "escalados": escalados,
            "tasa_escalado": round(escalados/total*100, 1) if total else 0,
            "intenciones": intenciones,
            "canales": canales,
            "pdfs": pdfs,
            "tendencia_fechas": [str(r[0]) for r in tendencia],
            "tendencia_conteos": [r[1] for r in tendencia],
            "tendencia_30_fechas": [str(r[0]) for r in tendencia_30],
            "tendencia_30_conteos": [r[1] for r in tendencia_30],
            "horas": [(int(h[0]), h[1]) for h in horas],
            "usuarios_unicos": usuarios_unicos,
            "top_usuarios": [(u[0], u[1]) for u in top_usuarios],
            "dias_semana": [(int(d[0]), d[1]) for d in dias_semana]
        }
    except Exception as e:
        print(f"❌ Error en estadisticas: {e}")
        return {}

def obtener_insights_reales():
    """Genera insights reales basados en datos"""
    stats = obtener_estadisticas_reales()
    insights = []
    
    if stats:
        # Insight 1: Tasa de escalado
        if stats.get('tasa_escalado', 0) > 15:
            insights.append("⚠️ Alta tasa de escalamiento ({:.1f}%). Recomendamos mejorar la resolución automática.".format(stats['tasa_escalado']))
        elif stats.get('tasa_escalado', 0) < 5:
            insights.append("✅ Excelente tasa de resolución automática ({:.1f}%).".format(stats['tasa_escalado']))
        
        # Insight 2: Intención más común
        intenciones = stats.get('intenciones', {})
        if intenciones:
            top_intencion = max(intenciones.items(), key=lambda x: x[1])
            insights.append(f"📊 La intención más común es '{top_intencion[0]}' con {top_intencion[1]} conversaciones.")
            
            if top_intencion[0] == "saldo":
                insights.append("💡 Los usuarios consultan frecuentemente su saldo. Considera mejorar la visualización de saldos.")
            elif top_intencion[0] == "prestamo":
                insights.append("💡 Alta demanda de préstamos. Ofrece simulaciones rápidas y personalizadas.")
        
        # Insight 3: Canal preferido
        canales = stats.get('canales', {})
        if canales:
            top_canal = max(canales.items(), key=lambda x: x[1])
            if top_canal[0] == "whatsapp":
                insights.append("📱 Los usuarios prefieren WhatsApp. Optimiza la experiencia móvil.")
            elif top_canal[0] == "web":
                insights.append("💻 Los usuarios usan mayormente la web. Asegura una experiencia fluida.")
        
        # Insight 4: Usuarios
        usuarios_unicos = stats.get('usuarios_unicos', 0)
        total = stats.get('total', 0)
        if usuarios_unicos > 0 and total > 0:
            promedio = round(total / usuarios_unicos, 1)
            insights.append(f"👤 {usuarios_unicos} usuarios únicos con un promedio de {promedio} mensajes por usuario.")
        
        # Insight 5: PDFs
        pdfs = stats.get('pdfs', 0)
        if pdfs > 0:
            insights.append(f"📄 Se han procesado {pdfs} documentos PDF. Útil para apelaciones y reclamaciones.")
        
        # Insight 6: Horas pico
        horas = stats.get('horas', [])
        if horas:
            horas_ordenadas = sorted(horas, key=lambda x: x[1], reverse=True)
            if horas_ordenadas:
                hora_pico = horas_ordenadas[0]
                insights.append(f"⏰ Mayor actividad a las {hora_pico[0]:02d}:00 hrs.")
    
    if not insights:
        insights = ["📈 No hay suficientes datos para generar insights. Continúa usando el asistente."]
    
    return insights

# ==================== ENDPOINTS MEJORADOS ====================

@app.get("/api/estadisticas-completas")
async def estadisticas_completas():
    """Obtiene todas las estadísticas reales"""
    return JSONResponse(obtener_estadisticas_reales())

@app.get("/api/insights-reales")
async def insights_reales():
    """Obtiene insights reales basados en datos"""
    return JSONResponse({"insights": obtener_insights_reales()})

@app.get("/api/historial/{telefono}")
async def obtener_historial(telefono: str):
    """Obtiene el historial de un usuario"""
    try:
        conn = get_db()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT mensaje, intencion, respuesta, fecha, escalado 
            FROM conversaciones 
            WHERE telefono = %s 
            ORDER BY id DESC LIMIT 20
        """, (telefono,))
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return JSONResponse({"telefono": telefono, "historial": rows, "total": len(rows)})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/api/analisis-avanzado")
async def analisis_avanzado():
    """Análisis avanzado de conversaciones"""
    try:
        conn = get_db()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT intencion, telefono, mensaje FROM conversaciones")
        conversaciones = cur.fetchall()
        cur.close()
        conn.close()
        
        # Análisis en memoria
        intenciones = {}
        usuarios_por_intencion = {}
        palabras_clave = Counter()
        
        for c in conversaciones:
            intencion = c.get('intencion', 'general')
            telefono = c.get('telefono', 'unknown')
            mensaje = c.get('mensaje', '')
            
            intenciones[intencion] = intenciones.get(intencion, 0) + 1
            
            if intencion not in usuarios_por_intencion:
                usuarios_por_intencion[intencion] = set()
            usuarios_por_intencion[intencion].add(telefono)
            
            # Extraer palabras clave
            palabras = mensaje.lower().split()
            for p in palabras:
                if len(p) > 3:
                    palabras_clave[p] += 1
        
        top_intenciones = sorted(intenciones.items(), key=lambda x: x[1], reverse=True)[:5]
        top_palabras = palabras_clave.most_common(10)
        
        return JSONResponse({
            "total_conversaciones": len(conversaciones),
            "total_intenciones": len(intenciones),
            "top_intenciones": [{"intencion": k, "total": v} for k, v in top_intenciones],
            "top_palabras": [{"palabra": k, "frecuencia": v} for k, v in top_palabras],
            "usuarios_por_intencion": {k: len(v) for k, v in usuarios_por_intencion.items()}
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# ==================== WHATSAPP ====================
async def enviar_whatsapp(telefono: str, texto: str):
    if WHATSAPP_TOKEN == "TU_TOKEN_AQUI":
        print(f"📱 Simulando envío a {telefono}")
        return True
    url = f"https://graph.facebook.com/v18.0/{PHONE_NUMBER_ID}/messages"
    try:
        async with httpx.AsyncClient() as client:
            r = await client.post(url, headers={"Authorization": f"Bearer {WHATSAPP_TOKEN}", "Content-Type": "application/json"},
                                 json={"messaging_product": "whatsapp", "to": telefono, "type": "text", "text": {"body": texto}})
            return r.status_code == 200
    except:
        return False

@app.get("/webhook")
async def verificar_webhook(request: Request):
    p = request.query_params
    if p.get("hub.verify_token") == VERIFY_TOKEN:
        return int(p.get("hub.challenge"))
    return JSONResponse({"error": "Token inválido"}, status_code=403)

@app.post("/webhook")
async def recibir_whatsapp(request: Request):
    body = await request.json()
    try:
        entry = body["entry"][0]["changes"][0]["value"]
        if "messages" not in entry:
            return JSONResponse({"status": "ok"})
        msg = entry["messages"][0]
        if msg.get("type") != "text":
            return JSONResponse({"status": "ok"})
        telefono = msg["from"]
        texto = msg["text"]["body"]
        
        intencion = clasificar_intencion(texto, telefono)
        respuesta = generar_respuesta(intencion, telefono)
        escalado = 1 if intencion in ["asesor", "apelacion"] else 0
        
        guardar_conversacion(telefono, texto, intencion, respuesta, escalado, None, None, "whatsapp", None)
        await enviar_whatsapp(telefono, respuesta)
    except:
        pass
    return JSONResponse({"status": "ok"})

# ==================== API ENDPOINTS ====================
@app.get("/api/conversaciones")
async def api_conversaciones(canal: str = Query(default="")):
    return JSONResponse(obtener_conversaciones(100, canal))

@app.get("/api/estadisticas")
async def api_estadisticas():
    stats = obtener_estadisticas_reales()
    return JSONResponse(stats)

@app.post("/api/simular")
async def simular(telefono: str = Form(...), mensaje: str = Form(...), pdf: UploadFile = File(None)):
    pdf_texto = ""
    pdf_nombre = ""
    pdf_archivo = None
    tiene_pdf = False
    
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
    
    intencion = clasificar_intencion(mensaje, telefono, tiene_pdf)
    emocion = clasificar_emocion(mensaje)
    respuesta = generar_respuesta(intencion, telefono)
    escalado = 1 if intencion in ["asesor", "apelacion"] else 0
    
    guardar_conversacion(telefono, mensaje, intencion, respuesta, escalado, pdf_texto, pdf_nombre, "web", pdf_archivo)
    
    return JSONResponse({
        "respuesta": respuesta,
        "intencion": intencion,
        "emocion": emocion,
        "escalado": escalado,
        "pdf_leido": bool(pdf_texto),
        "pdf_nombre": pdf_nombre
    })

@app.delete("/api/conversaciones")
async def eliminar_todas():
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
    conn = get_db()
    df = pd.read_sql_query("SELECT id, telefono, mensaje, intencion, respuesta, escalado, canal, pdf_nombre, fecha FROM conversaciones ORDER BY id DESC", conn)
    conn.close()
    out = BytesIO()
    with pd.ExcelWriter(out, engine='openpyxl') as w:
        df.to_excel(w, index=False, sheet_name='Conversaciones BBVA')
    out.seek(0)
    return StreamingResponse(out, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           headers={"Content-Disposition": "attachment; filename=conversaciones_bbva.xlsx"})

@app.get("/api/exportar/csv")
async def exportar_csv():
    conn = get_db()
    df = pd.read_sql_query("SELECT id, telefono, mensaje, intencion, respuesta, escalado, canal, pdf_nombre, fecha FROM conversaciones ORDER BY id DESC", conn)
    conn.close()
    buf = StringIO()
    df.to_csv(buf, index=False, encoding='utf-8-sig')
    buf.seek(0)
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv",
                           headers={"Content-Disposition": "attachment; filename=conversaciones_bbva.csv"})

@app.get("/api/descargar-pdf/{conversacion_id}")
async def descargar_pdf(conversacion_id: int):
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT pdf_nombre, pdf_archivo FROM conversaciones WHERE id = %s", (conversacion_id,))
        row = cur.fetchone()
        cur.close()
        conn.close()
        
        if not row or not row[0]:
            return JSONResponse({"error": "No hay PDF asociado"}, status_code=404)
        
        pdf_nombre, pdf_archivo = row
        
        if not pdf_archivo:
            return JSONResponse({"error": "El archivo PDF original no está disponible"}, status_code=404)
        
        return StreamingResponse(
            BytesIO(pdf_archivo), 
            media_type="application/pdf", 
            headers={"Content-Disposition": f"attachment; filename={pdf_nombre}"}
        )
    except Exception as e:
        print(f"❌ Error descargando PDF: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/health")
async def health():
    return JSONResponse({"status": "ok", "version": "3.0.0"})

# ==================== HTML ULTIMATE ====================
HTML = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BBVA Colombia | Asistente Inteligente ULTIMATE</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-datalabels@2"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
    <style>
        :root {
            --bbva-primary: #072146;
            --bbva-secondary: #0A2B5C;
            --bbva-dark: #051833;
            --bbva-gold: #FFD200;
            --bbva-gold-dark: #E6B800;
            --bbva-gold-light: #FFE44D;
            --bbva-blue: #1A6BFF;
            --bbva-cyan: #00D4FF;
            --bbva-green: #00C853;
            --bbva-red: #FF1744;
            --bbva-orange: #FF9100;
            --bbva-purple: #D500F9;
            --glass-bg: rgba(255,255,255,0.04);
            --glass-border: rgba(255,255,255,0.08);
            --text-white: #FFFFFF;
            --text-gray: rgba(255,255,255,0.7);
            --text-dim: rgba(255,255,255,0.35);
            --shadow-glow: 0 8px 48px rgba(0,0,0,0.5);
            --radius-xl: 28px;
            --radius-lg: 20px;
            --radius-md: 14px;
            --radius-sm: 10px;
            --transition: all 0.4s cubic-bezier(0.25,0.46,0.45,0.94);
        }

        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: 'Inter', -apple-system, sans-serif;
            background: var(--bbva-dark);
            min-height: 100vh;
            overflow-x: hidden;
            color: var(--text-white);
        }

        .bg-premium {
            position: fixed;
            inset: 0;
            z-index: 0;
            background: 
                radial-gradient(ellipse 80% 60% at 0% 0%, rgba(26,107,255,0.12) 0%, transparent 60%),
                radial-gradient(ellipse 60% 50% at 100% 100%, rgba(255,210,0,0.06) 0%, transparent 50%),
                radial-gradient(ellipse 40% 40% at 50% 50%, rgba(0,212,255,0.03) 0%, transparent 40%);
            pointer-events: none;
        }

        .bg-grid {
            position: fixed;
            inset: 0;
            z-index: 0;
            background-image: 
                linear-gradient(rgba(255,255,255,0.02) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255,255,255,0.02) 1px, transparent 1px);
            background-size: 80px 80px;
            pointer-events: none;
            mask-image: radial-gradient(ellipse at center, black 20%, transparent 70%);
        }

        .bg-orb {
            position: fixed;
            border-radius: 50%;
            filter: blur(120px);
            pointer-events: none;
            z-index: 0;
        }
        .bg-orb-1 { width: 600px; height: 600px; top: -200px; right: -200px; background: rgba(255,210,0,0.03); }
        .bg-orb-2 { width: 500px; height: 500px; bottom: -150px; left: -150px; background: rgba(26,107,255,0.03); }

        .container {
            position: relative;
            z-index: 1;
            max-width: 1500px;
            margin: 0 auto;
            padding: 16px 24px;
        }

        /* HEADER */
        .header-premium {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 16px;
            padding: 14px 24px;
            margin-bottom: 20px;
            border-radius: var(--radius-xl);
            background: var(--glass-bg);
            backdrop-filter: blur(24px);
            border: 1px solid var(--glass-border);
            position: relative;
            overflow: hidden;
        }

        .header-premium::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 2px;
            background: linear-gradient(90deg, transparent, var(--bbva-gold), var(--bbva-blue), transparent);
            opacity: 0.5;
        }

        .logo-premium {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .logo-icon-premium {
            width: 48px;
            height: 48px;
            background: linear-gradient(135deg, var(--bbva-gold), var(--bbva-gold-dark));
            border-radius: var(--radius-sm);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
            box-shadow: 0 0 40px rgba(255,210,0,0.15);
        }

        .logo-text-premium {
            font-size: 22px;
            font-weight: 800;
            background: linear-gradient(135deg, #FFFFFF, rgba(255,255,255,0.6));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .logo-sub-premium {
            font-size: 9px;
            color: var(--text-dim);
            font-weight: 500;
            letter-spacing: 2.5px;
            text-transform: uppercase;
            font-family: 'JetBrains Mono', monospace;
        }

        .header-right-premium {
            display: flex;
            align-items: center;
            gap: 14px;
            flex-wrap: wrap;
        }

        .status-premium {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 6px 16px;
            background: rgba(0,200,83,0.08);
            border: 1px solid rgba(0,200,83,0.12);
            border-radius: 50px;
            font-size: 11px;
            font-weight: 600;
            color: var(--bbva-green);
            font-family: 'JetBrains Mono', monospace;
        }

        .status-dot-premium {
            width: 7px;
            height: 7px;
            background: var(--bbva-green);
            border-radius: 50%;
            animation: pulse-dot 2s ease-in-out infinite;
            box-shadow: 0 0 16px rgba(0,200,83,0.2);
        }

        @keyframes pulse-dot {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.4; transform: scale(0.7); }
        }

        .clock-premium {
            text-align: right;
            padding: 2px 0;
        }

        .clock-time-premium {
            font-family: 'JetBrains Mono', monospace;
            font-size: 26px;
            font-weight: 700;
            color: var(--bbva-gold);
            letter-spacing: 2px;
            line-height: 1;
            text-shadow: 0 0 30px rgba(255,210,0,0.08);
        }

        .clock-date-premium {
            font-size: 9px;
            color: var(--text-dim);
            margin-top: 2px;
            font-weight: 500;
            letter-spacing: 1px;
        }

        /* TABS */
        .tabs-premium {
            display: flex;
            gap: 3px;
            margin-bottom: 20px;
            padding: 5px;
            background: rgba(0,0,0,0.3);
            border-radius: var(--radius-md);
            border: 1px solid var(--glass-border);
            backdrop-filter: blur(12px);
            width: fit-content;
            flex-wrap: wrap;
        }

        .tab-premium {
            padding: 9px 24px;
            border: none;
            border-radius: var(--radius-sm);
            background: transparent;
            color: var(--text-gray);
            font-family: 'Inter', sans-serif;
            font-size: 12px;
            font-weight: 600;
            cursor: pointer;
            transition: var(--transition);
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .tab-premium i { font-size: 13px; }

        .tab-premium:hover:not(.active) {
            color: white;
            background: rgba(255,255,255,0.04);
        }

        .tab-premium.active {
            background: linear-gradient(135deg, var(--bbva-secondary), var(--bbva-primary));
            color: white;
            box-shadow: 0 4px 20px rgba(26,107,255,0.2);
        }

        /* CARDS */
        .card-premium {
            background: var(--glass-bg);
            backdrop-filter: blur(24px);
            border: 1px solid var(--glass-border);
            border-radius: var(--radius-xl);
            padding: 24px 28px;
            margin-bottom: 20px;
            transition: var(--transition);
            position: relative;
            overflow: hidden;
        }

        .card-premium::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 1px;
            background: linear-gradient(90deg, transparent, var(--bbva-gold), transparent);
            opacity: 0.15;
        }

        .card-premium:hover {
            border-color: rgba(255,255,255,0.12);
            box-shadow: var(--shadow-glow);
            transform: translateY(-1px);
        }

        .card-badge {
            font-size: 8px;
            font-weight: 600;
            letter-spacing: 2.5px;
            text-transform: uppercase;
            color: var(--bbva-gold);
            font-family: 'JetBrains Mono', monospace;
            margin-bottom: 2px;
        }

        .card-title-premium {
            font-size: 18px;
            font-weight: 700;
            margin-bottom: 14px;
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .card-title-premium .accent-bar {
            width: 3px;
            height: 24px;
            background: linear-gradient(180deg, var(--bbva-gold), var(--bbva-blue));
            border-radius: 4px;
        }

        .divider-premium {
            height: 1px;
            margin: 4px 0 16px;
            background: linear-gradient(90deg, transparent, rgba(255,210,0,0.15), transparent);
        }

        /* STATS */
        .stats-premium {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 12px;
            margin-bottom: 20px;
        }

        .stat-premium {
            background: rgba(255,255,255,0.02);
            border: 1px solid var(--glass-border);
            border-radius: var(--radius-md);
            padding: 18px 20px;
            transition: var(--transition);
            position: relative;
            overflow: hidden;
        }

        .stat-premium::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 2.5px;
            background: linear-gradient(90deg, var(--bbva-gold), var(--bbva-blue));
            opacity: 0;
            transition: var(--transition);
        }

        .stat-premium:hover::before { opacity: 1; }
        .stat-premium:hover {
            transform: translateY(-3px);
            border-color: rgba(255,255,255,0.12);
            box-shadow: var(--shadow-glow);
        }

        .stat-number-premium {
            font-size: 30px;
            font-weight: 800;
            color: white;
            font-family: 'JetBrains Mono', monospace;
            line-height: 1;
        }

        .stat-label-premium {
            font-size: 9px;
            color: var(--text-gray);
            margin-top: 4px;
            font-weight: 500;
            letter-spacing: 0.5px;
        }

        .stat-icon-premium {
            position: absolute;
            right: 12px;
            bottom: 12px;
            font-size: 28px;
            opacity: 0.05;
        }

        /* CHAT */
        .chat-premium {
            height: 400px;
            overflow-y: auto;
            padding: 20px;
            background: rgba(0,0,0,0.25);
            border-radius: var(--radius-md);
            border: 1px solid var(--glass-border);
            margin-bottom: 14px;
        }

        .chat-premium::-webkit-scrollbar { width: 4px; }
        .chat-premium::-webkit-scrollbar-track { background: transparent; }
        .chat-premium::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.12); border-radius: 10px; }

        .message-premium {
            display: flex;
            margin-bottom: 14px;
            animation: fade-slide 0.35s ease;
        }

        @keyframes fade-slide {
            from { opacity: 0; transform: translateY(12px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .message-premium.user { justify-content: flex-end; }

        .avatar-premium {
            width: 36px;
            height: 36px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 15px;
            flex-shrink: 0;
        }

        .avatar-bot-premium {
            background: linear-gradient(135deg, var(--bbva-gold), var(--bbva-gold-dark));
            color: var(--bbva-dark);
            margin-right: 12px;
        }

        .avatar-user-premium {
            background: linear-gradient(135deg, var(--bbva-blue), var(--bbva-cyan));
            color: white;
            margin-left: 12px;
        }

        .bubble-premium {
            max-width: 72%;
            padding: 12px 18px;
            border-radius: var(--radius-md);
            font-size: 13px;
            line-height: 1.6;
            word-wrap: break-word;
        }

        .message-premium.user .bubble-premium {
            background: linear-gradient(135deg, var(--bbva-blue), var(--bbva-secondary));
            color: white;
            border-bottom-right-radius: 4px;
        }

        .message-premium.bot .bubble-premium {
            background: rgba(255,255,255,0.05);
            color: var(--text-white);
            border: 1px solid var(--glass-border);
            border-bottom-left-radius: 4px;
        }

        .bubble-premium .intent-tag-premium {
            display: inline-block;
            margin-top: 6px;
            font-size: 8px;
            font-family: 'JetBrains Mono', monospace;
            color: var(--text-dim);
            letter-spacing: 0.5px;
        }

        /* INPUT */
        .input-premium {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            align-items: flex-end;
        }

        .input-wrapper-premium { flex: 1; min-width: 130px; }

        .input-label-premium {
            font-size: 8px;
            font-weight: 600;
            color: var(--text-dim);
            text-transform: uppercase;
            letter-spacing: 1.5px;
            margin-bottom: 4px;
            display: block;
        }

        .input-premium-field, .textarea-premium, .filter-premium {
            width: 100%;
            padding: 10px 14px;
            background: rgba(255,255,255,0.03);
            border: 1px solid var(--glass-border);
            border-radius: var(--radius-sm);
            color: white;
            font-family: 'Inter', sans-serif;
            font-size: 13px;
            transition: var(--transition);
        }

        .input-premium-field:focus, .textarea-premium:focus, .filter-premium:focus {
            outline: none;
            border-color: var(--bbva-gold);
            box-shadow: 0 0 0 4px rgba(255,210,0,0.04);
        }

        .filter-premium {
            appearance: none;
            cursor: pointer;
            color: var(--text-gray);
            min-width: 120px;
        }

        .filter-premium option { background: var(--bbva-dark); color: white; }
        .textarea-premium { resize: vertical; min-height: 38px; font-family: 'Inter', sans-serif; }

        /* BUTTONS */
        .btn-premium {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 10px 20px;
            border: none;
            border-radius: var(--radius-sm);
            font-family: 'Inter', sans-serif;
            font-size: 12px;
            font-weight: 600;
            cursor: pointer;
            transition: var(--transition);
            white-space: nowrap;
            position: relative;
            overflow: hidden;
        }

        .btn-premium::after {
            content: '';
            position: absolute;
            inset: 0;
            background: linear-gradient(135deg, rgba(255,255,255,0.08), transparent);
            opacity: 0;
            transition: var(--transition);
        }

        .btn-premium:hover::after { opacity: 1; }
        .btn-premium:hover { transform: translateY(-2px); }
        .btn-premium:active { transform: scale(0.97); }

        .btn-primary-premium {
            background: linear-gradient(135deg, var(--bbva-gold), var(--bbva-gold-dark));
            color: var(--bbva-dark);
            box-shadow: 0 4px 20px rgba(255,210,0,0.15);
        }

        .btn-primary-premium:hover { box-shadow: 0 8px 30px rgba(255,210,0,0.25); }

        .btn-secondary-premium {
            background: var(--glass-bg);
            color: white;
            border: 1px solid var(--glass-border);
        }

        .btn-secondary-premium:hover { background: rgba(255,255,255,0.06); }

        .btn-success-premium {
            background: linear-gradient(135deg, #00C853, #00E676);
            color: white;
            box-shadow: 0 4px 20px rgba(0,200,83,0.15);
        }

        .btn-danger-premium {
            background: linear-gradient(135deg, #D50000, #FF1744);
            color: white;
            box-shadow: 0 4px 20px rgba(255,23,68,0.15);
        }

        .btn-sm-premium { padding: 6px 14px; font-size: 10px; border-radius: var(--radius-sm); }

        .file-btn-premium {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 10px 16px;
            background: rgba(255,255,255,0.02);
            border: 1.5px dashed var(--glass-border);
            border-radius: var(--radius-sm);
            color: var(--text-gray);
            cursor: pointer;
            font-family: 'Inter', sans-serif;
            font-size: 12px;
            font-weight: 500;
            transition: var(--transition);
        }

        .file-btn-premium:hover {
            border-color: var(--bbva-gold);
            color: white;
            background: rgba(255,210,0,0.03);
        }

        input[type="file"] { display: none; }

        .toolbar-premium {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            margin-bottom: 14px;
            align-items: center;
        }

        /* TABLE */
        .table-premium-wrapper {
            overflow-x: auto;
            border-radius: var(--radius-md);
            border: 1px solid var(--glass-border);
        }

        .table-premium {
            width: 100%;
            border-collapse: collapse;
            font-size: 12px;
        }

        .table-premium th {
            background: rgba(255,255,255,0.02);
            color: var(--text-gray);
            font-weight: 600;
            font-size: 9px;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            padding: 12px 14px;
            text-align: left;
            border-bottom: 1px solid var(--glass-border);
        }

        .table-premium td {
            padding: 10px 14px;
            border-bottom: 1px solid rgba(255,255,255,0.02);
            color: var(--text-gray);
            font-size: 12px;
        }

        .table-premium tr:hover td { background: rgba(255,255,255,0.015); }

        /* BADGES */
        .badge-premium {
            display: inline-flex;
            align-items: center;
            gap: 4px;
            padding: 2px 10px;
            border-radius: 5px;
            font-size: 9px;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
        }

        .badge-saldo { background: rgba(0,200,83,0.12); color: #69F0AE; }
        .badge-prestamo { background: rgba(26,107,255,0.12); color: #82B1FF; }
        .badge-apelacion { background: rgba(255,210,0,0.12); color: #FFD740; }
        .badge-asesor { background: rgba(255,23,68,0.12); color: #FF5252; }
        .badge-tarjeta { background: rgba(213,0,249,0.12); color: #EA80FC; }
        .badge-general { background: rgba(255,255,255,0.04); color: var(--text-gray); }
        .badge-inversion { background: rgba(0,212,255,0.12); color: #4DD0E1; }
        .badge-seguro { background: rgba(255,145,0,0.12); color: #FFAB40; }
        .badge-transferencia { background: rgba(213,0,249,0.12); color: #EA80FC; }

        .badge-pdf-premium {
            display: inline-flex;
            align-items: center;
            gap: 4px;
            padding: 3px 10px;
            background: rgba(255,23,68,0.10);
            border-radius: 5px;
            color: #FF5252;
            font-size: 9px;
            font-weight: 600;
            cursor: pointer;
            transition: var(--transition);
            border: none;
            font-family: 'Inter', sans-serif;
        }

        .badge-pdf-premium:hover {
            background: rgba(255,23,68,0.18);
            transform: translateY(-1px);
        }

        /* CHARTS */
        .charts-premium {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(380px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }

        .chart-premium-container {
            background: rgba(255,255,255,0.02);
            border: 1px solid var(--glass-border);
            border-radius: var(--radius-md);
            padding: 20px;
        }

        .chart-premium-container h3 {
            font-size: 12px;
            font-weight: 600;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 8px;
            color: var(--text-gray);
        }

        canvas { max-height: 220px; }

        .loader-premium {
            display: inline-block;
            width: 16px;
            height: 16px;
            border: 2px solid rgba(255,255,255,0.08);
            border-top-color: var(--bbva-gold);
            border-radius: 50%;
            animation: spin 0.7s linear infinite;
            margin-right: 6px;
        }

        @keyframes spin { to { transform: rotate(360deg); } }

        .empty-premium {
            text-align: center;
            padding: 40px 16px;
            color: var(--text-dim);
        }

        .empty-premium i {
            font-size: 40px;
            margin-bottom: 12px;
            display: block;
            opacity: 0.15;
        }

        .footer-premium {
            text-align: center;
            padding: 16px;
            color: var(--text-dim);
            font-size: 10px;
            font-family: 'JetBrains Mono', monospace;
            letter-spacing: 1.5px;
            border-top: 1px solid var(--glass-border);
            margin-top: 4px;
        }

        /* INSIGHTS */
        .insight-box {
            background: rgba(255,210,0,0.04);
            border: 1px solid rgba(255,210,0,0.1);
            border-radius: var(--radius-md);
            padding: 16px 20px;
            margin-bottom: 16px;
        }

        .insight-box ul {
            list-style: none;
            padding: 0;
        }

        .insight-box li {
            padding: 8px 0;
            color: var(--text-gray);
            font-size: 13px;
            border-bottom: 1px solid rgba(255,255,255,0.03);
        }

        .insight-box li:last-child { border-bottom: none; }
        .insight-box i { margin-right: 10px; }

        .insight-stat {
            background: rgba(255,255,255,0.02);
            border: 1px solid var(--glass-border);
            border-radius: var(--radius-md);
            padding: 14px 18px;
            text-align: center;
        }

        .insight-stat .num {
            font-size: 24px;
            font-weight: 800;
            font-family: 'JetBrains Mono', monospace;
            color: white;
        }

        .insight-stat .lbl {
            font-size: 9px;
            color: var(--text-dim);
            margin-top: 2px;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        /* WA STATS */
        .wa-premium-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 10px;
            margin-bottom: 14px;
        }

        .wa-premium-stat {
            background: rgba(255,255,255,0.02);
            border: 1px solid var(--glass-border);
            border-radius: var(--radius-md);
            padding: 14px;
            text-align: center;
        }

        .wa-premium-stat .num {
            font-size: 22px;
            font-weight: 800;
            font-family: 'JetBrains Mono', monospace;
            color: white;
        }

        .wa-premium-stat .lbl {
            font-size: 9px;
            color: var(--text-dim);
            margin-top: 2px;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .wa-premium-stat.whatsapp .num { color: #25D366; }
        .wa-premium-stat.escalados .num { color: #FF5252; }

        @media (max-width: 768px) {
            .container { padding: 10px 12px; }
            .header-premium { padding: 12px 16px; flex-direction: column; align-items: stretch; }
            .header-right-premium { justify-content: space-between; flex-wrap: wrap; }
            .tabs-premium { width: 100%; overflow-x: auto; flex-wrap: nowrap; }
            .tab-premium { padding: 6px 14px; font-size: 11px; white-space: nowrap; }
            .stats-premium { grid-template-columns: repeat(2, 1fr); }
            .charts-premium { grid-template-columns: 1fr; }
            .bubble-premium { max-width: 88%; }
            .input-premium { flex-direction: column; }
            .input-wrapper-premium { min-width: 100%; }
            .stat-number-premium { font-size: 24px; }
            .card-premium { padding: 16px; }
            .clock-time-premium { font-size: 20px; }
        }

        @media (max-width: 480px) {
            .stats-premium { grid-template-columns: 1fr; }
            .logo-text-premium { font-size: 16px; }
            .logo-icon-premium { width: 36px; height: 36px; font-size: 18px; }
        }
    </style>
</head>
<body>

<div class="bg-premium"></div>
<div class="bg-grid"></div>
<div class="bg-orb bg-orb-1"></div>
<div class="bg-orb bg-orb-2"></div>

<div class="container">

    <!-- HEADER -->
    <header class="header-premium">
        <div class="logo-premium">
            <div class="logo-icon-premium">🏦</div>
            <div>
                <div class="logo-text-premium">BBVA Colombia</div>
                <div class="logo-sub-premium">Asistente ULTIMATE · v3.0</div>
            </div>
        </div>
        <div class="header-right-premium">
            <div class="status-premium">
                <span class="status-dot-premium"></span>
                Sistema Activo
            </div>
            <div class="clock-premium">
                <div class="clock-time-premium" id="clockTimePremium">--:--:--</div>
                <div class="clock-date-premium" id="clockDatePremium">--/--/----</div>
            </div>
        </div>
    </header>

    <!-- TABS -->
    <div class="tabs-premium">
        <button class="tab-premium active" onclick="showTabPremium('chat')">
            <i class="fas fa-comment-dots"></i> Chat
        </button>
        <button class="tab-premium" onclick="showTabPremium('whatsapp')">
            <i class="fab fa-whatsapp"></i> WhatsApp
        </button>
        <button class="tab-premium" onclick="showTabPremium('dashboard')">
            <i class="fas fa-chart-pie"></i> Dashboard
        </button>
        <button class="tab-premium" onclick="showTabPremium('analytics')">
            <i class="fas fa-brain"></i> Analytics
        </button>
        <button class="tab-premium" onclick="showTabPremium('insights')">
            <i class="fas fa-lightbulb"></i> Insights
        </button>
    </div>

    <!-- CHAT -->
    <div id="chat-section-premium">
        <div class="card-premium">
            <div class="card-badge">🤖 Módulo Activo</div>
            <div class="card-title-premium">
                <span class="accent-bar"></span>
                Simulador de Conversación
                <span style="font-size:10px;color:var(--text-dim);font-weight:400;margin-left:auto;">
                    <i class="fas fa-brain"></i> ML v3.0
                </span>
            </div>
            <div class="divider-premium"></div>

            <div class="chat-premium" id="chatMessagesPremium">
                <div class="message-premium bot">
                    <div class="avatar-premium avatar-bot-premium"><i class="fas fa-robot"></i></div>
                    <div>
                        <div class="bubble-premium">
                            👋 ¡Bienvenido al <strong>Asistente BBVA Colombia</strong>!<br><br>
                            🚀 <strong>Servicios disponibles:</strong><br><br>
                            <strong>💰 Saldo</strong> · <strong>🏦 Préstamo</strong> · <strong>💳 Tarjeta</strong><br>
                            <strong>📈 Inversión</strong> · <strong>🛡️ Seguro</strong> · <strong>💸 Transferencia</strong><br>
                            <strong>📄 Apelación</strong> · <strong>👨‍💼 Asesor</strong><br><br>
                            💡 <em>¿En qué te ayudo hoy?</em>
                            <div class="intent-tag-premium">🤖 BBVA-ML · v3.0 · listo</div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="input-premium">
                <div class="input-wrapper-premium" style="max-width:150px">
                    <label class="input-label-premium">📱 Teléfono</label>
                    <input class="input-premium-field" type="text" id="telefonoPremium" placeholder="573053667704" value="573053667704">
                </div>
                <div class="input-wrapper-premium">
                    <label class="input-label-premium">💬 Mensaje</label>
                    <textarea class="textarea-premium" id="mensajePremium" placeholder="Escribe tu mensaje aquí…" rows="2"></textarea>
                </div>
                <label class="file-btn-premium" for="pdfFilePremium">
                    <i class="fas fa-paperclip"></i> PDF
                </label>
                <input type="file" id="pdfFilePremium" accept="application/pdf">
                <button class="btn-premium btn-primary-premium" onclick="enviarMensajePremium()">
                    <i class="fas fa-paper-plane"></i> Enviar
                </button>
                <button class="btn-premium btn-secondary-premium" onclick="limpiarChatPremium()">
                    <i class="fas fa-eraser"></i> Limpiar
                </button>
            </div>
        </div>
    </div>

    <!-- WHATSAPP -->
    <div id="whatsapp-section-premium" style="display:none">
        <div class="card-premium">
            <div class="card-badge">📱 Canal Externo</div>
            <div class="card-title-premium">
                <span class="accent-bar"></span>
                <i class="fab fa-whatsapp" style="color:#25D366"></i> Mensajes WhatsApp
            </div>
            <div class="divider-premium"></div>

            <div class="toolbar-premium">
                <button class="btn-premium btn-primary-premium btn-sm-premium" onclick="cargarWhatsAppPremium()">
                    <i class="fas fa-sync-alt"></i> Actualizar
                </button>
                <button class="btn-premium btn-success-premium btn-sm-premium" onclick="exportarExcelPremium()">
                    <i class="fas fa-file-excel"></i> Excel
                </button>
                <button class="btn-premium btn-secondary-premium btn-sm-premium" onclick="exportarCSVPremium()">
                    <i class="fas fa-file-csv"></i> CSV
                </button>
            </div>

            <div id="waStatsPremium"></div>
            <div id="waTablaPremium"></div>
        </div>
    </div>

    <!-- DASHBOARD -->
    <div id="dashboard-section-premium" style="display:none">
        <div id="statsContainerPremium" class="stats-premium"></div>

        <div class="charts-premium">
            <div class="chart-premium-container">
                <h3><i class="fas fa-chart-line" style="color:var(--bbva-gold)"></i> Tendencia (Últimos 7 días)</h3>
                <canvas id="trendChartPremium"></canvas>
            </div>
            <div class="chart-premium-container">
                <h3><i class="fas fa-chart-pie" style="color:var(--bbva-cyan)"></i> Distribución de Intenciones</h3>
                <canvas id="intencionesChartPremium"></canvas>
            </div>
        </div>

        <div class="card-premium">
            <div class="card-badge">📊 Registro en Tiempo Real</div>
            <div class="card-title-premium">
                <span class="accent-bar"></span>
                Conversaciones Recientes
            </div>
            <div class="divider-premium"></div>

            <div class="toolbar-premium">
                <select class="filter-premium" id="filtroCanalPremium" onchange="cargarDashboardPremium()">
                    <option value="">🌐 Todos</option>
                    <option value="web">💻 Web</option>
                    <option value="whatsapp">📱 WhatsApp</option>
                </select>
                <button class="btn-premium btn-success-premium btn-sm-premium" onclick="exportarExcelPremium()">
                    <i class="fas fa-file-excel"></i> Excel
                </button>
                <button class="btn-premium btn-secondary-premium btn-sm-premium" onclick="exportarCSVPremium()">
                    <i class="fas fa-file-csv"></i> CSV
                </button>
                <button class="btn-premium btn-danger-premium btn-sm-premium" onclick="borrarTodoPremium()">
                    <i class="fas fa-trash-alt"></i> Borrar todo
                </button>
                <button class="btn-premium btn-primary-premium btn-sm-premium" onclick="cargarDashboardPremium()">
                    <i class="fas fa-sync-alt"></i> Actualizar
                </button>
            </div>

            <div id="tablaConversacionesPremium"></div>
        </div>
    </div>

    <!-- ANALYTICS -->
    <div id="analytics-section-premium" style="display:none">
        <div class="card-premium">
            <div class="card-badge">🧠 Inteligencia de Datos</div>
            <div class="card-title-premium">
                <span class="accent-bar"></span>
                Analytics Avanzado
            </div>
            <div class="divider-premium"></div>

            <div id="analyticsStats" class="stats-premium"></div>

            <div class="charts-premium">
                <div class="chart-premium-container">
                    <h3><i class="fas fa-clock" style="color:var(--bbva-gold)"></i> Actividad por Hora</h3>
                    <canvas id="hourlyChart"></canvas>
                </div>
                <div class="chart-premium-container">
                    <h3><i class="fas fa-users" style="color:var(--bbva-cyan)"></i> Canales de Atención</h3>
                    <canvas id="channelChart"></canvas>
                </div>
            </div>

            <div class="charts-premium">
                <div class="chart-premium-container">
                    <h3><i class="fas fa-calendar-week" style="color:var(--bbva-green)"></i> Actividad por Día</h3>
                    <canvas id="diasChart"></canvas>
                </div>
                <div class="chart-premium-container">
                    <h3><i class="fas fa-trend-up" style="color:var(--bbva-gold)"></i> Tendencia (30 días)</h3>
                    <canvas id="trend30Chart"></canvas>
                </div>
            </div>
        </div>
    </div>

    <!-- INSIGHTS -->
    <div id="insights-section-premium" style="display:none">
        <div class="card-premium">
            <div class="card-badge">💡 Inteligencia de Negocios</div>
            <div class="card-title-premium">
                <span class="accent-bar"></span>
                Insights y Recomendaciones
            </div>
            <div class="divider-premium"></div>

            <div id="insightsContainer"></div>

            <div id="insightsStats" class="stats-premium"></div>

            <div class="charts-premium">
                <div class="chart-premium-container">
                    <h3><i class="fas fa-users" style="color:var(--bbva-cyan)"></i> Top Usuarios</h3>
                    <canvas id="topUsuariosChart"></canvas>
                </div>
                <div class="chart-premium-container">
                    <h3><i class="fas fa-wordpress" style="color:var(--bbva-gold)"></i> Palabras Clave</h3>
                    <canvas id="palabrasChart"></canvas>
                </div>
            </div>
        </div>
    </div>

    <footer class="footer-premium">
        🏦 BBVA Colombia · Asistente ULTIMATE · Datos en tiempo real · ML + IA
    </footer>
</div>

<script>
    Chart.register(ChartDataLabels);
    let trendChartPremium, intencionesChartPremium, hourlyChartPremium, channelChartPremium;
    let diasChartPremium, trend30ChartPremium, topUsuariosChartPremium, palabrasChartPremium;

    // ── RELOJ ──
    function updateClockPremium() {
        const now = new Date();
        document.getElementById('clockTimePremium').textContent =
            now.toLocaleTimeString('es-CO', { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false });
        document.getElementById('clockDatePremium').textContent =
            now.toLocaleDateString('es-CO', { weekday: 'short', day: '2-digit', month: 'short', year: 'numeric' }).toUpperCase();
    }
    updateClockPremium();
    setInterval(updateClockPremium, 1000);

    // ── TABS ──
    function showTabPremium(tab) {
        ['chat', 'whatsapp', 'dashboard', 'analytics', 'insights'].forEach(t => {
            document.getElementById(t + '-section-premium').style.display = t === tab ? '' : 'none';
        });
        document.querySelectorAll('.tab-premium').forEach((el, i) => {
            const tabs = ['chat', 'whatsapp', 'dashboard', 'analytics', 'insights'];
            el.classList.toggle('active', tabs[i] === tab);
        });
        if (tab === 'dashboard') cargarDashboardPremium();
        if (tab === 'whatsapp') cargarWhatsAppPremium();
        if (tab === 'analytics') cargarAnalyticsPremium();
        if (tab === 'insights') cargarInsightsPremium();
    }

    function escapeHtml(text) {
        if (!text) return '';
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // ── CHAT ──
    async function enviarMensajePremium() {
        const telefono = document.getElementById('telefonoPremium').value.trim();
        const mensaje = document.getElementById('mensajePremium').value.trim();
        const pdfFile = document.getElementById('pdfFilePremium').files[0];

        if (!mensaje && !pdfFile) {
            alert('✏️ Escribe un mensaje o adjunta un PDF');
            return;
        }

        const chatDiv = document.getElementById('chatMessagesPremium');
        let userMsg = mensaje || '📎 Documento adjunto';
        if (pdfFile) userMsg += `<br><i class="fas fa-paperclip"></i> ${escapeHtml(pdfFile.name)}`;

        chatDiv.innerHTML += `
            <div class="message-premium user">
                <div><div class="bubble-premium">${userMsg}</div></div>
                <div class="avatar-premium avatar-user-premium"><i class="fas fa-user"></i></div>
            </div>
        `;
        chatDiv.scrollTop = chatDiv.scrollHeight;

        const fd = new FormData();
        fd.append('telefono', telefono);
        fd.append('mensaje', mensaje || '📎 Adjunto PDF');
        if (pdfFile) fd.append('pdf', pdfFile);

        document.getElementById('mensajePremium').value = '';
        document.getElementById('pdfFilePremium').value = '';

        const loading = document.createElement('div');
        loading.className = 'message-premium bot';
        loading.innerHTML = `
            <div class="avatar-premium avatar-bot-premium"><i class="fas fa-robot"></i></div>
            <div><div class="bubble-premium"><span class="loader-premium"></span> Procesando con ML...</div></div>
        `;
        chatDiv.appendChild(loading);
        chatDiv.scrollTop = chatDiv.scrollHeight;

        try {
            const res = await fetch('/api/simular', { method: 'POST', body: fd });
            const data = await res.json();
            loading.remove();

            const escaladoBadge = data.escalado ? '<span style="color:#FF5252;margin-left:8px">⚠️ Escalado</span>' : '';
            const emocionBadge = data.emocion ? `<span style="color:var(--bbva-cyan);margin-left:8px">${data.emocion}</span>` : '';

            chatDiv.innerHTML += `
                <div class="message-premium bot">
                    <div class="avatar-premium avatar-bot-premium"><i class="fas fa-robot"></i></div>
                    <div>
                        <div class="bubble-premium">
                            🤖 ${escapeHtml(data.respuesta)}
                            <div class="intent-tag-premium">
                                <i class="fas fa-tag"></i> ${data.intencion}
                                ${escaladoBadge}
                                ${emocionBadge}
                                ${data.pdf_nombre ? ` · 📄 ${escapeHtml(data.pdf_nombre)}` : ''}
                            </div>
                        </div>
                    </div>
                </div>
            `;
            chatDiv.scrollTop = chatDiv.scrollHeight;
            if (data.escalado) cargarDashboardPremium();

        } catch (e) {
            loading.remove();
            chatDiv.innerHTML += `
                <div class="message-premium bot">
                    <div class="avatar-premium avatar-bot-premium"><i class="fas fa-robot"></i></div>
                    <div><div class="bubble-premium" style="border-color:#FF5252">
                        <i class="fas fa-exclamation-circle" style="color:#FF5252"></i> Error: ${e.message}
                    </div></div>
                </div>
            `;
        }
    }

    function limpiarChatPremium() {
        document.getElementById('chatMessagesPremium').innerHTML = `
            <div class="message-premium bot">
                <div class="avatar-premium avatar-bot-premium"><i class="fas fa-robot"></i></div>
                <div><div class="bubble-premium">
                    🧹 Chat limpiado. ¿En qué te ayudo?
                    <div class="intent-tag-premium">🤖 BBVA-ML · v3.0 · listo</div>
                </div></div>
            </div>
        `;
    }

    // ── WHATSAPP ──
    async function cargarWhatsAppPremium() {
        try {
            const res = await fetch('/api/conversaciones?canal=whatsapp');
            const data = await res.json();
            const esc = data.filter(c => c.escalado).length;

            document.getElementById('waStatsPremium').innerHTML = `
                <div class="wa-premium-grid">
                    <div class="wa-premium-stat whatsapp"><div class="num">📱 ${data.length}</div><div class="lbl">Mensajes WhatsApp</div></div>
                    <div class="wa-premium-stat escalados"><div class="num">🚨 ${esc}</div><div class="lbl">Escalados</div></div>
                    <div class="wa-premium-stat"><div class="num">${data.length > 0 ? Math.round((data.length-esc)/data.length*100) : 0}%</div><div class="lbl">Tasa de Resolución</div></div>
                </div>
            `;

            if (!data.length) {
                document.getElementById('waTablaPremium').innerHTML = `<div class="empty-premium"><i class="fas fa-inbox"></i>Sin mensajes de WhatsApp aún</div>`;
                return;
            }

            let h = `<div class="table-premium-wrapper"><table class="table-premium"><thead><tr>
                <th>Fecha</th><th>Teléfono</th><th>Mensaje</th><th>Intención</th>
            </tr></thead><tbody>`;

            for (let c of data.slice(0, 30)) {
                const fecha = c.fecha ? c.fecha.substring(0, 16).replace('T', ' ') : 'Sin fecha';
                h += `<tr>
                    <td>${fecha}</td>
                    <td>${c.telefono}</td>
                    <td>${escapeHtml((c.mensaje || '').substring(0, 40))}</td>
                    <td><span class="badge-premium badge-${c.intencion || 'general'}">${c.intencion || 'general'}</span></td>
                </tr>`;
            }
            h += '</tbody></table></div>';
            document.getElementById('waTablaPremium').innerHTML = h;

        } catch (e) { console.error(e); }
    }

    // ── DASHBOARD ──
    async function cargarDashboardPremium() {
        try {
            const s = await (await fetch('/api/estadisticas-completas')).json();

            document.getElementById('statsContainerPremium').innerHTML = `
                <div class="stat-premium"><div class="stat-number-premium">${s.total || 0}</div><div class="stat-label-premium">📊 Total</div><div class="stat-icon-premium">💬</div></div>
                <div class="stat-premium"><div class="stat-number-premium">${s.canales?.whatsapp || 0}</div><div class="stat-label-premium">📱 WhatsApp</div><div class="stat-icon-premium">📱</div></div>
                <div class="stat-premium"><div class="stat-number-premium">${s.canales?.web || 0}</div><div class="stat-label-premium">💻 Web</div><div class="stat-icon-premium">💻</div></div>
                <div class="stat-premium"><div class="stat-number-premium">${s.escalados || 0}</div><div class="stat-label-premium">🚨 Escalados</div><div class="stat-icon-premium">⚠️</div></div>
                <div class="stat-premium"><div class="stat-number-premium">${s.tasa_escalado || 0}%</div><div class="stat-label-premium">📈 Tasa</div><div class="stat-icon-premium">📈</div></div>
                <div class="stat-premium"><div class="stat-number-premium">${s.pdfs || 0}</div><div class="stat-label-premium">📄 PDFs</div><div class="stat-icon-premium">📄</div></div>
            `;

            // Trend Chart (7 días)
            if (trendChartPremium) trendChartPremium.destroy();
            const trendCtx = document.getElementById('trendChartPremium').getContext('2d');
            const trendLabels = s.tendencia_fechas || ['Sin datos'];
            const trendData = s.tendencia_conteos || [0];

            trendChartPremium = new Chart(trendCtx, {
                type: 'line',
                data: {
                    labels: trendLabels,
                    datasets: [{
                        label: 'Conversaciones',
                        data: trendData,
                        borderColor: '#FFD200',
                        backgroundColor: 'rgba(255,210,0,0.06)',
                        fill: true,
                        tension: 0.4,
                        pointBackgroundColor: '#FFD200',
                        pointBorderColor: '#072146',
                        pointRadius: 5,
                        pointHoverRadius: 8,
                        borderWidth: 2.5
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: true,
                    plugins: {
                        legend: { display: false },
                        datalabels: {
                            anchor: 'end',
                            align: 'top',
                            color: 'rgba(255,255,255,0.3)',
                            font: { size: 8, family: 'JetBrains Mono' },
                            formatter: v => v > 0 ? v : ''
                        }
                    },
                    scales: {
                        x: { grid: { color: 'rgba(255,255,255,0.02)' }, ticks: { color: 'rgba(255,255,255,0.3)', font: { family: 'JetBrains Mono', size: 8 } } },
                        y: { grid: { color: 'rgba(255,255,255,0.02)' }, ticks: { color: 'rgba(255,255,255,0.3)', font: { family: 'JetBrains Mono', size: 8 }, stepSize: 1 } }
                    }
                }
            });

            // Intenciones Chart
            if (intencionesChartPremium) intencionesChartPremium.destroy();
            const intCtx = document.getElementById('intencionesChartPremium').getContext('2d');
            const labels = Object.keys(s.intenciones || {});
            const values = Object.values(s.intenciones || {});
            const colors = ['#69F0AE','#82B1FF','#FFD740','#FF5252','#EA80FC','#FF8A65','#4DD0E1','#FFD200'];

            intencionesChartPremium = new Chart(intCtx, {
                type: 'doughnut',
                data: {
                    labels: labels.length ? labels : ['Sin datos'],
                    datasets: [{
                        data: values.length ? values : [1],
                        backgroundColor: colors.slice(0, labels.length || 1),
                        borderColor: '#051833',
                        borderWidth: 2
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: true,
                    cutout: '60%',
                    plugins: {
                        legend: { position: 'right', labels: { color: 'rgba(255,255,255,0.5)', font: { size: 9, family: 'Inter' }, padding: 8 } },
                        datalabels: {
                            color: 'white',
                            font: { size: 9, weight: 'bold', family: 'JetBrains Mono' },
                            formatter: (v, ctx) => {
                                const total = ctx.dataset.data.reduce((a,b) => a+b, 0);
                                return total > 0 ? ((v/total)*100).toFixed(1)+'%' : '0%';
                            }
                        }
                    }
                }
            });

            // Tabla
            const canal = document.getElementById('filtroCanalPremium').value;
            const conv = await (await fetch('/api/conversaciones?canal=' + canal)).json();

            if (!conv.length) {
                document.getElementById('tablaConversacionesPremium').innerHTML = `<div class="empty-premium"><i class="fas fa-inbox"></i>No hay conversaciones aún</div>`;
                return;
            }

            let h = `<div class="table-premium-wrapper"><table class="table-premium"><thead><tr>
                <th>ID</th><th>Fecha</th><th>Teléfono</th><th>Mensaje</th><th>Intención</th><th>PDF</th><th>Descargar</th>
            </tr></thead><tbody>`;

            for (let c of conv) {
                const fecha = c.fecha ? c.fecha.substring(0, 16).replace('T', ' ') : 'Sin fecha';
                const pdfBtn = c.pdf_nombre ? `<button class="badge-pdf-premium" onclick="descargarPDFPremium(${c.id})"><i class="fas fa-file-pdf"></i> ${escapeHtml(c.pdf_nombre)}</button>` : '—';

                h += `<tr>
                    <td>${c.id}</td>
                    <td>${fecha}</td>
                    <td>${c.telefono || ''}</td>
                    <td>${escapeHtml((c.mensaje || '').substring(0, 40))}</td>
                    <td><span class="badge-premium badge-${c.intencion || 'general'}">${c.intencion || 'general'}</span></td>
                    <td>${c.pdf_nombre ? '<i class="fas fa-file-pdf" style="color:#FF5252"></i>' : '—'}</td>
                    <td>${pdfBtn}</td>
                </tr>`;
            }

            h += '</tbody></table></div>';
            document.getElementById('tablaConversacionesPremium').innerHTML = h;

        } catch (e) {
            console.error(e);
            document.getElementById('tablaConversacionesPremium').innerHTML = `<div class="empty-premium" style="color:#FF5252"><i class="fas fa-exclamation-triangle"></i>Error: ${e.message}</div>`;
        }
    }

    // ── ANALYTICS ──
    async function cargarAnalyticsPremium() {
        try {
            const s = await (await fetch('/api/estadisticas-completas')).json();
            
            document.getElementById('analyticsStats').innerHTML = `
                <div class="stat-premium"><div class="stat-number-premium">${s.total || 0}</div><div class="stat-label-premium">📊 Total</div><div class="stat-icon-premium">📊</div></div>
                <div class="stat-premium"><div class="stat-number-premium">${s.usuarios_unicos || 0}</div><div class="stat-label-premium">👤 Usuarios</div><div class="stat-icon-premium">👤</div></div>
                <div class="stat-premium"><div class="stat-number-premium">${s.tasa_escalado || 0}%</div><div class="stat-label-premium">📈 Escalado</div><div class="stat-icon-premium">📈</div></div>
                <div class="stat-premium"><div class="stat-number-premium">${s.pdfs || 0}</div><div class="stat-label-premium">📄 PDFs</div><div class="stat-icon-premium">📄</div></div>
                <div class="stat-premium"><div class="stat-number-premium">${Object.keys(s.intenciones || {}).length || 0}</div><div class="stat-label-premium">🎯 Intenciones</div><div class="stat-icon-premium">🎯</div></div>
                <div class="stat-premium"><div class="stat-number-premium">${s.canales?.whatsapp || 0}</div><div class="stat-label-premium">📱 WhatsApp</div><div class="stat-icon-premium">📱</div></div>
            `;

            // Hourly Chart
            if (hourlyChartPremium) hourlyChartPremium.destroy();
            const hourlyCtx = document.getElementById('hourlyChart').getContext('2d');
            const horas = s.horas || [];
            const hourLabels = horas.map(h => `${h[0].toString().padStart(2,'0')}:00`);
            const hourData = horas.map(h => h[1]);

            hourlyChartPremium = new Chart(hourlyCtx, {
                type: 'bar',
                data: {
                    labels: hourLabels.length ? hourLabels : ['Sin datos'],
                    datasets: [{
                        label: 'Mensajes',
                        data: hourData.length ? hourData : [0],
                        backgroundColor: 'rgba(255,210,0,0.4)',
                        borderColor: '#FFD200',
                        borderWidth: 1.5,
                        borderRadius: 4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: true,
                    plugins: {
                        legend: { display: false },
                        datalabels: {
                            color: 'rgba(255,255,255,0.3)',
                            font: { size: 7, family: 'JetBrains Mono' },
                            anchor: 'end',
                            align: 'top'
                        }
                    },
                    scales: {
                        x: { grid: { color: 'rgba(255,255,255,0.02)' }, ticks: { color: 'rgba(255,255,255,0.3)', font: { size: 7 } } },
                        y: { grid: { color: 'rgba(255,255,255,0.02)' }, ticks: { color: 'rgba(255,255,255,0.3)', font: { size: 7 } } }
                    }
                }
            });

            // Channel Chart
            if (channelChartPremium) channelChartPremium.destroy();
            const channelCtx = document.getElementById('channelChart').getContext('2d');
            const channelLabels = Object.keys(s.canales || {});
            const channelValues = Object.values(s.canales || {});

            channelChartPremium = new Chart(channelCtx, {
                type: 'doughnut',
                data: {
                    labels: channelLabels.length ? channelLabels : ['Sin datos'],
                    datasets: [{
                        data: channelValues.length ? channelValues : [1],
                        backgroundColor: ['#25D366', '#1A6BFF', '#FFD200'],
                        borderColor: '#051833',
                        borderWidth: 2
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: true,
                    cutout: '55%',
                    plugins: {
                        legend: { position: 'right', labels: { color: 'rgba(255,255,255,0.5)', font: { size: 9 }, padding: 8 } },
                        datalabels: {
                            color: 'white',
                            font: { size: 9, weight: 'bold', family: 'JetBrains Mono' },
                            formatter: (v, ctx) => {
                                const total = ctx.dataset.data.reduce((a,b) => a+b, 0);
                                return total > 0 ? ((v/total)*100).toFixed(1)+'%' : '0%';
                            }
                        }
                    }
                }
            });

            // Días de la semana Chart
            if (diasChartPremium) diasChartPremium.destroy();
            const diasCtx = document.getElementById('diasChart').getContext('2d');
            const dias = s.dias_semana || [];
            const diasNombres = ['Dom','Lun','Mar','Mié','Jue','Vie','Sáb'];
            const diasLabels = dias.map(d => diasNombres[d[0]]);
            const diasData = dias.map(d => d[1]);

            diasChartPremium = new Chart(diasCtx, {
                type: 'bar',
                data: {
                    labels: diasLabels.length ? diasLabels : ['Sin datos'],
                    datasets: [{
                        label: 'Conversaciones',
                        data: diasData.length ? diasData : [0],
                        backgroundColor: ['rgba(255,210,0,0.5)','rgba(26,107,255,0.5)','rgba(0,212,255,0.5)','rgba(0,200,83,0.5)','rgba(255,210,0,0.5)','rgba(26,107,255,0.5)','rgba(0,212,255,0.5)'],
                        borderColor: ['#FFD200','#1A6BFF','#00D4FF','#00C853','#FFD200','#1A6BFF','#00D4FF'],
                        borderWidth: 1.5,
                        borderRadius: 4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: true,
                    plugins: {
                        legend: { display: false },
                        datalabels: {
                            color: 'rgba(255,255,255,0.3)',
                            font: { size: 8, family: 'JetBrains Mono' },
                            anchor: 'end',
                            align: 'top'
                        }
                    },
                    scales: {
                        x: { grid: { color: 'rgba(255,255,255,0.02)' }, ticks: { color: 'rgba(255,255,255,0.3)', font: { size: 8 } } },
                        y: { grid: { color: 'rgba(255,255,255,0.02)' }, ticks: { color: 'rgba(255,255,255,0.3)', font: { size: 8 } } }
                    }
                }
            });

            // Trend 30 días Chart
            if (trend30ChartPremium) trend30ChartPremium.destroy();
            const trend30Ctx = document.getElementById('trend30Chart').getContext('2d');
            const trend30Labels = s.tendencia_30_fechas || ['Sin datos'];
            const trend30Data = s.tendencia_30_conteos || [0];

            trend30ChartPremium = new Chart(trend30Ctx, {
                type: 'line',
                data: {
                    labels: trend30Labels,
                    datasets: [{
                        label: 'Conversaciones',
                        data: trend30Data,
                        borderColor: '#00D4FF',
                        backgroundColor: 'rgba(0,212,255,0.06)',
                        fill: true,
                        tension: 0.4,
                        pointBackgroundColor: '#00D4FF',
                        pointBorderColor: '#072146',
                        pointRadius: 3,
                        pointHoverRadius: 6,
                        borderWidth: 2
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: true,
                    plugins: {
                        legend: { display: false },
                        datalabels: {
                            display: false
                        }
                    },
                    scales: {
                        x: { grid: { color: 'rgba(255,255,255,0.02)' }, ticks: { color: 'rgba(255,255,255,0.3)', font: { family: 'JetBrains Mono', size: 7 } } },
                        y: { grid: { color: 'rgba(255,255,255,0.02)' }, ticks: { color: 'rgba(255,255,255,0.3)', font: { family: 'JetBrains Mono', size: 7 } } }
                    }
                }
            });

        } catch (e) { console.error(e); }
    }

    // ── INSIGHTS ──
    async function cargarInsightsPremium() {
        try {
            const insightsRes = await fetch('/api/insights-reales');
            const insightsData = await insightsRes.json();
            
            const stats = await (await fetch('/api/estadisticas-completas')).json();
            const analisis = await (await fetch('/api/analisis-avanzado')).json();

            // Mostrar insights
            let insightsHtml = `<div class="insight-box"><ul>`;
            if (insightsData.insights && insightsData.insights.length > 0) {
                insightsData.insights.forEach(i => {
                    insightsHtml += `<li><i class="fas fa-lightbulb" style="color:var(--bbva-gold)"></i> ${i}</li>`;
                });
            } else {
                insightsHtml += `<li><i class="fas fa-info-circle" style="color:var(--bbva-cyan)"></i> No hay insights disponibles. Continúa usando el asistente.</li>`;
            }
            insightsHtml += `</ul></div>`;
            document.getElementById('insightsContainer').innerHTML = insightsHtml;

            // Stats de insights
            const topUsuarios = stats.top_usuarios || [];
            
            document.getElementById('insightsStats').innerHTML = `
                <div class="stat-premium"><div class="stat-number-premium">${stats.total || 0}</div><div class="stat-label-premium">📊 Total</div><div class="stat-icon-premium">📊</div></div>
                <div class="stat-premium"><div class="stat-number-premium">${stats.usuarios_unicos || 0}</div><div class="stat-label-premium">👤 Usuarios</div><div class="stat-icon-premium">👤</div></div>
                <div class="stat-premium"><div class="stat-number-premium">${stats.tasa_escalado || 0}%</div><div class="stat-label-premium">📈 Escalado</div><div class="stat-icon-premium">📈</div></div>
                <div class="stat-premium"><div class="stat-number-premium">${stats.pdfs || 0}</div><div class="stat-label-premium">📄 PDFs</div><div class="stat-icon-premium">📄</div></div>
                <div class="stat-premium"><div class="stat-number-premium">${analisis.total_intenciones || 0}</div><div class="stat-label-premium">🎯 Intenciones</div><div class="stat-icon-premium">🎯</div></div>
                <div class="stat-premium"><div class="stat-number-premium">${topUsuarios.length || 0}</div><div class="stat-label-premium">🏆 Top Usuarios</div><div class="stat-icon-premium">🏆</div></div>
            `;

            // Top Usuarios Chart
            if (topUsuariosChartPremium) topUsuariosChartPremium.destroy();
            const topCtx = document.getElementById('topUsuariosChart').getContext('2d');
            const topLabels = topUsuarios.map(u => u[0].substring(0, 10));
            const topData = topUsuarios.map(u => u[1]);

            topUsuariosChartPremium = new Chart(topCtx, {
                type: 'bar',
                data: {
                    labels: topLabels.length ? topLabels : ['Sin datos'],
                    datasets: [{
                        label: 'Mensajes',
                        data: topData.length ? topData : [0],
                        backgroundColor: ['#FFD740','#82B1FF','#69F0AE','#FF5252','#EA80FC'],
                        borderColor: '#051833',
                        borderWidth: 1.5,
                        borderRadius: 4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: true,
                    plugins: {
                        legend: { display: false },
                        datalabels: {
                            color: 'rgba(255,255,255,0.3)',
                            font: { size: 8, family: 'JetBrains Mono' },
                            anchor: 'end',
                            align: 'top'
                        }
                    },
                    scales: {
                        x: { grid: { color: 'rgba(255,255,255,0.02)' }, ticks: { color: 'rgba(255,255,255,0.3)', font: { size: 8 } } },
                        y: { grid: { color: 'rgba(255,255,255,0.02)' }, ticks: { color: 'rgba(255,255,255,0.3)', font: { size: 8 } } }
                    }
                }
            });

            // Palabras Clave Chart
            if (palabrasChartPremium) palabrasChartPremium.destroy();
            const palCtx = document.getElementById('palabrasChart').getContext('2d');
            const topPalabras = (analisis.top_palabras || []).slice(0, 8);
            const palLabels = topPalabras.map(p => p.palabra);
            const palData = topPalabras.map(p => p.frecuencia);

            palabrasChartPremium = new Chart(palCtx, {
                type: 'bar',
                data: {
                    labels: palLabels.length ? palLabels : ['Sin datos'],
                    datasets: [{
                        label: 'Frecuencia',
                        data: palData.length ? palData : [0],
                        backgroundColor: 'rgba(255,210,0,0.4)',
                        borderColor: '#FFD200',
                        borderWidth: 1.5,
                        borderRadius: 4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: true,
                    plugins: {
                        legend: { display: false },
                        datalabels: {
                            color: 'rgba(255,255,255,0.3)',
                            font: { size: 8, family: 'JetBrains Mono' },
                            anchor: 'end',
                            align: 'top'
                        }
                    },
                    scales: {
                        x: { grid: { color: 'rgba(255,255,255,0.02)' }, ticks: { color: 'rgba(255,255,255,0.3)', font: { size: 8 } } },
                        y: { grid: { color: 'rgba(255,255,255,0.02)' }, ticks: { color: 'rgba(255,255,255,0.3)', font: { size: 8 } } }
                    }
                }
            });

        } catch (e) { console.error(e); }
    }

    // ── ACCIONES ──
    async function descargarPDFPremium(id) {
        window.open(`/api/descargar-pdf/${id}`, '_blank');
    }

    async function borrarTodoPremium() {
        if (confirm('⚠️ ¿Estás seguro de borrar TODAS las conversaciones?')) {
            await fetch('/api/conversaciones', { method: 'DELETE' });
            cargarDashboardPremium();
            cargarWhatsAppPremium();
            alert('✅ Todas las conversaciones han sido eliminadas');
        }
    }

    function exportarExcelPremium() { window.open('/api/exportar/excel', '_blank'); }
    function exportarCSVPremium() { window.open('/api/exportar/csv', '_blank'); }

    // ── INICIALIZAR ──
    cargarDashboardPremium();
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def home():
    return HTML

# ==================== INICIO ====================
if __name__ == "__main__":
    import uvicorn
    print("=" * 70)
    print("🏦 BBVA Colombia Asistente Inteligente ULTIMATE")
    print("=" * 70)
    print(f"📊 Dashboard: http://localhost:8000")
    print(f"📱 Webhook: http://localhost:8000/webhook")
    print(f"📈 Analytics: http://localhost:8000#analytics")
    print(f"💡 Insights: http://localhost:8000#insights")
    print("=" * 70)
    print("🚀 FUNCIONALIDADES ULTIMATE:")
    print("  📊 Estadísticas reales desde la DB")
    print("  💡 Insights generados automáticamente")
    print("  📈 Gráficos dinámicos con datos reales")
    print("  🎯 Análisis de palabras clave")
    print("  👤 Top usuarios por actividad")
    print("  ⏰ Actividad por hora y día")
    print("  📱 WhatsApp integrado")
    print("  📄 Procesamiento de PDFs")
    print("=" * 70)
    uvicorn.run(app, host="127.0.0.1", port=8000)