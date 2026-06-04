# main.py - BBVA Colombia Asistente Inteligente con Neon PostgreSQL
from fastapi import FastAPI, Request, Form, UploadFile, File, Query
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
import random
import unicodedata
import re
from datetime import datetime
import os
import pandas as pd
from io import BytesIO, StringIO
import joblib
import httpx
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

app = FastAPI(title="BBVA Colombia Asistente Inteligente")

# ==================== CONFIG WHATSAPP ====================
WHATSAPP_TOKEN  = os.getenv("WHATSAPP_TOKEN", "TU_TOKEN_AQUI")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID", "TU_PHONE_NUMBER_ID")
VERIFY_TOKEN    = os.getenv("VERIFY_TOKEN", "bbva_colombia_2024")

# ==================== MEMORIA ====================
MEMORIA = {}

def obtener_memoria(telefono):
    if telefono not in MEMORIA:
        MEMORIA[telefono] = {"ultima_intencion": "general", "esperando_respuesta": False, "historial": []}
    return MEMORIA[telefono]

def actualizar_memoria(telefono, intencion, esperando=False):
    MEMORIA[telefono] = {"ultima_intencion": intencion, "esperando_respuesta": esperando, "historial": MEMORIA.get(telefono, {}).get("historial", [])}

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
    if any(p in t for p in ["saldo", "cuanto tengo", "dinero", "balance", "plata", "mis ahorros"]):
        return "saldo"
    if any(p in t for p in ["prestamo", "credito", "financiamiento", "pedir dinero", "prestado"]):
        return "prestamo"
    if any(p in t for p in ["apelar", "reclamar", "cargo", "cobro", "reversion", "disputar", "queja", "factura"]):
        return "apelacion"
    if any(p in t for p in ["asesor", "humano", "ejecutivo", "persona", "hablar con"]):
        return "asesor"
    if any(p in t for p in ["tarjeta", "credito", "visa", "mastercard"]):
        return "tarjeta"
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

# ==================== RESPUESTAS MEJORADAS ====================
RESPUESTAS = {
    "saldo": [
        "💰 ¡Claro! Tu saldo actual es *$5.000.000 COP*. ¿Necesitas ver los movimientos recientes?",
        "🏦 Tu cuenta tiene un saldo de *$5.000.000 COP*. Recuerda que puedes transferir desde la app BBVA.",
        "💵 Consulta realizada: Tu saldo disponible es *$5.000.000 COP*."
    ],
    "prestamo": [
        "🏦 BBVA Colombia te ofrece créditos desde *$1.000.000 COP*. ¿Te interesa?",
        "✅ Estás preaprobado hasta *$10.000.000 COP*. ¿Continuamos con la simulación?",
        "📊 Nuestros créditos tienen aprobación en 24 horas. ¿Quieres más información?"
    ],
    "prestamo_aceptado": [
        "✅ ¡Excelente! Un asesor te contactará en 24 horas para formalizar tu crédito.",
        "🏦 Hemos registrado tu solicitud. En breve recibirás la aprobación.",
        "📄 ¡Felicidades! Te enviaremos los documentos digitales por WhatsApp."
    ],
    "apelacion": [
        "📄 Hemos recibido tu documento. Un ejecutivo revisará tu caso en 24 horas.",
        "🔍 Revisaremos tu caso prioritariamente. Te notificaremos por WhatsApp.",
        "✅ Apelación registrada. Espera comunicación de nuestro equipo."
    ],
    "asesor": [
        "👨‍💼 Te conecto con un asesor BBVA. Te llamarán en los próximos 15 minutos.",
        "📞 Un ejecutivo te contactará en breve. ¿Prefieres WhatsApp o llamada?",
        "🕒 Te llamaremos antes de 1 hora. Gracias por tu paciencia."
    ],
    "tarjeta": [
        "💳 Tu tarjeta BBVA tiene disponible *$3.000.000 COP*. Tu próximo pago es el 15 de cada mes.",
        "💰 Tu tarjeta de crédito tiene un cupo de *$3.000.000 COP*. ¿Necesitas diferir una compra?",
        "🛡️ Recuerda que tienes seguro de protección de compras con tu tarjeta BBVA."
    ],
    "rechazo": [
        "Entendido. ¿En qué más puedo ayudarte?",
        "No hay problema. Estoy aquí para lo que necesites.",
        "👍 Perfecto. Si cambias de opinión, solo dímelo."
    ],
    "general": [
        "🙋 Hola, soy el asistente BBVA Colombia. ¿En qué te ayudo? Puedo consultar saldos, préstamos o conectar con un asesor.",
        "🏦 ¡Bienvenido! ¿Necesitas información sobre tu cuenta, tarjeta o préstamos?",
        "💬 Hola. ¿En qué puedo asistirte hoy?"
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

# ==================== DB HELPERS ====================
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
        print(f"✅ Guardado: {intencion} - PDF: {pdf_nombre if pdf_nombre else 'Sin PDF'}")
    except Exception as e:
        print(f"❌ Error guardando: {e}")

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

def obtener_estadisticas():
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM conversaciones")
        total = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM conversaciones WHERE escalado=1")
        escalados = cur.fetchone()[0]
        cur.execute("SELECT intencion, COUNT(*) FROM conversaciones WHERE intencion IS NOT NULL GROUP BY intencion")
        intenciones = dict(cur.fetchall())
        cur.execute("SELECT canal, COUNT(*) FROM conversaciones GROUP BY canal")
        canales = dict(cur.fetchall())
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
        rows = list(reversed(rows))
        return [str(r[0]) for r in rows], [r[1] for r in rows]
    except Exception as e:
        print(f"❌ Error en tendencia: {e}")
        return [], []

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
    stats = obtener_estadisticas()
    fechas, conteos = obtener_tendencia()
    stats["tendencia_fechas"] = fechas
    stats["tendencia_conteos"] = conteos
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
    respuesta = generar_respuesta(intencion, telefono)
    escalado = 1 if intencion in ["asesor", "apelacion"] else 0
    
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

@app.get("/health")
async def health():
    return JSONResponse({"status": "ok"})

# ==================== HTML MEJORADO ====================
HTML = """<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BBVA Colombia | Asistente Inteligente</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-datalabels@2/dist/chartjs-plugin-datalabels.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:opsz,wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Inter', sans-serif; background: linear-gradient(135deg, #f5f7fa 0%, #e9ecef 100%); min-height: 100vh; }
        .container { max-width: 1400px; margin: 0 auto; padding: 24px; }
        
        /* Header */
        .header { background: linear-gradient(135deg, #0033A0, #002166); border-radius: 20px; padding: 20px 32px; margin-bottom: 28px; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 20px; box-shadow: 0 10px 30px rgba(0,51,160,0.3); }
        .logo-area { display: flex; align-items: center; gap: 16px; }
        .logo { width: 50px; height: 50px; background: white; border-radius: 15px; display: flex; align-items: center; justify-content: center; font-size: 26px; box-shadow: 0 4px 10px rgba(0,0,0,0.1); }
        .header-text h1 { color: white; font-size: 24px; font-weight: 700; letter-spacing: -0.5px; }
        .header-text p { color: rgba(255,255,255,0.85); font-size: 13px; margin-top: 4px; }
        .clock-area { background: rgba(255,255,255,0.12); padding: 10px 24px; border-radius: 50px; text-align: center; backdrop-filter: blur(10px); }
        .clock-time { font-size: 26px; font-weight: 700; color: #FFD700; font-family: monospace; }
        .clock-date { font-size: 11px; color: rgba(255,255,255,0.8); }
        
        /* Tabs */
        .tabs { display: flex; gap: 12px; margin-bottom: 28px; flex-wrap: wrap; }
        .tab { background: white; border: none; padding: 12px 32px; border-radius: 50px; font-size: 14px; font-weight: 600; cursor: pointer; transition: all 0.3s ease; color: #5a6874; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }
        .tab i { margin-right: 8px; }
        .tab.active { background: #0033A0; color: white; box-shadow: 0 6px 20px rgba(0,51,160,0.4); transform: translateY(-2px); }
        .tab:hover:not(.active) { background: #e8eeff; transform: translateY(-1px); }
        
        /* Cards */
        .card { background: white; border-radius: 24px; padding: 28px; margin-bottom: 28px; box-shadow: 0 5px 20px rgba(0,0,0,0.06); transition: all 0.3s ease; }
        .card:hover { transform: translateY(-3px); box-shadow: 0 10px 30px rgba(0,0,0,0.1); }
        .card h2 { color: #0033A0; font-size: 20px; font-weight: 700; margin-bottom: 20px; border-left: 4px solid #FFD700; padding-left: 16px; }
        .card h3 { color: #0033A0; font-size: 18px; font-weight: 600; margin-bottom: 16px; }
        
        /* Stats Grid */
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 20px; margin-bottom: 28px; }
        .stat-card { background: linear-gradient(135deg, #0033A0, #002880); border-radius: 20px; padding: 22px; text-align: center; color: white; cursor: pointer; transition: all 0.3s ease; }
        .stat-card:hover { transform: translateY(-5px); box-shadow: 0 12px 30px rgba(0,51,160,0.4); }
        .stat-number { font-size: 38px; font-weight: 800; }
        .stat-label { font-size: 12px; opacity: 0.9; margin-top: 8px; letter-spacing: 0.5px; }
        
        /* Chat */
        .chat-area { height: 420px; overflow-y: auto; border: 1px solid #eef2f6; border-radius: 20px; padding: 20px; background: #fafcff; margin-bottom: 20px; }
        .message { display: flex; margin-bottom: 16px; animation: fadeInUp 0.3s ease; }
        @keyframes fadeInUp { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        .message.user { justify-content: flex-end; }
        .bubble { max-width: 75%; padding: 12px 20px; border-radius: 22px; font-size: 14px; line-height: 1.45; word-wrap: break-word; }
        .message.user .bubble { background: linear-gradient(135deg, #0033A0, #002880); color: white; border-bottom-right-radius: 6px; }
        .message.bot .bubble { background: #f0f4f9; color: #1a2c3e; border-bottom-left-radius: 6px; }
        .intent-badge { font-size: 10px; opacity: 0.7; margin-top: 8px; }
        
        /* Input Group */
        .input-group { display: flex; gap: 12px; flex-wrap: wrap; }
        .input-group input, .input-group textarea { flex: 1; padding: 12px 16px; border: 1px solid #e2e8f0; border-radius: 14px; font-size: 14px; font-family: inherit; transition: all 0.3s ease; }
        .input-group input:focus, .input-group textarea:focus { outline: none; border-color: #0033A0; box-shadow: 0 0 0 3px rgba(0,51,160,0.1); }
        
        /* Botones Mejorados */
        .btn { background: linear-gradient(135deg, #0033A0, #002880); color: white; border: none; padding: 12px 26px; border-radius: 50px; font-size: 14px; font-weight: 600; cursor: pointer; transition: all 0.3s ease; display: inline-flex; align-items: center; gap: 8px; box-shadow: 0 2px 8px rgba(0,51,160,0.2); }
        .btn:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(0,51,160,0.35); }
        .btn:active { transform: translateY(0); }
        
        .btn-outline { background: transparent; color: #0033A0; border: 2px solid #0033A0; box-shadow: none; }
        .btn-outline:hover { background: #e8eeff; transform: translateY(-2px); box-shadow: 0 6px 20px rgba(0,51,160,0.15); }
        
        .btn-danger { background: linear-gradient(135deg, #dc3545, #c82333); box-shadow: 0 2px 8px rgba(220,53,69,0.2); }
        .btn-danger:hover { box-shadow: 0 6px 20px rgba(220,53,69,0.35); }
        
        .btn-success { background: linear-gradient(135deg, #28a745, #1e7e34); box-shadow: 0 2px 8px rgba(40,167,69,0.2); }
        .btn-success:hover { box-shadow: 0 6px 20px rgba(40,167,69,0.35); }
        
        .btn-sm { padding: 8px 16px; font-size: 12px; }
        .btn-icon { padding: 10px 18px; }
        
        .file-btn { background: #f8fafc; padding: 12px 24px; border-radius: 50px; cursor: pointer; border: 2px solid #e2e8f0; font-weight: 600; transition: all 0.3s ease; display: inline-flex; align-items: center; gap: 8px; }
        .file-btn:hover { background: #eef2f6; border-color: #0033A0; transform: translateY(-1px); }
        input[type="file"] { display: none; }
        
        /* Toolbar */
        .toolbar { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 24px; }
        .filter-select { padding: 10px 20px; border-radius: 50px; border: 1px solid #e2e8f0; background: white; font-size: 14px; cursor: pointer; transition: all 0.3s ease; }
        .filter-select:focus { outline: none; border-color: #0033A0; }
        
        /* Tabla */
        .table-wrapper { overflow-x: auto; border-radius: 20px; }
        table { width: 100%; border-collapse: collapse; font-size: 13px; }
        th, td { padding: 16px; text-align: left; border-bottom: 1px solid #eef2f6; }
        th { background: #f8fafd; color: #0033A0; font-weight: 700; font-size: 13px; }
        tr:hover td { background: #fafcff; }
        
        /* Badges */
        .badge { display: inline-block; padding: 4px 14px; border-radius: 50px; font-size: 11px; font-weight: 700; }
        .badge.saldo { background: #d4edda; color: #155724; }
        .badge.prestamo { background: #d1ecf1; color: #0c5460; }
        .badge.apelacion { background: #fff3cd; color: #856404; }
        .badge.asesor { background: #f8d7da; color: #721c24; }
        .badge.tarjeta { background: #e0d4ff; color: #4a1d96; }
        .badge.general { background: #e9ecef; color: #495057; }
        
        /* Botón PDF en tabla */
        .btn-pdf { background: linear-gradient(135deg, #28a745, #1e7e34); border: none; padding: 6px 14px; border-radius: 40px; color: white; cursor: pointer; font-size: 11px; font-weight: 600; transition: all 0.3s ease; display: inline-flex; align-items: center; gap: 6px; }
        .btn-pdf:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(40,167,69,0.4); }
        .btn-pdf i { font-size: 11px; }
        
        /* Charts */
        .charts-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 28px; }
        canvas { max-height: 280px; }
        
        /* Loader */
        .loader { display: inline-block; width: 18px; height: 18px; border: 2px solid #e2e8f0; border-top-color: #0033A0; border-radius: 50%; animation: spin 0.8s linear infinite; margin-right: 8px; }
        @keyframes spin { to { transform: rotate(360deg); } }
        
        /* Footer */
        .footer { text-align: center; padding: 24px; color: #6c757d; font-size: 12px; border-top: 1px solid #e2e8f0; margin-top: 20px; }
        
        /* Empty State */
        .empty-table { text-align: center; padding: 60px; color: #6c757d; font-size: 14px; }
        
        /* Scrollbar */
        ::-webkit-scrollbar { width: 8px; height: 8px; }
        ::-webkit-scrollbar-track { background: #f1f1f1; border-radius: 10px; }
        ::-webkit-scrollbar-thumb { background: #c1c1c1; border-radius: 10px; }
        ::-webkit-scrollbar-thumb:hover { background: #a8a8a8; }
        
        @media (max-width: 768px) { .container { padding: 16px; } .stat-number { font-size: 28px; } .charts-grid { grid-template-columns: 1fr; } .bubble { max-width: 90%; } .tabs .tab { padding: 8px 20px; font-size: 12px; } }
    </style>
</head>
<body>
<div class="container">
    <div class="header">
        <div class="logo-area">
            <div class="logo">🏦</div>
            <div class="header-text">
                <h1>BBVA Colombia</h1>
                <p>Asistente Inteligente · Machine Learning · WhatsApp</p>
            </div>
        </div>
        <div class="clock-area">
            <div class="clock-time" id="clockTime">--:--:--</div>
            <div class="clock-date" id="clockDate">--/--/----</div>
        </div>
    </div>

    <div class="tabs">
        <button class="tab active" onclick="showTab('chat')"><i class="fas fa-comment-dots"></i> Chat Web</button>
        <button class="tab" onclick="showTab('whatsapp')"><i class="fab fa-whatsapp"></i> WhatsApp</button>
        <button class="tab" onclick="showTab('dashboard')"><i class="fas fa-chart-line"></i> Dashboard</button>
    </div>

    <div id="chat-section">
        <div class="card">
            <h2><i class="fas fa-comments"></i> Simulador de Conversación</h2>
            <div class="chat-area" id="chatMessages">
                <div class="message bot"><div class="bubble">🙋 ¡Hola! Soy el asistente BBVA Colombia.<br><br>💰 <strong>Saldo</strong><br>🏦 <strong>Préstamo</strong><br>💳 <strong>Tarjeta</strong><br>📄 <strong>Apelación</strong> (adjunta PDF)<br>👨‍💼 <strong>Asesor</strong><br><br>¿En qué te ayudo?</div></div>
            </div>
            <div class="input-group">
                <input type="text" id="telefono" placeholder="📱 Teléfono" value="573053667704">
                <textarea id="mensaje" placeholder="✏️ Escribe tu mensaje..." rows="2"></textarea>
                <label class="file-btn" for="pdfFile"><i class="fas fa-paperclip"></i> Adjuntar PDF</label>
                <input type="file" id="pdfFile" accept="application/pdf">
                <button class="btn" onclick="enviarMensaje()"><i class="fas fa-paper-plane"></i> Enviar</button>
                <button class="btn-outline" onclick="limpiarChat()"><i class="fas fa-trash-alt"></i> Limpiar</button>
            </div>
        </div>
    </div>

    <div id="whatsapp-section" style="display:none">
        <div class="card">
            <h2><i class="fab fa-whatsapp"></i> Mensajes WhatsApp</h2>
            <div class="toolbar">
                <button class="btn" onclick="cargarWhatsApp()"><i class="fas fa-sync-alt"></i> Actualizar</button>
                <button class="btn-success" onclick="exportarExcel()"><i class="fas fa-file-excel"></i> Excel</button>
                <button class="btn-outline" onclick="exportarCSV()"><i class="fas fa-file-csv"></i> CSV</button>
            </div>
            <div id="waStats"></div>
            <div id="waTabla"></div>
        </div>
    </div>

    <div id="dashboard-section" style="display:none">
        <div class="stats-grid" id="statsContainer"></div>
        <div class="charts-grid">
            <div class="card"><h3><i class="fas fa-chart-line"></i> Tendencia (7 días)</h3><canvas id="trendChart"></canvas></div>
            <div class="card"><h3><i class="fas fa-chart-pie"></i> Intenciones</h3><canvas id="intencionesChart"></canvas></div>
        </div>
        <div class="card">
            <h3><i class="fas fa-table"></i> Conversaciones Recientes</h3>
            <div class="toolbar">
                <select class="filter-select" id="filtroCanal" onchange="cargarDashboard()">
                    <option value="">🌐 Todos</option><option value="web">💻 Web</option><option value="whatsapp">📱 WhatsApp</option>
                </select>
                <button class="btn-success" onclick="exportarExcel()"><i class="fas fa-file-excel"></i> Excel</button>
                <button class="btn-outline" onclick="exportarCSV()"><i class="fas fa-file-csv"></i> CSV</button>
                <button class="btn-danger" onclick="borrarTodo()"><i class="fas fa-trash-alt"></i> Borrar Todo</button>
                <button class="btn" onclick="cargarDashboard()"><i class="fas fa-sync-alt"></i> Actualizar</button>
            </div>
            <div id="tablaConversaciones"></div>
        </div>
    </div>
    <div class="footer">🏦 BBVA Colombia Asistente Inteligente · Datos en tiempo real · Desarrollado con ❤️</div>
</div>

<script>
Chart.register(ChartDataLabels);
let trendChart, intencionesChart;

function actualizarReloj() {
    const ahora = new Date();
    document.getElementById('clockTime').textContent = ahora.toLocaleTimeString('es-CO');
    document.getElementById('clockDate').textContent = ahora.toLocaleDateString('es-CO', { weekday:'long', year:'numeric', month:'long', day:'numeric' });
}
actualizarReloj();
setInterval(actualizarReloj, 1000);

function showTab(tab) {
    document.getElementById('chat-section').style.display = tab === 'chat' ? 'block' : 'none';
    document.getElementById('whatsapp-section').style.display = tab === 'whatsapp' ? 'block' : 'none';
    document.getElementById('dashboard-section').style.display = tab === 'dashboard' ? 'block' : 'none';
    if (tab === 'whatsapp') cargarWhatsApp();
    if (tab === 'dashboard') cargarDashboard();
    document.querySelectorAll('.tab').forEach((t, i) => { t.classList.toggle('active', (tab === 'chat' && i === 0) || (tab === 'whatsapp' && i === 1) || (tab === 'dashboard' && i === 2)); });
}

async function enviarMensaje() {
    const telefono = document.getElementById('telefono').value;
    const mensaje = document.getElementById('mensaje').value;
    const pdfFile = document.getElementById('pdfFile').files[0];
    if (!mensaje.trim()) { alert('✏️ Escribe un mensaje'); return; }
    
    const chatDiv = document.getElementById('chatMessages');
    chatDiv.innerHTML += `<div class="message user"><div class="bubble">${escapeHtml(mensaje)}${pdfFile ? '<br><i class="fas fa-paperclip"></i> ' + escapeHtml(pdfFile.name) : ''}</div></div>`;
    chatDiv.scrollTop = chatDiv.scrollHeight;
    
    const fd = new FormData();
    fd.append('telefono', telefono);
    fd.append('mensaje', mensaje);
    if (pdfFile) fd.append('pdf', pdfFile);
    document.getElementById('mensaje').value = '';
    document.getElementById('pdfFile').value = '';
    
    const loading = document.createElement('div');
    loading.className = 'message bot';
    loading.innerHTML = '<div class="bubble"><span class="loader"></span> Procesando...</div>';
    chatDiv.appendChild(loading);
    chatDiv.scrollTop = chatDiv.scrollHeight;
    
    try {
        const res = await fetch('/api/simular', { method: 'POST', body: fd });
        const data = await res.json();
        loading.remove();
        chatDiv.innerHTML += `<div class="message bot"><div class="bubble">🤖 ${escapeHtml(data.respuesta)}<div class="intent-badge"><i class="fas fa-tag"></i> ${data.intencion} ${data.escalado ? '· <i class="fas fa-exclamation-triangle"></i> Escalado' : ''}</div></div></div>`;
        chatDiv.scrollTop = chatDiv.scrollHeight;
        if (data.escalado) cargarDashboard();
    } catch(e) {
        loading.remove();
        chatDiv.innerHTML += `<div class="message bot"><div class="bubble"><i class="fas fa-exclamation-circle"></i> Error: ${e.message}</div></div>`;
    }
}

function limpiarChat() {
    document.getElementById('chatMessages').innerHTML = '<div class="message bot"><div class="bubble">🙋 ¡Hola! Soy el asistente BBVA.<br><br>💰 Saldo<br>🏦 Préstamo<br>💳 Tarjeta<br>📄 Apelación<br>👨‍💼 Asesor<br><br>¿En qué te ayudo?</div></div>';
}

async function cargarWhatsApp() {
    try {
        const res = await fetch('/api/conversaciones?canal=whatsapp');
        const data = await res.json();
        const esc = data.filter(c => c.escalado).length;
        document.getElementById('waStats').innerHTML = `<div style="display:flex; gap:16px; flex-wrap:wrap; margin-bottom:20px"><div style="background:linear-gradient(135deg,#25D366,#128C7E);border-radius:20px;padding:18px 28px;color:white"><div style="font-size:32px;font-weight:800">📱 ${data.length}</div><div>Mensajes WhatsApp</div></div><div style="background:linear-gradient(135deg,#dc3545,#c82333);border-radius:20px;padding:18px 28px;color:white"><div style="font-size:32px;font-weight:800">🚨 ${esc}</div><div>Escalados</div></div></div>`;
        if (!data.length) { document.getElementById('waTabla').innerHTML = '<div class="empty-table"><i class="fas fa-inbox"></i> Sin mensajes de WhatsApp aún</div>'; return; }
        let h = '<div class="table-wrapper"><table><thead><tr><th><i class="fas fa-calendar"></i> Fecha</th><th><i class="fas fa-phone"></i> Teléfono</th><th><i class="fas fa-comment"></i> Mensaje</th><th><i class="fas fa-tag"></i> Intención</th></tr></thead><tbody>';
        for (let c of data.slice(0,30)) {
            let fecha = c.fecha ? c.fecha.substring(0,16).replace('T', ' ') : 'Sin fecha';
            h += `<tr><td>${fecha}</td><td>${c.telefono}</td><td>${escapeHtml((c.mensaje||'').substring(0,50))}</td><td><span class="badge ${c.intencion}">${c.intencion||'general'}</span></td></tr>`;
        }
        h += '</tbody></table></div>';
        document.getElementById('waTabla').innerHTML = h;
    } catch(e) { console.error(e); }
}

async function cargarDashboard() {
    try {
        const s = await (await fetch('/api/estadisticas')).json();
        document.getElementById('statsContainer').innerHTML = `
            <div class="stat-card"><div class="stat-number">${s.total}</div><div class="stat-label">📊 Total Conversaciones</div></div>
            <div class="stat-card"><div class="stat-number">${s.canales?.whatsapp||0}</div><div class="stat-label">📱 WhatsApp</div></div>
            <div class="stat-card"><div class="stat-number">${s.canales?.web||0}</div><div class="stat-label">💻 Web Chat</div></div>
            <div class="stat-card"><div class="stat-number">${s.escalados}</div><div class="stat-label">🚨 Escalados</div></div>
            <div class="stat-card"><div class="stat-number">${s.tasa_escalado}%</div><div class="stat-label">📈 Tasa de Escalamiento</div></div>
            <div class="stat-card"><div class="stat-number">${s.pdfs||0}</div><div class="stat-label">📄 Documentos PDF</div></div>
        `;
        
        if (trendChart) trendChart.destroy();
        trendChart = new Chart(document.getElementById('trendChart'), {
            type: 'line', 
            data: { labels: s.tendencia_fechas || [], datasets: [{ label: 'Conversaciones', data: s.tendencia_conteos || [], borderColor: '#0033A0', backgroundColor: 'rgba(0,51,160,0.1)', fill: true, tension: 0.4 }] },
            options: { responsive: true, maintainAspectRatio: true, plugins: { datalabels: { anchor: 'end', align: 'top', color: '#0033A0', fontWeight: 'bold', font: { size: 11 } } } }
        });
        
        if (intencionesChart) intencionesChart.destroy();
        const labels = Object.keys(s.intenciones);
        const values = Object.values(s.intenciones);
        const colores = ['#28a745', '#17a2b8', '#ffc107', '#dc3545', '#6c757d', '#fd7e14', '#20c997', '#e83e8c'];
        intencionesChart = new Chart(document.getElementById('intencionesChart'), {
            type: 'pie', 
            data: { labels: labels, datasets: [{ data: values, backgroundColor: colores.slice(0, labels.length), borderWidth: 0 }] },
            options: { responsive: true, maintainAspectRatio: true, plugins: { legend: { position: 'right' }, datalabels: { formatter: (v, ctx) => { const total = values.reduce((a,b) => a + b, 0); return ((v / total) * 100).toFixed(1) + '%'; }, color: 'white', fontWeight: 'bold', font: { size: 12 } } } }
        });
        
        const canal = document.getElementById('filtroCanal').value;
        const conv = await (await fetch('/api/conversaciones?canal='+canal)).json();
        
        if (!conv.length) { document.getElementById('tablaConversaciones').innerHTML = '<div class="empty-table"><i class="fas fa-inbox"></i> No hay conversaciones aún</div>'; return; }
        
        let h = '<div class="table-wrapper"><table><thead><tr><th><i class="fas fa-hashtag"></i> ID</th><th><i class="fas fa-calendar"></i> Fecha</th><th><i class="fas fa-phone"></i> Teléfono</th><th><i class="fas fa-comment"></i> Mensaje</th><th><i class="fas fa-tag"></i> Intención</th><th><i class="fas fa-paperclip"></i> PDF</th><th><i class="fas fa-download"></i> Descargar</th></tr></thead><tbody>';
        for (let c of conv) {
            let fecha = c.fecha ? c.fecha.substring(0,16).replace('T', ' ') : 'Sin fecha';
            let btn = c.pdf_nombre ? `<button class="btn-pdf" onclick="descargarPDF(${c.id})"><i class="fas fa-download"></i> PDF</button>` : '—';
            h += `<tr><td>${c.id}</td><td>${fecha}</td><td>${c.telefono || ''}</td><td>${escapeHtml((c.mensaje || '').substring(0,60))}</td><td><span class="badge ${c.intencion}">${c.intencion || 'general'}</span></td><td>${c.pdf_nombre ? '<i class="fas fa-file-pdf" style="color:#dc3545"></i>' : '—'}</td><td>${btn}</td></tr>`;
        }
        h += '</tbody></table></div>';
        document.getElementById('tablaConversaciones').innerHTML = h;
    } catch(e) { console.error(e); document.getElementById('tablaConversaciones').innerHTML = '<div class="empty-table" style="color:red"><i class="fas fa-exclamation-triangle"></i> Error: ' + e.message + '</div>'; }
}

async function descargarPDF(id) { window.open(`/api/descargar-pdf/${id}`, '_blank'); }
async function borrarTodo() { if(confirm('⚠️ ¿Estás seguro de borrar TODAS las conversaciones? Esta acción no se puede deshacer.')){ await fetch('/api/conversaciones',{method:'DELETE'}); cargarDashboard(); cargarWhatsApp(); alert('✅ Todas las conversaciones han sido eliminadas'); } }
function exportarExcel() { window.open('/api/exportar/excel', '_blank'); }
function exportarCSV() { window.open('/api/exportar/csv', '_blank'); }
function escapeHtml(t) { if(!t) return ''; const d=document.createElement('div'); d.textContent=t; return d.innerHTML; }

cargarDashboard();
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
    print("=" * 50)
    print("🏦 BBVA Colombia Asistente Inteligente")
    print("=" * 50)
    print(f"📊 Dashboard: http://localhost:8000")
    print(f"📱 Webhook: http://localhost:8000/webhook")
    print("=" * 50)
    uvicorn.run(app, host="127.0.0.1", port=8000)