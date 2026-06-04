# main.py - BBVA Colombia Asistente Inteligente
from fastapi import FastAPI, Request, Form, UploadFile, File, Query
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
import sqlite3
import random
import unicodedata
import re
from datetime import datetime
import os
import pandas as pd
from io import BytesIO, StringIO
import joblib
import httpx

app = FastAPI(title="BBVA Colombia Asistente Inteligente")

# ==================== CONFIG WHATSAPP ====================
WHATSAPP_TOKEN  = os.getenv("WHATSAPP_TOKEN", "TU_TOKEN_AQUI")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID", "TU_PHONE_NUMBER_ID")
VERIFY_TOKEN    = os.getenv("VERIFY_TOKEN", "bbva_colombia_2024")

# ==================== MEMORIA ====================
MEMORIA = {}

def obtener_memoria(telefono):
    if telefono not in MEMORIA:
        MEMORIA[telefono] = {"ultima_intencion": "general", "esperando_respuesta": False}
    return MEMORIA[telefono]

def actualizar_memoria(telefono, intencion, esperando=False):
    MEMORIA[telefono] = {"ultima_intencion": intencion, "esperando_respuesta": esperando}

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
    if any(p in t for p in ["saldo", "cuanto tengo", "dinero", "balance", "plata"]):
        return "saldo"
    if any(p in t for p in ["prestamo", "credito", "financiamiento", "pedir dinero"]):
        return "prestamo"
    if any(p in t for p in ["apelar", "reclamar", "cargo", "cobro", "reversion", "disputar", "queja", "factura"]):
        return "apelacion"
    if any(p in t for p in ["asesor", "humano", "ejecutivo", "persona", "hablar con"]):
        return "asesor"
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

# ==================== BASE DE DATOS ====================
DB_PATH = "conversaciones.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS conversaciones (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            telefono TEXT,
            mensaje TEXT,
            intencion TEXT,
            respuesta TEXT,
            escalado INTEGER DEFAULT 0,
            canal TEXT DEFAULT 'web',
            pdf_texto TEXT,
            pdf_nombre TEXT,
            fecha TEXT
        )
    """)
    conn.commit()
    conn.close()
    print("✅ Base de datos inicializada")

init_db()

# ==================== RESPUESTAS ====================
RESPUESTAS = {
    "saldo": ["💰 Tu saldo actual es *$5.000.000 COP*. ¿Necesitas algo más?"],
    "prestamo": ["🏦 BBVA Colombia te ofrece créditos desde *$1.000.000 COP*. ¿Te interesa?", "✅ Estás preaprobado hasta *$10.000.000 COP*. ¿Continuamos?"],
    "prestamo_aceptado": ["✅ ¡Excelente! Un asesor te contactará en 24 horas."],
    "apelacion": ["📄 Hemos recibido tu documento. Un ejecutivo revisará tu caso en 24 horas."],
    "asesor": ["👨‍💼 Te conecto con un asesor BBVA. Te llamarán en breve."],
    "rechazo": ["Entendido. ¿En qué más puedo ayudarte?"],
    "general": ["🙋 Hola, soy el asistente BBVA Colombia. ¿En qué te ayudo?"],
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
        return texto[:2000] if texto.strip() else ""
    except Exception as e:
        print(f"Error PDF: {e}")
        return ""

# ==================== DB HELPERS ====================
def guardar_conversacion(telefono, mensaje, intencion, respuesta, escalado, pdf_texto, pdf_nombre, canal):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""INSERT INTO conversaciones (telefono, mensaje, intencion, respuesta, escalado, pdf_texto, pdf_nombre, canal, fecha)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                 (telefono, mensaje[:500], intencion, respuesta[:500], escalado, 
                  (pdf_texto or "")[:2000], pdf_nombre, canal, datetime.now().isoformat()))
    conn.commit()
    conn.close()

def obtener_conversaciones(limite=50, canal=""):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    if canal:
        rows = conn.execute("SELECT * FROM conversaciones WHERE canal=? ORDER BY id DESC LIMIT ?", (canal, limite)).fetchall()
    else:
        rows = conn.execute("SELECT * FROM conversaciones ORDER BY id DESC LIMIT ?", (limite,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def obtener_estadisticas():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM conversaciones")
    total = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM conversaciones WHERE escalado=1")
    escalados = cursor.fetchone()[0]
    cursor.execute("SELECT intencion, COUNT(*) FROM conversaciones WHERE intencion IS NOT NULL GROUP BY intencion")
    intenciones = dict(cursor.fetchall())
    cursor.execute("SELECT canal, COUNT(*) FROM conversaciones GROUP BY canal")
    canales = dict(cursor.fetchall())
    cursor.execute("SELECT COUNT(*) FROM conversaciones WHERE pdf_nombre IS NOT NULL")
    pdfs = cursor.fetchone()[0]
    conn.close()
    return {"total": total, "escalados": escalados, "tasa_escalado": round(escalados/total*100,1) if total else 0, "intenciones": intenciones, "canales": canales, "pdfs": pdfs}

def obtener_tendencia():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT DATE(fecha), COUNT(*) FROM conversaciones WHERE fecha IS NOT NULL GROUP BY DATE(fecha) ORDER BY DATE(fecha) DESC LIMIT 7").fetchall()
    conn.close()
    rows = list(reversed(rows))
    return [r[0] for r in rows], [r[1] for r in rows]

@app.get("/api/descargar-pdf/{conversacion_id}")
async def descargar_pdf(conversacion_id: int):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT pdf_nombre, pdf_texto FROM conversaciones WHERE id = ?", (conversacion_id,))
    row = cursor.fetchone()
    conn.close()
    
    if not row or not row[1]:
        return JSONResponse({"error": "No hay PDF asociado"}, status_code=404)
    
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER
    import io
    
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    
    titulo_style = ParagraphStyle('Titulo', parent=styles['Heading1'], alignment=TA_CENTER, textColor='#0033A0', fontSize=16, spaceAfter=20)
    
    contenido = [
        Paragraph("BBVA Colombia", titulo_style),
        Spacer(1, 20),
        Paragraph(f"<b>📄 Archivo:</b> {row[0]}", styles['Normal']),
        Spacer(1, 10),
        Paragraph("<b>📝 Contenido:</b>", styles['Normal']),
        Spacer(1, 10),
    ]
    
    for parrafo in row[1].split('\n'):
        if parrafo.strip():
            contenido.append(Paragraph(parrafo[:2000], styles['Normal']))
            contenido.append(Spacer(1, 6))
    
    doc.build(contenido)
    pdf_buffer.seek(0)
    return StreamingResponse(pdf_buffer, media_type="application/pdf", headers={"Content-Disposition": f"attachment; filename=documento_{conversacion_id}.pdf"})

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
        guardar_conversacion(telefono, texto, intencion, respuesta, escalado, None, None, "whatsapp")
        await enviar_whatsapp(telefono, respuesta)
    except:
        pass
    return JSONResponse({"status": "ok"})

# ==================== API ENDPOINTS ====================
@app.get("/api/conversaciones")
async def api_conversaciones(canal: str = Query(default="")):
    return JSONResponse(obtener_conversaciones(50, canal))

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
    tiene_pdf = False
    
    if pdf and pdf.filename:
        pdf_nombre = pdf.filename
        tiene_pdf = True
        try:
            contenido = await pdf.read()
            pdf_texto = extraer_texto_pdf(contenido)
        except Exception as e:
            print(f"Error PDF: {e}")
    
    intencion = clasificar_intencion(mensaje, telefono, tiene_pdf)
    respuesta = generar_respuesta(intencion, telefono)
    escalado = 1 if intencion in ["asesor", "apelacion"] else 0
    
    guardar_conversacion(telefono, mensaje, intencion, respuesta, escalado, pdf_texto, pdf_nombre, "web")
    
    return JSONResponse({
        "respuesta": respuesta,
        "intencion": intencion,
        "escalado": escalado,
        "pdf_leido": bool(pdf_texto),
        "pdf_nombre": pdf_nombre
    })

@app.delete("/api/conversaciones")
async def eliminar_todas():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM conversaciones")
    conn.commit()
    conn.close()
    return JSONResponse({"message": "Eliminadas"})

@app.get("/api/exportar/excel")
async def exportar_excel():
    conn = sqlite3.connect(DB_PATH)
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
    conn = sqlite3.connect(DB_PATH)
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

# ==================== HTML PROFESIONAL BBVA ====================
HTML = """<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BBVA Colombia | Asistente Inteligente</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-datalabels@2/dist/chartjs-plugin-datalabels.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,300;14..32,400;14..32,500;14..32,600;14..32,700;14..32,800&display=swap" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: #f0f2f5;
            min-height: 100vh;
            color: #1a1a2e;
        }
        
        .container { max-width: 1400px; margin: 0 auto; padding: 24px; }
        
        /* Header BBVA */
        .header {
            background: #0033A0;
            border-radius: 20px;
            padding: 20px 32px;
            margin-bottom: 28px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 20px;
            box-shadow: 0 4px 20px rgba(0, 51, 160, 0.2);
        }
        
        .logo-area { display: flex; align-items: center; gap: 16px; }
        
        .logo {
            width: 48px;
            height: 48px;
            background: white;
            border-radius: 14px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
        }
        
        .header-text h1 {
            color: white;
            font-size: 22px;
            font-weight: 700;
            letter-spacing: -0.3px;
        }
        
        .header-text p {
            color: rgba(255, 255, 255, 0.8);
            font-size: 13px;
            margin-top: 4px;
        }
        
        /* Reloj */
        .clock-area {
            background: rgba(255, 255, 255, 0.15);
            padding: 10px 24px;
            border-radius: 40px;
            text-align: center;
            backdrop-filter: blur(4px);
        }
        
        .clock-time {
            font-size: 24px;
            font-weight: 700;
            color: #FFD700;
            font-family: monospace;
        }
        
        .clock-date {
            font-size: 11px;
            color: rgba(255, 255, 255, 0.8);
        }
        
        /* Tabs */
        .tabs {
            display: flex;
            gap: 12px;
            margin-bottom: 28px;
            flex-wrap: wrap;
        }
        
        .tab {
            background: white;
            border: none;
            padding: 10px 28px;
            border-radius: 40px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
            color: #5a6874;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        }
        
        .tab.active {
            background: #0033A0;
            color: white;
            box-shadow: 0 4px 12px rgba(0, 51, 160, 0.3);
        }
        
        .tab:hover:not(.active) {
            background: #e8eeff;
            transform: translateY(-1px);
        }
        
        /* Cards */
        .card {
            background: white;
            border-radius: 24px;
            padding: 28px;
            margin-bottom: 28px;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.06);
        }
        
        .card h2 {
            color: #0033A0;
            font-size: 20px;
            font-weight: 700;
            margin-bottom: 20px;
        }
        
        /* Stats */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 20px;
            margin-bottom: 28px;
        }
        
        .stat-card {
            background: linear-gradient(135deg, #0033A0, #002880);
            border-radius: 20px;
            padding: 22px;
            text-align: center;
            color: white;
            transition: transform 0.2s;
        }
        
        .stat-card:hover { transform: translateY(-3px); }
        
        .stat-number { font-size: 38px; font-weight: 800; }
        .stat-label { font-size: 12px; opacity: 0.9; margin-top: 6px; }
        
        /* Chat */
        .chat-area {
            height: 420px;
            overflow-y: auto;
            border: 1px solid #eef2f6;
            border-radius: 20px;
            padding: 20px;
            background: #fafcff;
            margin-bottom: 20px;
        }
        
        .message { display: flex; margin-bottom: 16px; }
        .message.user { justify-content: flex-end; }
        
        .bubble {
            max-width: 70%;
            padding: 12px 18px;
            border-radius: 22px;
            font-size: 14px;
            line-height: 1.45;
        }
        
        .message.user .bubble {
            background: #0033A0;
            color: white;
            border-bottom-right-radius: 6px;
        }
        
        .message.bot .bubble {
            background: #f0f4f9;
            color: #1a2c3e;
            border-bottom-left-radius: 6px;
        }
        
        .intent-badge { font-size: 10px; opacity: 0.7; margin-top: 6px; }
        
        /* Inputs */
        .input-group {
            display: flex;
            gap: 12px;
            flex-wrap: wrap;
        }
        
        .input-group input, .input-group textarea {
            flex: 1;
            padding: 12px 16px;
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            font-size: 14px;
            font-family: inherit;
            transition: all 0.2s;
        }
        
        .input-group input:focus, .input-group textarea:focus {
            outline: none;
            border-color: #0033A0;
            box-shadow: 0 0 0 3px rgba(0, 51, 160, 0.1);
        }
        
        /* Buttons */
        .btn {
            background: #0033A0;
            color: white;
            border: none;
            padding: 12px 26px;
            border-radius: 14px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .btn:hover {
            background: #002880;
            transform: translateY(-2px);
        }
        
        .btn-outline {
            background: transparent;
            color: #0033A0;
            border: 1.5px solid #0033A0;
        }
        
        .btn-outline:hover {
            background: #e8eeff;
            transform: none;
        }
        
        .btn-danger { background: #dc3545; }
        .btn-danger:hover { background: #bb2d3b; }
        .btn-success { background: #28a745; }
        .btn-success:hover { background: #1e7e34; }
        
        .file-btn {
            background: #f8fafc;
            padding: 12px 20px;
            border-radius: 14px;
            cursor: pointer;
            border: 1px solid #e2e8f0;
            font-weight: 600;
            transition: all 0.2s;
        }
        
        .file-btn:hover { background: #eef2f6; }
        input[type="file"] { display: none; }
        
        /* Toolbar */
        .toolbar {
            display: flex;
            gap: 12px;
            flex-wrap: wrap;
            margin-bottom: 20px;
        }
        
        .filter-select {
            padding: 10px 16px;
            border-radius: 12px;
            border: 1px solid #e2e8f0;
            background: white;
            font-size: 14px;
            cursor: pointer;
        }
        
        /* Table */
        .table-wrapper { overflow-x: auto; border-radius: 16px; }
        table { width: 100%; border-collapse: collapse; font-size: 13px; }
        th, td { padding: 14px 16px; text-align: left; border-bottom: 1px solid #eef2f6; }
        th { background: #f8fafd; color: #0033A0; font-weight: 600; }
        tr:hover td { background: #fafcff; }
        
        .badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 30px;
            font-size: 11px;
            font-weight: 700;
        }
        .badge.saldo { background: #e6f9ee; color: #1e7a3c; }
        .badge.prestamo { background: #e3f6fb; color: #0d6e8a; }
        .badge.apelacion { background: #fff8e1; color: #b8860b; }
        .badge.asesor { background: #fdecea; color: #b71c1c; }
        .badge.general { background: #f0f0f0; color: #555; }
        
        /* Charts */
        .charts-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 28px;
        }
        canvas { max-height: 280px; }
        
        /* Loader */
        .loader {
            display: inline-block;
            width: 18px;
            height: 18px;
            border: 2px solid #e2e8f0;
            border-top-color: #0033A0;
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
            margin-right: 8px;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
        
        /* Footer */
        .footer {
            text-align: center;
            padding: 20px;
            color: #6c757d;
            font-size: 11px;
        }
        
        @media (max-width: 768px) {
            .container { padding: 16px; }
            .header { flex-direction: column; text-align: center; }
            .stat-number { font-size: 32px; }
            .charts-grid { grid-template-columns: 1fr; }
            .bubble { max-width: 85%; }
        }
    </style>
</head>
<body>
<div class="container">
    <!-- Header BBVA -->
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

    <!-- Tabs -->
    <div class="tabs">
        <button class="tab active" onclick="showTab('chat')">💬 Chat Web</button>
        <button class="tab" onclick="showTab('whatsapp')">📱 WhatsApp</button>
        <button class="tab" onclick="showTab('dashboard')">📊 Dashboard</button>
    </div>

    <!-- Chat -->
    <div id="chat-section">
        <div class="card">
            <h2>💬 Simulador de Conversación</h2>
            <div class="chat-area" id="chatMessages">
                <div class="message bot">
                    <div class="bubble">
                        🙋 ¡Hola! Soy el asistente virtual BBVA Colombia.<br><br>
                        💰 <strong>Saldo</strong> · Consulta tu dinero<br>
                        🏦 <strong>Préstamo</strong> · Solicita un crédito<br>
                        📄 <strong>Apelación</strong> · Adjunta un PDF<br>
                        👨‍💼 <strong>Asesor</strong> · Habla con un humano<br><br>
                        ¿En qué te ayudo?
                    </div>
                </div>
            </div>
            <div class="input-group">
                <input type="text" id="telefono" placeholder="Teléfono" value="573053667704">
                <textarea id="mensaje" placeholder="Escribe tu mensaje..." rows="2"></textarea>
                <label class="file-btn" for="pdfFile">📎 Adjuntar PDF</label>
                <input type="file" id="pdfFile" accept="application/pdf">
                <button class="btn" onclick="enviarMensaje()">✉️ Enviar</button>
                <button class="btn-outline" onclick="limpiarChat()">🔄 Limpiar</button>
            </div>
        </div>
    </div>

    <!-- WhatsApp -->
    <div id="whatsapp-section" style="display:none">
        <div class="card">
            <h2>📱 Mensajes WhatsApp</h2>
            <div class="toolbar">
                <button class="btn" onclick="cargarWhatsApp()">🔄 Actualizar</button>
                <button class="btn-success" onclick="exportarExcel()">📊 Excel</button>
                <button class="btn-outline" onclick="exportarCSV()">📄 CSV</button>
            </div>
            <div id="waStats" style="display:flex; gap:16px; flex-wrap:wrap; margin-bottom:20px"></div>
            <div id="waTabla"></div>
        </div>
    </div>

    <!-- Dashboard -->
    <div id="dashboard-section" style="display:none">
        <div class="stats-grid" id="statsContainer"></div>
        <div class="charts-grid">
            <div class="card"><h3>📈 Tendencia (7 días)</h3><canvas id="trendChart"></canvas></div>
            <div class="card"><h3>🥧 Intenciones</h3><canvas id="intencionesChart"></canvas></div>
        </div>
        <div class="card">
            <h3>📋 Conversaciones Recientes</h3>
            <div class="toolbar">
                <select class="filter-select" id="filtroCanal" onchange="cargarDashboard()">
                    <option value="">Todos</option><option value="web">🌐 Web</option><option value="whatsapp">💬 WhatsApp</option>
                </select>
                <button class="btn-success" onclick="exportarExcel()">📊 Excel</button>
                <button class="btn-outline" onclick="exportarCSV()">📄 CSV</button>
                <button class="btn-danger" onclick="borrarTodo()">🗑️ Borrar</button>
                <button class="btn" onclick="cargarDashboard()">🔄 Actualizar</button>
            </div>
            <div class="table-wrapper" id="tablaConversaciones"></div>
        </div>
    </div>

    <div class="footer">BBVA Colombia Asistente Inteligente · Datos en tiempo real</div>
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
}

async function enviarMensaje() {
    const telefono = document.getElementById('telefono').value;
    const mensaje = document.getElementById('mensaje').value;
    const pdfFile = document.getElementById('pdfFile').files[0];
    if (!mensaje.trim()) { alert('Escribe un mensaje'); return; }
    
    const chatDiv = document.getElementById('chatMessages');
    chatDiv.innerHTML += `<div class="message user"><div class="bubble">${escapeHtml(mensaje)}${pdfFile ? ' 📎 ' + escapeHtml(pdfFile.name) : ''}</div></div>`;
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
    
    try {
        const res = await fetch('/api/simular', { method: 'POST', body: fd });
        const data = await res.json();
        loading.remove();
        chatDiv.innerHTML += `<div class="message bot"><div class="bubble">🤖 ${escapeHtml(data.respuesta)}<div class="intent-badge">🎯 ${data.intencion} ${data.escalado ? '· ⚠️ Escalado' : ''}</div></div></div>`;
        chatDiv.scrollTop = chatDiv.scrollHeight;
        if (data.escalado) cargarDashboard();
    } catch(e) {
        loading.remove();
        chatDiv.innerHTML += `<div class="message bot"><div class="bubble">❌ Error</div></div>`;
    }
}

function limpiarChat() {
    document.getElementById('chatMessages').innerHTML = `<div class="message bot"><div class="bubble">🙋 ¡Hola! Soy el asistente BBVA.<br><br>💰 Saldo<br>🏦 Préstamo<br>📄 Apelación<br>👨‍💼 Asesor<br><br>¿En qué te ayudo?</div></div>`;
}

async function cargarWhatsApp() {
    try {
        const res = await fetch('/api/conversaciones?canal=whatsapp');
        const data = await res.json();
        const esc = data.filter(c => c.escalado).length;
        document.getElementById('waStats').innerHTML = `
            <div style="background:#25D366;border-radius:20px;padding:18px 28px;color:white"><div style="font-size:32px;font-weight:800">📱 ${data.length}</div><div>Mensajes WhatsApp</div></div>
            <div style="background:#dc3545;border-radius:20px;padding:18px 28px;color:white"><div style="font-size:32px;font-weight:800">🚨 ${esc}</div><div>Escalados</div></div>
        `;
        if (!data.length) { document.getElementById('waTabla').innerHTML = '<div style="text-align:center;padding:60px">Sin mensajes</div>'; return; }
        let h = '<div class="table-wrapper"><tr><thead><tr><th>Fecha</th><th>Teléfono</th><th>Mensaje</th><th>Intención</th></tr></thead><tbody>';
        for (let c of data.slice(0,30)) {
            h += `<tr><td>${c.fecha?.substring(0,16) || ''}</td><td>${c.telefono}</td><td>${escapeHtml((c.mensaje||'').substring(0,50))}</td><td><span class="badge ${c.intencion}">${c.intencion||'general'}</span></td></tr>`;
        }
        h += '</tbody></table></div>';
        document.getElementById('waTabla').innerHTML = h;
    } catch(e) { console.error(e); }
}

async function cargarDashboard() {
    try {
        const s = await (await fetch('/api/estadisticas')).json();
        document.getElementById('statsContainer').innerHTML = `
            <div class="stat-card"><div class="stat-number">${s.total}</div><div class="stat-label">Total</div></div>
            <div class="stat-card"><div class="stat-number">${s.canales?.whatsapp||0}</div><div class="stat-label">WhatsApp</div></div>
            <div class="stat-card"><div class="stat-number">${s.escalados}</div><div class="stat-label">Escalados</div></div>
            <div class="stat-card"><div class="stat-number">${s.tasa_escalado}%</div><div class="stat-label">Tasa</div></div>
            <div class="stat-card"><div class="stat-number">${s.pdfs||0}</div><div class="stat-label">PDFs</div></div>
        `;
        if (trendChart) trendChart.destroy();
        trendChart = new Chart(document.getElementById('trendChart'), {
            type: 'line', data: { labels: s.tendencia_fechas, datasets: [{ label: 'Conversaciones', data: s.tendencia_conteos, borderColor: '#0033A0', fill: true }] }
        });
        if (intencionesChart) intencionesChart.destroy();
        intencionesChart = new Chart(document.getElementById('intencionesChart'), {
            type: 'pie', data: { labels: Object.keys(s.intenciones), datasets: [{ data: Object.values(s.intenciones), backgroundColor: ['#28a745','#17a2b8','#ffc107','#dc3545','#6c757d'] }] }
        });
        const canal = document.getElementById('filtroCanal').value;
        const conv = await (await fetch('/api/conversaciones?canal='+canal)).json();
        let h = '<div class="table-wrapper"><table><thead><tr><th>ID</th><th>Fecha</th><th>Teléfono</th><th>Mensaje</th><th>Intención</th><th>PDF</th><th>Descargar</th></tr></thead><tbody>';
        for (let c of conv.slice(0,30)) {
            let btn = c.pdf_nombre ? `<button onclick="descargarPDF(${c.id})" style="background:#28a745;border:none;padding:5px10px;border-radius:8px;color:white;cursor:pointer">📥</button>` : '—';
            h += `<tr><td>${c.id}</td><td>${c.fecha?.substring(0,16)||''}</td><td>${c.telefono}</td><td>${escapeHtml((c.mensaje||'').substring(0,40))}</td><td><span class="badge ${c.intencion}">${c.intencion||'general'}</span></td><td>${c.pdf_nombre?'📄':'—'}</td><td>${btn}</td></tr>`;
        }
        h += '</tbody></table></div>';
        document.getElementById('tablaConversaciones').innerHTML = h;
    } catch(e) { console.error(e); }
}

async function descargarPDF(id) { window.open(`/api/descargar-pdf/${id}`, '_blank'); }
async function borrarTodo() { if(confirm('¿Borrar todo?')){ await fetch('/api/conversaciones',{method:'DELETE'}); cargarDashboard(); cargarWhatsApp(); } }
function exportarExcel() { window.open('/api/exportar/excel', '_blank'); }
function exportarCSV() { window.open('/api/exportar/csv', '_blank'); }
function escapeHtml(t) { const d=document.createElement('div'); d.textContent=t; return d.innerHTML; }
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
    print("=" * 50)
    uvicorn.run(app, host="127.0.0.1", port=8000)