from flask import Flask, request, jsonify
import torch
import cv2
import numpy as np
import pandas as pd
import requests
from io import BytesIO
from PIL import Image
import base64
import os
import logging
from datetime import datetime

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuración de Telegram
TOKEN = os.getenv("TELEGRAM_TOKEN", "8107580499:AAG3FyXhtmXSPRb0To3hgZCa3WTTQm9Wfbo")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "5973683280")

# Variable global para el modelo
model = None

def load_model():
    """Cargar el modelo YOLOv5"""
    global model
    try:
        # Instalar ultralytics si no está disponible
        import subprocess
        subprocess.check_call(['pip', 'install', 'ultralytics'])
        
        model_path = './modelo/impresion.pt'  # Ruta local del modelo
        if os.path.exists(model_path):
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
        else:
            # Si no existe el modelo local, usar uno por defecto
            logger.warning("Modelo personalizado no encontrado, usando YOLOv5s")
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        
        # Optimizar modelo
        model.conf = 0.25
        model.iou = 0.45
        model.agnostic = False
        model.multi_label = False
        model.max_det = 1000
        
        logger.info("Modelo cargado exitosamente")
        return True
    except Exception as e:
        logger.error(f"Error cargando modelo: {e}")
        return False

def send_telegram_alert(image, detections):
    """Envía una alerta a Telegram con la imagen y las detecciones (excepto 'imprimiendo')"""
    try:
        # Filtrar detecciones para excluir 'imprimiendo'
        filtered_detections = detections[detections['name'].str.lower() != 'imprimiendo']
        
        # Si no hay detecciones después del filtro, no enviar alerta
        if filtered_detections.empty:
            logger.info("No se envía alerta: solo se detectó 'imprimiendo' (estado normal)")
            return {"status": "no_alert", "message": "Estado normal - solo imprimiendo"}
        
        # Convertir la imagen a bytes
        is_success, buffer = cv2.imencode(".jpg", image)
        if not is_success:
            logger.error("Error al codificar la imagen")
            return {"status": "error", "message": "Error al codificar imagen"}

        # Enviar la foto
        photo_bytes = BytesIO(buffer)
        photo_bytes.seek(0)
        files = {'photo': ('detection.jpg', photo_bytes)}

        # Crear mensaje con las detecciones filtradas
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = f"⚠ *Detección de error en impresión 3D* ⚠\n"
        message += f"🕐 Tiempo: {timestamp}\n\n"
        
        for _, row in filtered_detections.iterrows():
            message += f"🔹 *{row['name']}*\n"
            message += f"Confianza: {row['confidence']:.2f}\n"
            message += f"Posición: x1={row['xmin']:.0f}, y1={row['ymin']:.0f}, x2={row['xmax']:.0f}, y2={row['ymax']:.0f}\n\n"

        # Enviar a Telegram
        url = f"https://api.telegram.org/bot{TOKEN}/sendPhoto"
        data = {"chat_id": CHAT_ID, "caption": message, "parse_mode": "Markdown"}
        response = requests.post(url, data=data, files=files)

        if response.status_code == 200:
            logger.info("Alerta enviada a Telegram exitosamente")
            return {"status": "sent", "message": "Alerta enviada"}
        else:
            logger.error(f"Error al enviar alerta a Telegram: {response.text}")
            return {"status": "error", "message": f"Error Telegram: {response.text}"}
            
    except Exception as e:
        logger.error(f"Error en send_telegram_alert: {str(e)}")
        return {"status": "error", "message": str(e)}

@app.route('/')
def home():
    """Endpoint de bienvenida"""
    return jsonify({
        "message": "Servidor de Detección 3D activo",
        "status": "running",
        "modelo_cargado": model is not None
    })

@app.route('/health')
def health():
    """Endpoint de salud"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "modelo_disponible": model is not None
    })

@app.route('/detect', methods=['POST'])
def detect_objects():
    """Endpoint principal para recibir imágenes y realizar detección"""
    try:
        # Verificar que el modelo esté cargado
        if model is None:
            return jsonify({"error": "Modelo no disponible"}), 500
        
        # Obtener datos de la request
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({"error": "No se proporcionó imagen"}), 400
        
        # Decodificar imagen base64
        try:
            image_data = base64.b64decode(data['image'])
            nparr = np.frombuffer(image_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                return jsonify({"error": "Error al decodificar imagen"}), 400
                
        except Exception as e:
            logger.error(f"Error decodificando imagen: {e}")
            return jsonify({"error": "Imagen inválida"}), 400
        
        # Realizar detección
        detect = model(frame)
        info = detect.pandas().xyxy[0]
        
        # Procesar resultados
        detections = []
        error_count = 0
        normal_count = 0
        
        for _, row in info.iterrows():
            detection = {
                "name": row['name'],
                "confidence": float(row['confidence']),
                "bbox": {
                    "xmin": float(row['xmin']),
                    "ymin": float(row['ymin']),
                    "xmax": float(row['xmax']),
                    "ymax": float(row['ymax'])
                }
            }
            detections.append(detection)
            
            if row['name'].lower() == 'imprimiendo':
                normal_count += 1
            else:
                error_count += 1
        
        # Enviar alerta si hay errores
        alert_result = {"status": "no_detections"}
        if len(info) > 0:
            # Renderizar imagen con detecciones
            result_img = np.squeeze(detect.render())
            alert_result = send_telegram_alert(result_img, info)
        
        # Respuesta
        response = {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "detections": detections,
            "summary": {
                "total_detections": len(detections),
                "errors_detected": error_count,
                "normal_states": normal_count
            },
            "alert": alert_result,
            "raspberry_id": data.get('raspberry_id', 'unknown')
        }
        
        logger.info(f"Detección completada: {error_count} errores, {normal_count} normales")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error en detect_objects: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/status')
def status():
    """Endpoint para verificar estado del sistema"""
    return jsonify({
        "servidor": "activo",
        "modelo": model is not None,
        "telegram_token": TOKEN is not None,
        "timestamp": datetime.now().isoformat()
    })

if __name__ == '__main__':
    # Cargar modelo al iniciar
    if load_model():
        logger.info("Servidor iniciando...")
        port = int(os.environ.get('PORT', 10000))
        app.run(host='0.0.0.0', port=port, debug=False)
    else:
        logger.error("No se pudo cargar el modelo. Servidor no iniciado.")