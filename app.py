# app.py - Servidor Flask para Render
from flask import Flask, request, jsonify
import torch
import cv2
import numpy as np
import pandas as pd
import requests
from io import BytesIO
from PIL import Image
import os

app = Flask(__name__)

# Configuración de Telegram
TOKEN = "8107580499:AAG3FyXhtmXSPRb0To3hgZCa3WTTQm9Wfbo"
CHAT_ID = "-1002221266716"

# Cargar modelo YOLOv5 al iniciar la aplicación
def load_model():
    model_path = 'impresion.pt'  # Asegúrate de subir este archivo a Render
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
    
    # Optimizar modelo
    model.conf = 0.25
    model.iou = 0.45
    model.agnostic = False
    model.multi_label = False
    model.max_det = 1000
    
    return model

model = load_model()

def send_telegram_alert(image, detections):
    """Envía una alerta a Telegram con la imagen y las detecciones (excepto 'imprimiendo')"""
    try:
        # Filtrar detecciones para excluir 'imprimiendo'
        filtered_detections = detections[detections['name'].str.lower() != 'imprimiendo']
        
        if filtered_detections.empty:
            print("No se envía alerta: solo se detectó 'imprimiendo' (estado normal)")
            return
        
        # Convertir la imagen a bytes
        is_success, buffer = cv2.imencode(".jpg", image)
        if not is_success:
            print("Error al codificar la imagen")
            return

        # Enviar la foto
        photo_bytes = BytesIO(buffer)
        photo_bytes.seek(0)
        files = {'photo': ('detection.jpg', photo_bytes)}

        # Crear mensaje con las detecciones filtradas
        message = "⚠ *Detección de error en impresión 3D* ⚠\n\n"
        for _, row in filtered_detections.iterrows():
            message += f"🔹 *{row['name']}*\n"
            message += f"Confianza: {row['confidence']:.2f}\n"
            message += f"Posición: x1={row['xmin']:.0f}, y1={row['ymin']:.0f}, x2={row['xmax']:.0f}, y2={row['ymax']:.0f}\n\n"

        # Enviar la foto con el caption
        url = f"https://api.telegram.org/bot{TOKEN}/sendPhoto"
        data = {"chat_id": CHAT_ID, "caption": message, "parse_mode": "Markdown"}
        response = requests.post(url, data=data, files=files)

        if response.status_code != 200:
            print(f"Error al enviar alerta a Telegram: {response.text}")
        else:
            print("Alerta enviada a Telegram (errores detectados)")
            
    except Exception as e:
        print(f"Error en send_telegram_alert: {str(e)}")

@app.route('/detect', methods=['POST'])
def detect_errors():
    try:
        # Verificar si se envió una imagen
        if 'image' not in request.files:
            return jsonify({'error': 'No se proporcionó imagen'}), 400
        
        # Leer la imagen
        file = request.files['image']
        img_bytes = file.read()
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        # Realizar detección
        results = model(img)
        detections = results.pandas().xyxy[0]
        
        # Procesar resultados
        result_img = np.squeeze(results.render())
        
        # Verificar si hay errores
        error_detections = detections[detections['name'].str.lower() != 'imprimiendo']
        if not error_detections.empty:
            send_telegram_alert(result_img, detections)
            return jsonify({
                'status': 'error_detected',
                'detections': detections.to_dict('records')
            })
        else:
            return jsonify({
                'status': 'normal',
                'message': 'No se detectaron errores'
            })
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def home():
    return "Servidor de detección de errores en impresiones 3D"

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)