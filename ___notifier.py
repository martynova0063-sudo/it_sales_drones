# notifier.py

# notifier.py
import base64
import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import config_monitor

app = FastAPI(title="Drone Notifier Service")

class Alert(BaseModel):
    node_id: str
    timestamp: str
    class_name: str
    confidence: float
    bbox: list[int]
    image_base64: str  # Картинка в формате Base64

def send_telegram_photo(caption: str, image_data: bytes):
    """
    Отправляет фото в Telegram через Bot API.
    """
    token = config_monitor.TELEGRAM_BOT_TOKEN
    chat_id = config_monitor.TELEGRAM_CHAT_ID

    if not token or not chat_id:
        print("[WARNING] Telegram token or Chat ID not set in .env")
        return

    url = f"https://api.telegram.org/bot{token}/sendPhoto"
    
    # Формируем данные формы
    payload = {
        "chat_id": chat_id,
        "caption": caption
    }
    
    # Файл передаем в словаре files
    # ('photo', (filename, file_bytes, mime_type))
    files = {
        "photo": ("alert.jpg", image_data, "image/jpeg")
    }

    try:
        response = requests.post(url, data=payload, files=files, timeout=10)
        response.raise_for_status()
        print(f"[INFO] Sent to Telegram: {caption.replace(chr(10), ' ')}") # chr(10) это \n
    except Exception as e:
        print(f"[ERROR] Telegram send failed: {e}")

@app.post("/alert")
def receive_alert(alert: Alert):
    """
    Принимает алерт от Ноды, декодирует картинку и шлет в ТГ.
    """
    # 1. Фильтрация (на всякий случай дублируем, или если хотим более строгий отбор)
    # Сейчас пропускаем всё, что прислала Нода, так как она уже отфильтровала "drone"
    
    print(f"[RECV] Alert from Node {alert.node_id}: {alert.class_name} ({alert.confidence:.2f})")

    # 2. Декодируем Base64 обратно в байты картинки
    try:
        image_data = base64.b64decode(alert.image_base64)
    except Exception as e:
        print(f"[ERROR] Failed to decode base64 image: {e}")
        raise HTTPException(status_code=400, detail="Invalid base64 image")

    # 3. Формируем сообщение
    # Можно добавить эмодзи или форматирование
    msg = (
        f"<b>ОБНАРУЖЕН ОБЪЕКТ</b>\n"
        f"<b>Камера:</b> {alert.node_id}\n"
        f"<b>Класс:</b> {alert.class_name}\n"
        f"<b>Вероятность:</b> {alert.confidence:.2f}\n"
        f"<b>Время:</b> {alert.timestamp}"
    )

    # 4. Отправляем в Telegram
    # Функция синхронная, но FastAPI запустит её в потоке, не блокируя сервер
    send_telegram_photo(msg, image_data)

    return {"status": "ok", "message": "Alert processed"}

if __name__ == "__main__":
    # Запуск сервера на порту 9000
    print("Starting Notifier on port 9000...")
    uvicorn.run(app, host="0.0.0.0", port=9000)