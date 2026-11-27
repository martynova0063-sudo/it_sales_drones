# app.py
import gradio as gr
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import base64
import io
import cv2
import numpy as np
from PIL import Image
import threading
import time
from typing import Optional, List, Any
import requests
import config_monitor

# ==========================================
# 1. КОНФИГУРАЦИЯ И МОДЕЛИ
# ==========================================

# Таймауты для статусов (в секундах)
TIMEOUT_WARNING = 35   # Если молчит 35 сек -> Желтый/Синий (потеря связи)
TIMEOUT_OFFLINE = 60   # Если молчит 60 сек -> Серый (Оффлайн)

# Глобальные настройки
global_settings = {
    "telegram_enabled": True
}

app = FastAPI()

class AlertPayload(BaseModel):
    node_id: str
    timestamp: str
    type: str = "alert"             # 'alert' или 'heartbeat'
    class_name: Optional[str] = None
    confidence: float = 0.0
    bbox: List[int] = []
    image_base64: str
    camera_name: Optional[str] = "Unknown Camera"
    model_name: Optional[str] = "Unknown Model"
    source_type: Optional[str] = "Unknown Source"

# ==========================================
# 2. МЕНЕДЖЕР СОСТОЯНИЯ КАМЕР
# ==========================================

class CameraState:
    def __init__(self, node_id):
        self.node_id = node_id
        self.last_seen = time.time()
        self.image = None           # PIL Image
        self.status = "active"      # active, alert, offline
        self.alert_active = False   # Есть ли активный, несброшенный алерт
        self.alert_info = ""        # Текст последнего алерта
        self.last_bbox = []         # Храним координаты последнего объекта
        self.camera_name = "Загрузка..."
        self.model_name = "-"
        self.source_type = "-"

    def update(self, payload: AlertPayload, pil_image):
        self.last_seen = time.time()
        self.image = pil_image
        
        # Если пришли метаданные (обычно в хартбите), обновляем их
        if payload.camera_name != "Unknown Camera":
            self.camera_name = payload.camera_name

        if payload.model_name != "Unknown Model":
            self.model_name = payload.model_name

        if payload.type == "alert":
            self.alert_active = True
            self.status = "alert"
            # Формируем описание
            cls = payload.class_name or "Unknown"
            conf = payload.confidence or 0.0
            t_str = payload.timestamp.split("T")[-1].split(".")[0]
            self.alert_info = f"{cls} ({conf:.2f}) at {t_str}"
            self.last_bbox = payload.bbox # Запоминаем бокс
        
        elif payload.type == "heartbeat":
            # Хартбит не сбрасывает статус 'alert', если оператор его не подтвердил
            if not self.alert_active:
                self.status = "active"

    def get_display_image(self):
        """Возвращает картинку с нарисованной рамкой статуса."""
        if self.image is None:
            # Заглушка, если нет картинки
            arr = np.zeros((360, 640, 3), dtype=np.uint8)
            cv2.putText(arr, f"No Signal {self.node_id}", (50, 180), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            return Image.fromarray(arr)

        # Конвертируем PIL -> CV2
        img_np = np.array(self.image)
        # Убедимся, что RGB (Gradio работает с RGB, OpenCV с BGR/RGB)
        # Обычно PIL открывает как RGB.
        
        # Определяем цвет рамки
        # BORDER: (R, G, B)
        color = (0, 255, 0) # Green (Active)

        time_diff = time.time() - self.last_seen

        if self.alert_active:
            color = (255, 0, 0) # Red (Alert)
        elif time_diff > TIMEOUT_OFFLINE:
            color = (128, 128, 128) # Grey (Offline)
            self.status = "offline"
            # Можно наложить текст поверх
            overlay = img_np.copy()
            cv2.rectangle(overlay, (0,0), (img_np.shape[1], img_np.shape[0]), (50,50,50), -1)
            cv2.addWeighted(overlay, 0.7, img_np, 0.3, 0, img_np)
            cv2.putText(img_np, "OFFLINE", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3)

        elif time_diff > TIMEOUT_WARNING:
            color = (0, 200, 255) # Light Blue/Orange (Warning)
        
        # Рисуем рамку (толщина 10px)
        cv2.rectangle(img_np, (0, 0), (img_np.shape[1], img_np.shape[0]), color, 15)
        
        return Image.fromarray(img_np)

    def get_thumbnail_image(self):
        """Возвращает кроп (если алерт) или полную картинку (если норма)."""
        base_img = self.get_display_image() # С цветной рамкой
        
        # Если тревога и есть координаты бокса - делаем КРОП
        if self.alert_active and self.last_bbox and len(self.last_bbox) == 4:
            try:
                x1, y1, x2, y2 = self.last_bbox
                w_orig, h_orig = base_img.size
                
                # Добавляем отступы (padding), чтобы объект не был впритык
                pad = 50 
                x1 = max(0, x1 - pad)
                y1 = max(0, y1 - pad)
                x2 = min(w_orig, x2 + pad)
                y2 = min(h_orig, y2 + pad)
                
                # Вырезаем
                crop = base_img.crop((x1, y1, x2, y2))
                return crop
            except Exception:
                return base_img # В случае ошибки вернем оригинал
        
        return base_img

# Глобальное хранилище: {node_id: CameraState}
cameras = {}
# ID выбранной камеры для главного экрана
selected_camera_id = None 

# ==========================================
# 3. API ENDPOINT
# ==========================================

@app.post("/alert")
async def receive_alert(payload: AlertPayload):
    try:
        image_bytes = base64.b64decode(payload.image_base64)
        pil_image = Image.open(io.BytesIO(image_bytes))
    except Exception:
        return {"status": "error", "msg": "Bad image"}

    node_id = payload.node_id
    if node_id not in cameras:
        cameras[node_id] = CameraState(node_id)
    
    cameras[node_id].update(payload, pil_image)
    
    # Отправка в Telegram (только если это Alert и включено в настройках)
    if payload.type == "alert" and global_settings["telegram_enabled"]:
        def send_tg():
            try:
                token = config_monitor.TELEGRAM_BOT_TOKEN
                chat_id = config_monitor.TELEGRAM_CHAT_ID
                if not token or not chat_id: return
                
                caption = (f"{payload.class_name} ({payload.confidence:.2f})\n"
                           f"{cameras[node_id].camera_name}\n"
                           f"ID: {node_id}")
                
                url = f"https://api.telegram.org/bot{token}/sendPhoto"
                # Используем байты, которые уже пришли
                files = {"photo": ("alert.jpg", image_bytes, "image/jpeg")}
                requests.post(url, data={"chat_id": chat_id, "caption": caption, "parse_mode": "HTML"}, files=files, timeout=5)
            except Exception as e:
                print(f"TG Error: {e}")
        
        threading.Thread(target=send_tg).start()
    
    # Если это первый алерт/камера, выберем её автоматически
    global selected_camera_id
    if selected_camera_id is None:
        selected_camera_id = node_id

    return {"status": "ok"}

# ==========================================
# 4. ЛОГИКА UI
# ==========================================

def get_thumbnails():
    """Возвращает список изображений для галереи."""
    # Сортируем камеры по ID
    sorted_ids = sorted(cameras.keys())
    imgs = []
    labels = []
    for nid in sorted_ids:
        cam = cameras[nid]

        # Получаем полную картинку
        full_img = cam.get_display_image() 
        # Делаем копию и уменьшаем для галереи (Thumbnail ~320px)
        # Это снизит нагрузку на браузер в 10-20 раз!
        thumb = full_img.copy()
        thumb.thumbnail((320, 180), Image.Resampling.LANCZOS)
        imgs.append(thumb)

#        imgs.append(cam.get_display_image())

        # Подпись под миниатюрой
        status_icon = "🔴" if cam.alert_active else "🟢"
        if time.time() - cam.last_seen > TIMEOUT_OFFLINE: status_icon = "⚫"
        labels.append(f"{status_icon} {nid}")
    
    # Заполняем пустоты, чтобы сетка не прыгала (фиксируем 4 слота для MVP)
    while len(imgs) < 4:
        # Для заглушек тоже используем маленький размер
        arr = np.zeros((180, 320, 3), dtype=np.uint8) ## ИЗМЕНЕНО (360,640 -> 180,320)
        imgs.append(Image.fromarray(arr))
        labels.append("Пустой слот")
        
    return imgs[:4], labels[:4] # Возвращаем 4 шт

def get_main_view(selected_evt: Optional[gr.SelectData] = None, zoom_enabled: bool = False):
    """Обновляет главный экран."""
    global selected_camera_id
    
    # Если кликнули по галерее
    if selected_evt:
        index = selected_evt.index
        sorted_ids = sorted(cameras.keys())
        if index < len(sorted_ids):
            selected_camera_id = sorted_ids[index]

    if not selected_camera_id or selected_camera_id not in cameras:
        # заглушки
        ### return np.zeros((480, 640, 3), dtype=np.uint8), "Камера не выбрана", gr.update(visible=False)
        ### return np.zeros((480, 640, 3), dtype=np.uint8), "Камера не выбрана", gr.update(visible=False), gr.update(visible=False)
        return np.zeros((480, 640, 3), dtype=np.uint8), "Камера не выбрана", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

    cam = cameras[selected_camera_id]

    # Если зум включен И есть активная тревога (или просто хотим зум), пытаемся кропнуть
    # Но кроп имеет смысл только если есть BBox. 
    # Зум работает, если есть Last BBox (даже если тревога уже прошла, но бокс остался в памяти)
    img = cam.get_display_image()
    if zoom_enabled:
        img = cam.get_thumbnail_image() # Берем кроп

    status_emoji = "🟢 OK"
    if cam.alert_active: status_emoji = "🔴 ТРЕВОГА"
    elif cam.status == "offline": status_emoji = "⚫ OFFLINE"

    last_seen_str = time.strftime('%H:%M:%S', time.localtime(cam.last_seen))

    # Формируем карточку, которая всегда показывает имя и модель
    info_text = f"## {cam.camera_name}\n"
    info_text += f"**ID:** `{cam.node_id}`\n"
    info_text += f"**Модель:** `{cam.model_name}`\n"
    info_text += f"**Источник:** `{cam.source_type}`\n"
    info_text += f"---\n"
    info_text += f"**Статус:** {status_emoji} | **Время:** {last_seen_str}\n"

    if cam.alert_active:
        info_text += f"\n### ОБНАРУЖЕНО:\n{cam.alert_info}"

    # Управление видимостью кнопок
    # Кнопка сброса: видна при тревоге
    btn_reset_vis = gr.update(visible=cam.alert_active)
    # Кнопка видео: видна всегда, когда выбрана камера (кроме оффлайна, но пока пусть всегда)
    btn_video_vis = gr.update(visible=True)

    # Кнопка зума: меняем текст в зависимости от состояния
    zoom_label = "Общий вид" if zoom_enabled else "Увеличить объект"
    btn_zoom_update = gr.update(visible=True, value=zoom_label, variant="secondary")

    return img, info_text, btn_reset_vis, btn_video_vis, btn_zoom_update

#    info_text = f"### Камера: {cam.node_id}\n"
#    if cam.alert_active:
#        info_text += f"**ТРЕВОГА!** {cam.alert_info}"
#    else:
#        info_text += f"Статус: Норма. Последняя активность: {time.strftime('%H:%M:%S', time.localtime(cam.last_seen))}"
#   
#    # Кнопка сброса видна только если есть алерт
#    btn_update = gr.update(visible=cam.alert_active)
    
#    return img, info_text, btn_update

def request_video():
    """Заглушка запроса видео."""
    # В будущем тут будет отправка запроса на Ноду
    print(f"[INFO] Запрос видео для камеры {selected_camera_id}")
    return

def reset_alert():
    """Сбрасывает тревогу на выбранной камере."""
    if selected_camera_id and selected_camera_id in cameras:
        cameras[selected_camera_id].alert_active = False
        cameras[selected_camera_id].status = "active"

def refresh_ui(zoom_state):
    """Периодическое обновление всего."""
    thumbs, labels = get_thumbnails()
    # gallery ожидает список кортежей (image, label)
    gallery_data = list(zip(thumbs, labels))
    
    main_img, main_info, btn_rst, btn_vid, btn_zm = get_main_view(zoom_enabled=zoom_state)
    
    return gallery_data, main_img, main_info, btn_rst, btn_vid, btn_zm

def toggle_telegram(val):
    global_settings["telegram_enabled"] = val

def toggle_zoom_state(current_state):
    return not current_state

# ==========================================
# 5. ИНТЕРФЕЙС GRADIO
# ==========================================

with gr.Blocks(title="Security Center") as demo:
    gr.Markdown("# 🛡️ Ситуационный Центр")
    
    # Секция 1: Миниатюры камер
    gr.Markdown("### Обзор камер")
    # Gallery позволяет выводить сетку картинок и обрабатывать клики
    camera_gallery = gr.Gallery(
        label="Камеры", 
        show_label=False, 
        columns=4, 
        rows=1, 
###        height=200, 
        height="auto",
        interactive=True,
        allow_preview=False # Отключаем всплывающее превью, хотим клик для выбора
    )

    # Секция 2: Главный экран
    with gr.Row():
        with gr.Column(scale=3):
            main_monitor = gr.Image(label="Основной монитор", interactive=False)
        with gr.Column(scale=1):
            cam_info = gr.Markdown("Ожидание данных...")
            
            # Панель управления
            with gr.Group():
                gr.Markdown("### Управление")
                tg_chk = gr.Checkbox(label="Отправка в Telegram", value=True, interactive=True)
                tg_chk.change(fn=toggle_telegram, inputs=tg_chk, outputs=None)
            
            gr.Markdown("---")

            # КНОПКИ УПРАВЛЕНИЯ
            with gr.Row():
                btn_zoom = gr.Button("Увеличить объект", variant="secondary", visible=False)

            with gr.Row():
                btn_reset = gr.Button("Сброс тревоги", variant="stop", visible=False)
                btn_video = gr.Button("Запросить видео", variant="secondary", visible=False)

    # Хранилище состояния зума (True/False)
    zoom_state = gr.State(False)

    # Таймер обновления (1 Гц)
    timer = gr.Timer(1)
    
    # Привязка событий
    # 1. Таймер обновляет Галерею и Главный экран и берет zoom_state на вход
    timer.tick(
        fn=refresh_ui,
        inputs=[zoom_state],
        outputs=[camera_gallery, main_monitor, cam_info, btn_reset, btn_video, btn_zoom]
    )
    
    # 2. Клик по галерее меняет выбранную камеру (через изменение selected_camera_id внутри get_main_view), а т.ж. сбрасываем зум при смене камеры
    camera_gallery.select(
        fn=lambda evt: get_main_view(evt, zoom_enabled=False),
        outputs=[main_monitor, cam_info, btn_reset, btn_video, btn_zoom]
    )

    # 3. Кнопка Зума
    # Сначала меняем стейт, потом обновляем экран
    btn_zoom.click(
        fn=toggle_zoom_state,
        inputs=[zoom_state],
        outputs=[zoom_state]
    ).then(
        fn=refresh_ui,
        inputs=[zoom_state],
        outputs=[camera_gallery, main_monitor, cam_info, btn_reset, btn_video, btn_zoom]
    )
    
    # 4. Кнопка сброса
    btn_reset.click(fn=reset_alert, inputs=None, outputs=None)

    # 5. Кнопка видео
    btn_video.click(fn=request_video, inputs=None, outputs=None)

app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    print("Запуск Ситуационного Центра на порту 9000...")
    uvicorn.run(app, host="0.0.0.0", port=9000)