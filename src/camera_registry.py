# src/camera_registry.py

import time
import numpy as np
import cv2
from PIL import Image
from typing import Dict, Any, Optional, List
import collections

import config_monitor
from src.runtime_logger import get_runtime_logger
from src.schemas import AlertPayload

logger = get_runtime_logger("CameraRegistry")

class CameraState:
    """
    Хранит текущее состояние одной камеры: последний кадр, статус, информацию о тревоге,
    метаданные и историю последнего обнаруженного объекта для зума.
    """
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.last_seen = time.time()
        self.image: Optional[Image.Image] = None # Последний кадр PIL Image
        self.status = "active" # "active", "alert", "offline"
        self.alert_active = False
        self.alert_info = ""
        self.camera_name = "Загрузка..."
        self.model_name = "-"
        self.source_type = "-"
        self.stream_url = ""
        self.last_bbox: List[int] = [] # Последний bounding box объекта, вызвавшего тревогу
        self.last_alert_timestamp_iso: Optional[str] = None # Для хранения точного timestamp тревоги
        self.all_classes_names: List[str] = []
        self.target_classes: List[str] = []
        self.last_prediction_iou: Optional[float] = None # Последний IoU score точности предсказания

    def update(self, payload: "AlertPayload", pil_image: Image.Image):
        """
        Обновляет состояние камеры на основе полученного Payload.

        Args:
            payload: Объект AlertPayload с данными от ноды.
            pil_image: Декодированное изображение в формате PIL Image.
        """
        self.last_seen = time.time()
        
        # Гарантируем RGB
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        self.image = pil_image
        
        # Обновляем метаданные
        if payload.camera_name and payload.camera_name != "Unknown Camera":
            self.camera_name = payload.camera_name
        if payload.model_name and payload.model_name != "Unknown Model":
            self.model_name = payload.model_name

        if payload.stream_url:
            self.stream_url = payload.stream_url

        # Обновляем списки классов, если они пришли в Payload
        if payload.all_classes_names is not None:
            self.all_classes_names = payload.all_classes_names
        if payload.target_classes is not None:
            self.target_classes = payload.target_classes

        if payload.type == "alert":
            self.alert_active = True
            self.status = "alert"
            cls = payload.class_name or "Unknown"
            conf = payload.confidence or 0.0
            t_str = payload.timestamp.split("T")[-1].split(".")[0]
            self.alert_info = f"{cls} ({conf:.2f}) at {t_str}"
            self.last_alert_timestamp_iso = payload.timestamp
            self.last_prediction_iou = payload.prediction_iou_score

            # Сохраняем бокс только если он есть (чтобы хартбит не затер)
            if payload.bbox and len(payload.bbox) == 4:
                self.last_bbox = payload.bbox

        elif payload.type == "heartbeat":
            # Не сбрасываем alert_active, если он уже был установлен, чтобы оператор мог сбросить вручную
            if not self.alert_active:
                self.status = "active"
                self.alert_info = ""
                self.last_bbox = []
                self.last_alert_timestamp_iso = None
                self.last_prediction_iou = None

    def get_thumbnail_image(self) -> Image.Image:
        """
        Возвращает изображение, кропнутое вокруг последнего обнаруженного объекта (для зума).
        Если объекта нет, возвращает полноразмерное изображение.
        """
        if self.image is None:
            return self._get_no_signal_image()
            
        base_img = self.image # PIL Image
        
        if self.alert_active and self.last_bbox and len(self.last_bbox) == 4:
            try:
                x1, y1, x2, y2 = map(int, self.last_bbox)
                w_orig, h_orig = base_img.size
                
                target_aspect = w_orig / h_orig

                obj_cx = (x1 + x2) / 2
                obj_cy = (y1 + y2) / 2

                obj_w = x2 - x1
                obj_h = y2 - y1

                # Увеличиваем область кропа вокруг объекта
                crop_h = obj_h * 2.5 # Чуть больше зум, чтобы было видно контекст
                crop_w = crop_h * target_aspect

                if crop_w < obj_w * 1.5:
                    crop_w = obj_w * 1.5
                    crop_h = crop_w / target_aspect

                # Ограничиваем размеры кропа пределами оригинального изображения
                crop_w = min(crop_w, w_orig)
                crop_h = min(crop_h, h_orig)

                crop_x1 = int(obj_cx - crop_w / 2)
                crop_y1 = int(obj_cy - crop_h / 2)
                crop_x2 = int(obj_cx + crop_w / 2)
                crop_y2 = int(obj_cy + crop_h / 2)

                # Сдвигаем рамку, если она выходит за границы
                if crop_x1 < 0: crop_x2 -= crop_x1; crop_x1 = 0
                if crop_y1 < 0: crop_y2 -= crop_y1; crop_y1 = 0
                if crop_x2 > w_orig: crop_x1 -= (crop_x2 - w_orig); crop_x2 = w_orig
                if crop_y2 > h_orig: crop_y1 -= (crop_y2 - h_orig); crop_y2 = h_orig

                crop_x1 = max(0, crop_x1)
                crop_y1 = max(0, crop_y1)
                crop_x2 = min(w_orig, crop_x2)
                crop_y2 = min(h_orig, crop_y2)

                crop = base_img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
                crop = crop.resize((w_orig, h_orig), Image.Resampling.LANCZOS)
                
                return crop
            
            except Exception as e:
                logger.error(f"[ERROR] Ошибка зума для камеры {self.node_id}: {e}")
                pass 
        
        return self.get_display_image()

    def get_display_image(self) -> Image.Image:
        """
        Возвращает полноразмерное изображение с цветовой индикацией статуса.
        """
        if self.image is None:
            return self._get_no_signal_image()

        img_np = np.array(self.image)
        
        color = (0, 255, 0) # Зеленый (BGR)
        time_diff = time.time() - self.last_seen

        if self.alert_active:
            color = (255, 0, 0) # Красный
        elif time_diff > config_monitor.TIMEOUT_OFFLINE:
            color = (128, 128, 128) # Серый
            self.status = "offline"
            overlay = img_np.copy()
            cv2.rectangle(overlay, (0,0), (img_np.shape[1], img_np.shape[0]), (50,50,50), -1)
            cv2.addWeighted(overlay, 0.7, img_np, 0.3, 0, img_np)
            cv2.putText(img_np, "OFFLINE", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3)
        elif time_diff > config_monitor.TIMEOUT_WARNING:
            color = (0, 200, 255) # Оранжевый

        if len(img_np.shape) == 3:
            cv2.rectangle(img_np, (0, 0), (img_np.shape[1], img_np.shape[0]), color, 15)
        
        return Image.fromarray(img_np)

    def _get_no_signal_image(self) -> Image.Image:
        """Создает заглушку "NO SIGNAL"."""
        arr = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.putText(arr, "NO SIGNAL", (int(arr.shape[1]*0.3), int(arr.shape[0]*0.5)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return Image.fromarray(arr)

class CameraRegistry:
    """
    Реестр для централизованного управления состоянием всех подключенных камер.
    """
    def __init__(self):
        self._cameras: Dict[str, CameraState] = collections.OrderedDict() # Используем OrderedDict для сохранения порядка
        self._selected_camera_id: Optional[str] = None
        self.logger = get_runtime_logger("CameraRegistry")

    def update_camera_state(self, node_id: str, payload: "AlertPayload", pil_image: Image.Image):
        """
        Обновляет или добавляет состояние камеры.

        Args:
            node_id: ID ноды (камеры).
            payload: Объект AlertPayload.
            pil_image: Декодированное изображение.
        """
        if node_id not in self._cameras:
            self._cameras[node_id] = CameraState(node_id)
            self.logger.info(f"Добавлена новая камера в реестр: {node_id}")
            # Автоматически выбираем первую добавленную камеру
            if self._selected_camera_id is None:
                self._selected_camera_id = node_id
        
        self._cameras[node_id].update(payload, pil_image)

    def get_all_camera_ids(self) -> List[str]:
        """Возвращает список всех ID зарегистрированных камер."""
        return list(self._cameras.keys())

    def get_camera_by_id(self, node_id: str) -> Optional[CameraState]:
        """Возвращает объект CameraState для заданного ID."""
        return self._cameras.get(node_id)

    def get_selected_camera_id(self) -> Optional[str]:
        """Возвращает ID текущей выбранной камеры."""
        return self._selected_camera_id

    def set_selected_camera_id(self, node_id: str):
        """Устанавливает ID выбранной камеры."""
        if node_id in self._cameras:
            self._selected_camera_id = node_id
            self.logger.debug(f"Выбрана камера: {node_id}")
        else:
            self.logger.warning(f"Попытка выбрать несуществующую камеру: {node_id}")

    def reset_alert_for_camera(self, node_id: str):
        """Сбрасывает статус тревоги для указанной камеры."""
        if node_id in self._cameras:
            self._cameras[node_id].alert_active = False
            self._cameras[node_id].alert_info = ""
            self._cameras[node_id].last_bbox = []
            self._cameras[node_id].status = "active" # Возвращаем к активному, если нет других проблем
            self.logger.info(f"Тревога сброшена для камеры: {node_id}")
        else:
            self.logger.warning(f"Попытка сбросить тревогу для несуществующей камеры: {node_id}")

    def update_camera_statuses(self):
        """
        Периодически проверяет и обновляет статусы камер на основе последнего времени активности.
        """
        current_time = time.time()
        for cam_id, cam_state in self._cameras.items():
            time_diff = current_time - cam_state.last_seen
            if cam_state.alert_active:
                # Если активна тревога, не меняем статус
                continue
            elif time_diff > config_monitor.TIMEOUT_OFFLINE:
                cam_state.status = "offline"
            elif time_diff > config_monitor.TIMEOUT_WARNING:
                cam_state.status = "warning" # Можно добавить статус "warning"
            else:
                cam_state.status = "active"
                
        # Если выбранная камера стала оффлайн и есть другие активные, можно переключиться
        if self._selected_camera_id and self._selected_camera_id in self._cameras:
            if self._cameras[self._selected_camera_id].status == "offline":
                active_cameras = [nid for nid, cs in self._cameras.items() if cs.status != "offline"]
                if active_cameras:
                    self.set_selected_camera_id(active_cameras[0])
                else:
                    self._selected_camera_id = None # Все камеры оффлайн