# src/detector.py

import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results, Boxes

class YOLODetector:
    """
    Детектор объектов на основе YOLOv11 с поддержкой трекинга.
    Загружает модель, выполняет инференс и трекинг объектов в кадре,
    возвращает оригинальный и аннотированный кадры, а также боксы.
    """

    def __init__(self, model_path: str = "yolov8n.pt") -> None:
        """
        Инициализирует YOLO-модель.
        
        Args:
            model_path: Путь к весам модели (.pt).
        """
        self.model = YOLO(model_path)
        ### self.model.fuse()  # ускорение инференса

    def predict_and_track(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, Boxes]:
        """Выполняет детекцию и трекинг объектов на кадре.

        Args:
            frame: Входной кадр в формате BGR (HxWxC).

        Returns:
            Кортеж из трёх элементов:
            - оригинальный кадр (np.ndarray)
            - аннотированный кадр с боксами и треками (np.ndarray)
            - объект Boxes с результатами детекции.
        """
        results = self.model.track(frame, persist=True, verbose=False, tracker='botsort.yaml')
###         results = self.model.track(frame, persist=True, verbose=False)
        annotated_frame = results[0].plot()
        boxes = results[0].boxes
        return frame, annotated_frame, boxes