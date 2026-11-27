# src/detection_logger.py

from __future__ import annotations
import csv
from datetime import datetime
from pathlib import Path
from ultralytics.engine.results import Boxes

class DetectionLogger:
    """
    Логгер обнаружений объектов в видеопотоке.
    Записывает информацию о детекциях (координаты, класс, уверенность, ID трека)
    в CSV-файл с временной меткой. Один файл — одна сессия обработки.
    """

    def __init__(self, log_dir: str | Path) -> None:
        """Инициализирует логгер и создаёт директорию для логов.

        Args:
            log_dir: Путь к директории для сохранения CSV-логов.
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.file = None
        self.writer = None

    def start_new_session(self) -> None:
        """Создаёт новый CSV-файл с заголовком для текущей сессии.

        Имя файла включает временную метку в формате YYYYMMDD_HHMMSS.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.file = open(self.log_dir / f"detections_{timestamp}.csv", "w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)
        self.writer.writerow(["timestamp", "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2", "track_id"])

    def log(self, boxes: Boxes, names: list[str]) -> None:
        """Записывает детекции из одного кадра в лог-файл.

        Args:
            boxes: Объект `Boxes` из Ultralytics YOLO (результат `model.track()`).
            names: Список имён классов, соответствующих индексам в `boxes.cls`.
        """
        ts = datetime.now().isoformat()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            track_id = int(box.id[0]) if box.id is not None else -1
            self.writer.writerow([ts, cls, names[cls], conf, x1, y1, x2, y2, track_id])

    def close(self) -> None:
        """Закрывает файловый дескриптор, если он открыт."""
        if self.file:
            self.file.close()