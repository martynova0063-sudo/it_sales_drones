# src/video_source.py

import numpy as np
import cv2
from typing import Union

class VideoSource:
    """
    Источник видеопотока: камера или видеофайл.
    Обеспечивает единообразное чтение кадров независимо от типа источника.
    """

    def __init__(self, source_type: str = "camera", path: Union[str, int] = 0, loop: bool = False) -> None:
        """Инициализирует источник видео.

        Args:
            source_type: Тип источника — "camera" или "file".".
            path: Индекс камеры (int) или путь к видеофайлу (str).
            loop: Зацикливать ли видео (актуально для source_type="file").
        """
        self.loop = loop
        self.source_type = source_type

        # Проверка, что path имеет смысл (для camera=0, для file='path/to/file')
        if source_type == "camera":
            # Проверка, что индекс камеры является разумным
            if not isinstance(path, (int, float)) or path < 0:
                raise ValueError(f"Неверный индекс камеры: {path}")
            self.cap = cv2.VideoCapture(int(path))
        else:
            # Проверка, что путь к файлу не пустой
            if not path or (isinstance(path, str) and len(path) < 2) :
                 # Это ловит пустые или некорректные пути, которые Gradio может передать при инициализации
                 raise ValueError(f"Недопустимый путь к видеофайлу: {path}")
            self.cap = cv2.VideoCapture(path)

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0

        # Получаем общее количество кадров (только для файлов)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if source_type == "file" else -1

        if not self.cap.isOpened():
            raise ValueError(f"Не удалось открыть видеоисточник: source_type={source_type}, path={path}")

    def read_frame(self) -> np.ndarray | None:
        """Читает следующий кадр из видеопотока.

        Returns:
            Кадр в формате BGR (numpy array) или None, если поток завершён.
        """
        ret, frame = self.cap.read()

        # Если конец файла и включен режим петли (loop)
        if not ret and self.loop:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()

        if not ret:
            return None
        return frame

    def release(self) -> None:
        """Освобождает ресурсы видеозахвата."""
        self.cap.release()