# app.py

import numpy as np
import gradio as gr
import config_monitor as config
import threading
import time
import cv2
import json

from typing import Generator
from src.video_source import VideoSource
from src.hls_video_source import HLSVideoSource
from src.stream_refresher import get_fresh_hls_url
from src.detector import YOLODetector
from src.video_writer import ChunkedVideoWriter
from src.detection_logger import DetectionLogger

def load_stream_cameras() -> list[tuple[str, str]]:
    """Загружает список камер из configs/stream_sources.json.
    
    Returns:
        Список кортежей (отображаемое_имя, page_url).
    """
    try:
        with open("configs/stream_sources.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        return [(cam["name"], cam["page_url"]) for cam in data.get("cameras", [])]
    except Exception as e:
        print(f"Ошибка загрузки списка камер: {e}")
        return [("Ошибка загрузки", "")]
    
# Глобальные переменные для управления потоком
processing = False
thread = None

def run_inference(model_choice: str, model_path: str, source_type: str, video_file: dict, camera_index: int, stream_camera: str | None) -> Generator[tuple[str, np.ndarray | None, np.ndarray | None], None, None]:    
    """
    Выполняет инференс модели на видеопотоке и возвращает кадры для Gradio.

    Args:  
        model_choice: Выбор модели — "default" или "custom".  
        model_path: Путь к кастомной модели (.pt), если выбрано "custom".  
        source_type: Тип источника — "camera" или "file".  
        video_path: Путь к видеофайлу (если source_type == "file").  
        camera_index: Индекс камеры (если source_type == "camera"). 
        stream_camera: Ссылка на камеру в интернете

    Yields:  
        tuple[str, np.ndarray | None, np.ndarray | None]:  
        - статусное сообщение  
        - оригинальный кадр в RGB  
        - аннотированный кадр в RGB  
    """

    global processing
    processing = True

    # Выбор модели
    weights = model_path if model_choice == "custom" else config.DEFAULT_MODEL_WEIGHTS
    detector = YOLODetector(weights)

    # Источник - файл
    src = None
    if source_type == "file":
        # 1. Определение пути к файлу
        # Gradio.File возвращает Path (если file_count='single') или список/объект FileData
        if isinstance(video_file, str): 
            src = video_file # Прямой путь к файлу (устаревшая, но частая форма)
        elif isinstance(video_file, dict) and 'name' in video_file:
            src = video_file['name'] # Объект Gradio.FileData (как словарь)
        elif isinstance(video_file, list) and video_file and 'name' in video_file[0]:
            src = video_file[0]['name'] # Список файлов (берем первый)
        else:
            # Выводим ошибку, если не удалось извлечь путь
            yield "Ошибка: Файл не выбран или не загружен", None, None
            return
        
        if src is None:
            yield "Ошибка: Файл не выбран или не загружен", None, None
            return
        
    # Источник - камера
    elif source_type == "camera":
        # src = camera_index
        src = int(camera_index) if isinstance(camera_index, (int, float)) else 0

    # Источник - поток по ссылке
    elif source_type == "stream":

        if not stream_camera or not stream_camera.strip():
            yield "Ошибка: камера не выбрана", None, None
            return

        # Получаем свежий .m3u8 URL через Selenium
        from src.stream_refresher import get_fresh_hls_url
        fresh_url = get_fresh_hls_url(
            camera_page_url=stream_camera,
            referer=stream_camera
        )
        if not fresh_url:
            yield "Ошибка: не удалось получить поток с камеры", None, None
            return

        # Создаём источник с динамическим referer
        video_source = HLSVideoSource(
            stream_url=fresh_url,
            referer=stream_camera,
            page_url=stream_camera
        )

    else:
        yield "Ошибка: неизвестный тип источника", None, None
        return        

###     # 2. Проверка и Инициализация
###     if src is None:
###         yield "Ошибка: Файл не выбран или не загружен", None, None
###         return

    # Защита от ошибочных данных (например, если Gradio передал 0 для камеры, но source_type=file)
    if source_type == "file" and (not src or src == 0):
        yield "Ошибка: Файл не выбран или не загружен", None, None
        return
        
    # Создаём источник только для file/camera (для stream он уже создан)
    if source_type != "stream":
        try:
            video_source = VideoSource(source_type, src)
        except ValueError as e:
            yield f"Ошибка инициализации источника: {e}", None, None
            return

    # Писатели и логгер
    logger = DetectionLogger(config.LOG_DIR)
    logger.start_new_session()

    fps = video_source.fps
    frame = video_source.read_frame()
    if frame is None:
        yield "Ошибка: не удалось прочитать первый кадр", None, None
        return
    h, w = frame.shape[:2]
    video_writer = ChunkedVideoWriter(config.OUTPUT_VIDEO_DIR, fps, (w, h), config.CHUNK_DURATION_SEC)

    frame_count = 0
    while processing and frame is not None:
        orig, annotated, boxes = detector.predict_and_track(frame)
        video_writer.write(annotated)
        
        names_list = [name for _, name in sorted(detector.model.names.items())]
        logger.log(boxes, names_list)

        # Отправляем в Gradio
        orig_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        # Преобразуем names из dict в list, упорядоченный по class_id
        names_list = [name for _, name in sorted(detector.model.names.items())]

        yield (
            f"Кадр {frame_count}, обнаружено объектов: {len(boxes)}",
            orig_rgb,
            annotated_rgb
        )

        # Проверяем, нужно ли обновить поток (только для HLS)
        if isinstance(video_source, HLSVideoSource):
            if not video_source.is_alive():
                logger.warning("Поток умер — попытка обновления...")
                if video_source.refresh():
                    fps = video_source.fps  # обновляем FPS
                    logger.info("Поток успешно обновлён")
                else:
                    yield "Ошибка: не удалось обновить поток", None, None
                    break

        frame = video_source.read_frame()
        frame_count += 1
###         time.sleep(1 / fps)  # имитация реального fps

        # Запись в логгер
        logger.log(boxes, names_list)

    # Очистка
    video_source.release()
    video_writer.release()
    logger.close()
    processing = False
    yield "Обработка завершена", None, None

def stop_processing() -> None:
    """
    Останавливает текущую сессию обработки видео.
    """
    global processing
    processing = False

with gr.Blocks(title="Drone Detector MVP") as demo:
    gr.Markdown("Детекция дронов и летательных объектов")


    with gr.Row(equal_height=True):
        # 1. Блок Модели
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("Модель")
                model_choice = gr.Radio(["default", "custom"], label="Выбор модели", value="default", type="value", container=False)
                model_path = gr.Textbox(label="Путь к кастомной модели (.pt)", visible=False)
                model_choice.change(
                    lambda x: gr.update(visible=(x == "custom")),
                    inputs=model_choice,
                    outputs=model_path
                )
        # 2. Блок Источника
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("Источник видео")
                source_type = gr.Radio(["camera", "file", "stream"], label="Тип источника", value="file", type="value", container=False)
                ### stream_url = gr.Textbox(label="URL видеопотока (HLS/RTSP)", visible=False)

                # Поля ввода в отдельной колонке для вертикального расположения
                with gr.Column():
                    video_file = gr.File(label="Выберите видеофайл (mp4, avi, ...)", file_count='single', visible=True)             
                    camera_index = gr.Number(label="Индекс камеры", value=0, visible=False)
                    ### stream_url = gr.Textbox(label="URL видеопотока (HLS/RTSP)", visible=False)
                    stream_camera = gr.Dropdown(choices=load_stream_cameras(), label="Выберите камеру", visible=False)

                source_type.change(
                    lambda x: (
                        gr.update(visible=(x == "file")),
                        gr.update(visible=(x == "camera")),
                        gr.update(visible=(x == "stream"))
                    ),
                    inputs=source_type,
                    outputs=[video_file, camera_index, stream_camera]
                )
            
        # 3. Кнопки (в отдельной строке для правого выравнивания)
        with gr.Column(scale=1):
            with gr.Row():
                start_btn = gr.Button("▶️ Запустить", scale=1)
                stop_btn = gr.Button("⏹️ Остановить", scale=1)

    with gr.Row():
        status = gr.Textbox(label="Статус", interactive=False)
            
    with gr.Row():
        orig_out = gr.Image(label="Оригинал", streaming=True, height="auto")
        proc_out = gr.Image(label="С детекцией", streaming=True, height="auto")

    start_btn.click(
        fn=run_inference,
        inputs=[model_choice, model_path, source_type, video_file, camera_index, stream_camera],
        outputs=[status, orig_out, proc_out]
    )

    stop_btn.click(fn=stop_processing)

demo.launch()