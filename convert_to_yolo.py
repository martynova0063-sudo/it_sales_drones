# convert_to_yolo.py

import os
import sys
import numpy as np
from scipy.io import loadmat
import cv2
import glob
# import pandas as pd # Хотя pandas тут необязателен, он удобен для больших таблиц
from pathlib import Path
import config

### # Добавляем корень проекта в путь, чтобы импортировать config
### PROJECT_ROOT = '/media/Cruiser/github/It_sales_drones/ALeksey_Kozlov'
### # Добавляем корень проекта в путь для корректной работы Path
### PROJECT_ROOT = Path(__file__).parent 
### 
### sys.path.append(PROJECT_ROOT)
### 
### # Проверяем, существует ли файл config.py
### if not os.path.exists(os.path.join(PROJECT_ROOT, 'config.py')):
###     print("FATAL ERROR: config.py not found in the project root.")
###     sys.exit(1)

### # Это папка, где лежат ваши V_AIRPLANE_001_1_001.png и .mat-батчи
### DATASET_MASTER_DIR = os.path.join(PROJECT_ROOT, 'data', 'Drone-detection-dataset-master') 
### IMAGES_SOURCE_DIR = os.path.join(DATASET_MASTER_DIR, 'Training_data_V')
### 
### # Выходные папки (train/images и train/labels)
### OUTPUT_BASE_DIR = os.path.join(DATASET_MASTER_DIR, 'train')
### OUTPUT_IMAGES_DIR = os.path.join(OUTPUT_BASE_DIR, 'images')
### OUTPUT_LABELS_DIR = os.path.join(OUTPUT_BASE_DIR, 'labels')
### 
### # Словарь для маппинга меток
### CLASS_MAPPING = {'AIRPLANE': 0, 'BIRD': 1, 'DRONE': 2, 'HELICOPTER': 3}



# КОНФИГУРАЦИЯ ПРОЕКТА (из config.py)
IMAGES_SOURCE_DIR = config.MATLAB_IMAGES_SOURCE_DIR # Исходная папка с PNG и MAT
OUTPUT_BASE_DIR = config.YOLO_OUTPUT_BASE_DIR # Целевая папка 'train'
OUTPUT_IMAGES_DIR = OUTPUT_BASE_DIR / 'images' # Папка train/images
OUTPUT_LABELS_DIR = OUTPUT_BASE_DIR / 'labels' # Папка train/labels
CLASS_MAPPING = config.MATLAB_CLASS_MAPPING # Словарь классов

def convert_mat_to_yolo():
    """Конвертирует все MAT-файлы батчей в формат YOLO, перемещая изображения."""
    
    # 1. Создание выходных каталогов
    os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)
    os.makedirs(OUTPUT_LABELS_DIR, exist_ok=True)
    print(f"Output directories created: {OUTPUT_IMAGES_DIR}, {OUTPUT_LABELS_DIR}")

    # 2. Поиск всех MAT-файлов батчей
    mat_files = list(IMAGES_SOURCE_DIR.glob('Training_data_V_Batch_*.mat')) 
    ### mat_files = glob.glob(os.path.join(IMAGES_SOURCE_DIR, 'Training_data_V_Batch_*.mat'))
    if not mat_files:
        print(f"ERROR: No batch MAT files found in {IMAGES_SOURCE_DIR}")
        return

    total_images_processed = 0
    
    for mat_file_path in mat_files:
        print(f"\nProcessing batch file: {os.path.basename(mat_file_path)}")
        
        try:
            # 3. Загрузка данных MATLAB
            data = loadmat(mat_file_path)


            # --- ПОИСК ИМЕНИ ПЕРЕМЕННОЙ В СЛОВАРЕ ---
            target_key = None
            for key, value in data.items():
                # Игнорируем служебные ключи и ищем массив, который не является 1x1
                if not key.startswith('__') and isinstance(value, np.ndarray) and value.ndim > 1:
                    target_key = key
                    break
            
            if target_key is None:
                # Если не нашли, выводим более информативную ошибку
                raise KeyError(f"Could not find a valid data key (expected 'trainingDataBatch' or similar). Found keys: {list(data.keys())}")

            # Таблица MATLAB имеет имя 'trainingDataBatch'
            training_data_table = data[target_key]
            
        except Exception as e:
            print(f"FATAL: Could not load or parse {mat_file_path}. Error: {e}")
            continue
            
        # 4. Итерация по таблице (каждая строка - одно изображение)
        # ВАЖНО: Структура таблицы trainingDataBatch: [Путь, BBoxes для AIRPLANE, BBoxes для BIRD, ...]
        # Нам нужно итерировать по строкам и извлекать BBoxes для каждого класса
        
        # Загружаем имена колонок, чтобы знать, где лежат метки (AIRPLANE, BIRD, ...)
        # Если вы не знаете имен колонок, их нужно извлечь из первого MAT-файла.
        # В случае objectDetectorTrainingData, колонки идут по порядку классов.
        
        # Получаем данные как массив NumPy
        data_array = training_data_table
        
        # СТРУКТУРА:
        # data_array[i, 0] = Путь к файлу (строка)
        # data_array[i, 1] = BBoxes для AIRPLANE (массив [x, y, w, h])
        # data_array[i, 2] = BBoxes для BIRD (массив [x, y, w, h])
        # ... и т.д.
        
        for row_index in range(data_array.shape[0]):
            
            # 4.1. Извлечение пути и создание имен файлов
            # Путь к картинке хранится в ячейке [0, 0] строки-объекта
            image_path_full = str(data_array[row_index, 0][0]) 
            image_filename = os.path.basename(image_path_full)
            label_filename = image_filename.replace('.png', '.txt')

            # 4.2. Получение размеров изображения
            img = cv2.imread(image_path_full)
            if img is None:
                print(f"Warning: Could not read image: {image_filename}. Skipping.")
                continue

            H, W, _ = img.shape
            
            yolo_labels = []
            
            # 4.3. Итерация по классам (начиная с 1-й колонки после пути)
            for class_index, class_name in enumerate(CLASS_MAPPING.keys()):
                
                # Получаем BBoxes для текущего класса
                # BBoxes находятся в колонке (class_index + 1)
                bboxes_cell = data_array[row_index, class_index + 1] 
                
                # Проверка, что метки есть
                if bboxes_cell.size == 0:
                    continue
                
                # BBoxes - это массив [N, 4], где N - число объектов: [x, y, w, h]
                # MATLAB сохраняет их как массив NumPy
                bboxes_np = bboxes_cell
                
                for bbox in bboxes_np:
                    x_min, y_min, w, h = bbox
                    
                    # 4.4. Конвертация в YOLO формат (Нормализация)
                    
                    # Центр: x_center = x_min + w/2
                    x_center_norm = (x_min + w / 2) / W
                    y_center_norm = (y_min + h / 2) / H
                    
                    # Ширина/высота
                    width_norm = w / W
                    height_norm = h / H
                    
                    # Получение ID класса
                    class_id = CLASS_MAPPING[class_name]
                    
                    yolo_labels.append(f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}")

            # 5. Перемещение изображения и сохранение метки
            
            # Сохранение файла .txt
            if yolo_labels:
                output_label_path = OUTPUT_LABELS_DIR / label_filename
                
                with open(output_label_path, 'w') as f:
                    f.write('\n'.join(yolo_labels))
                
                # Перемещение файла .png (Из Training_data_V в train/images)
                output_image_path = OUTPUT_IMAGES_DIR / image_filename
                os.rename(image_path_full, output_image_path)
                
                total_images_processed += 1

    # 6. Очистка (удаляем пустую исходную папку)
    if not os.listdir(IMAGES_SOURCE_DIR):
        os.rmdir(IMAGES_SOURCE_DIR)
        print(f"\nSource directory {IMAGES_SOURCE_DIR} is empty and deleted.")
        
    print("\n--- FINAL RESULT ---")
    print(f"Total images processed and moved: {total_images_processed}")
    print(f"YOLO dataset structure created at: {OUTPUT_BASE_DIR}")


if __name__ == '__main__':
    # Установка рабочего каталога для корректной работы путей
    ### os.chdir(PROJECT_ROOT)
    convert_mat_to_yolo()