# yolo_converter.py
# актуальный сборщик датасета

import os
import sys
import pandas as pd
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import config
import shutil

# --- КОНФИГУРАЦИЯ ---
PROJECT_ROOT = Path(__file__).parent 
DS_MASTER_DIR = PROJECT_ROOT / 'data' / 'Drone-detection-dataset-master'
CSV_PATH = DS_MASTER_DIR / 'Training_data_FINAL_EXPORT.csv'

# Целевые папки (для YOLO-разметки)
OUTPUT_BASE_DIR = DS_MASTER_DIR / 'train'
OUTPUT_IMAGES_DIR = OUTPUT_BASE_DIR / 'images'
OUTPUT_LABELS_DIR = OUTPUT_BASE_DIR / 'labels'
OUTPUT_VIZ_DIR = OUTPUT_BASE_DIR / 'visualize'

# Словарь для маппинга меток (на основе вашего MATLAB-кода)
CLASS_MAPPING = {'AIRPLANE': 0, 'BIRD': 1, 'DRONE': 2, 'HELICOPTER': 3}
FRAME_SKIP_N = config.FRAME_SKIP_N
STATS_LOG_PATH = PROJECT_ROOT / config.YOLO_CONVERSION_LOG_FILE

# В нашем CSV-файле только 4 класса (AIRPLANE, BIRD, DRONE, HELICOPTER)

### def convert_and_draw(sample_frames=[1, 202]):
###     """
###     1. Загружает FINAL_EXPORT.csv
###     2. Конвертирует в формат YOLO (.txt)
###     3. Перемещает PNG-файлы
###     4. Отрисовывает BBoxes для двух тестовых кадров
###     """
###     
###     # 1. Загрузка CSV
###     try:
###         df = pd.read_csv(CSV_PATH)
###     except FileNotFoundError:
###         print(f"FATAL: CSV file not found at {CSV_PATH}. Run MATLAB export first.")
###         return
### 
###     # 1. Создание выходных каталогов (Используем str() для os.makedirs)
###     os.makedirs(str(OUTPUT_IMAGES_DIR), exist_ok=True)
###     os.makedirs(str(OUTPUT_LABELS_DIR), exist_ok=True)
###     os.makedirs(str(OUTPUT_VIZ_DIR), exist_ok=True)
###     
###     total_images = len(df)
###     test_image_paths = {}
### 
###     # Используем tqdm для визуализации прогресса
###     for index, row in tqdm(df.iterrows(), total=total_images, desc="Converting to YOLO"):
###        
###         # Полный путь к исходной картинке (Training_data_V/V_AIRPLANE_001_1_001.png)
###         # Мы должны присоединить путь к папке, где лежали PNG
###         relative_path = row['imageFilename']
###         original_image_full_path = DS_MASTER_DIR / relative_path
### 
###         image_filename = Path(original_image_full_path).name
###         label_filename = image_filename.replace('.png', '.txt')
###         
###         yolo_labels = []
### 
###         # 2. Итерация по классам и BBoxes (8 BBoxes на класс в CSV)
###         for class_name, class_id in CLASS_MAPPING.items():
###             
###             # BBoxes хранятся в колонках ClassName_1, ClassName_2, ClassName_3, ClassName_4
###             # (x_min, y_min, w, h)
###             
###             # Мы берем BBoxes по 4 колонки за раз
###             for i in range(1, 9): # Идем по AIRPLANE_1, AIRPLANE_2, ..., AIRPLANE_8
###                 x_min_col = f'{class_name}_{i}'
###                 y_min_col = f'{class_name}_{i+1}' # Это неправильно, но BBoxes занимают 4 колонки
###                 w_col = f'{class_name}_{i+2}'
###                 h_col = f'{class_name}_{i+3}'
###                 
###                 # ВАЖНО: CSV-таблица, созданная MATLAB, имеет формат:
###                 # AIRPLANE_1 (x), AIRPLANE_2 (y), AIRPLANE_3 (w), AIRPLANE_4 (h), AIRPLANE_5, AIRPLANE_6, ...
###                 
###                 # Обнаруживаем начало BBox (x_min находится в колонке 1, w в колонке 3)
###                 x_min_col_idx = (i * 4) - 3
###                 
###                 try:
###                     # Извлекаем значения. NaN (пустые) будут пропущены.
###                     x_min = row[f'{class_name}_{x_min_col_idx}']
###                     y_min = row[f'{class_name}_{x_min_col_idx+1}']
###                     w = row[f'{class_name}_{x_min_col_idx+2}']
###                     h = row[f'{class_name}_{x_min_col_idx+3}']
###                 except KeyError:
###                     # Если достигли конца колонок для данного класса
###                     continue
### 
###                 # Пропуск пустых ячеек (NaN)
###                 if pd.isna(x_min):
###                     continue
### 
###                 # 3. Конвертация в YOLO формат (Нормализация)
###                 # Определяем, где сейчас находится файл (в исходной или целевой папке)
###                 current_image_path = OUTPUT_IMAGES_DIR / image_filename
###                 
###                 # Если файл еще не перемещен, читаем его из исходной папки
###                 if not current_image_path.exists():
###                     current_image_path = original_image_full_path
###                 
###                 # Получаем размеры изображения (для каждой строки!)
###                 img_dims = cv2.imread(str(current_image_path))
###                 if img_dims is None: continue
###                 H, W, _ = img_dims.shape
### 
###                 # Центр: x_center = x_min + w/2
###                 x_center_norm = (x_min + w / 2) / W
###                 y_center_norm = (y_min + h / 2) / H
###                 width_norm = w / W
###                 height_norm = h / H
###                 
###                 yolo_labels.append(f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}")
###                 
###                 # 4. Сбор тестовых кадров для отрисовки
###                 frame_number = int(image_filename.split('_')[-1].replace('.png', '')) # 'V_AIRPLANE_001_1_001.png' -> 001 -> 1
###                 if frame_number in sample_frames and image_filename not in test_image_paths:
###                     test_image_paths[image_filename] = (str(original_image_full_path), [])
###                 
###                 if image_filename in test_image_paths:
###                     # Сохраняем BBoxes в пикселях для отрисовки (x_min, y_min, x_max, y_max)
###                     x_max = x_min + w
###                     y_max = y_min + h
###                     test_image_paths[image_filename][1].append( (int(x_min), int(y_min), int(x_max), int(y_max), class_name) )
### 
###         # 5. Сохранение метки и перемещение PNG
###         if yolo_labels:
###             output_label_path = OUTPUT_LABELS_DIR / label_filename
###             output_image_path = OUTPUT_IMAGES_DIR / image_filename
### 
###             # Сохранение .txt
###             with open(str(output_label_path), 'w') as f:
###                 f.write('\n'.join(yolo_labels))
###             
###             # Перемещение .png (os.rename)
### 
###             try:
###                 # current_image_path - это то место, откуда мы его прочитали в этом цикле
###                 os.rename(str(current_image_path), str(output_image_path))
###                 ### os.rename(str(original_image_full_path), str(output_image_path))
###             except FileNotFoundError:
###                 pass # Пропустить, если уже переместили (для файлов, которые мы открываем 2 раза)
### 
###     # 6. Отрисовка BBoxes для тестовых кадров (для визуальной проверки)
###     print("\n--- Visual Check: Drawing BBoxes ---")
###     for img_filename, (img_path, bboxes) in test_image_paths.items():
###         if bboxes:
###             img = cv2.imread(img_path)
###             if img is None: 
###                 # Пробуем загрузить из нового места, если переместили
###                 img = cv2.imread(str(OUTPUT_IMAGES_DIR / img_filename))
###                 
###             if img is not None:
###                 for x1, y1, x2, y2, class_name in bboxes:
###                     color = (0, 255, 0) if class_name == 'AIRPLANE' else (0, 0, 255)
###                     cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
###                     cv2.putText(img, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
###                 
###                 output_draw_path = OUTPUT_VIZ_DIR / f'TEST_DRAWN_{img_filename}'
###                 cv2.imwrite(str(output_draw_path), img)
###                 print(f"Drawn BBoxes saved to: {output_draw_path}")
###             
###     print("\nConversion, file move, and test draw complete.")

def convert_and_draw(sample_frames=[1, 202]):
    """
    1. Загружает FINAL_EXPORT.csv
    2. Конвертирует в формат YOLO (.txt) с учетом N.
    3. Копирует PNG-файлы и создает visualize.
    """
    
    # 1. Инициализация и Логирование
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        print(f"FATAL: CSV file not found at {CSV_PATH}. Run MATLAB export first.")
        return

    os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)
    os.makedirs(OUTPUT_LABELS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_VIZ_DIR, exist_ok=True) # <--- Создаем Visualize
    
    total_images = len(df)
    test_image_paths = {}
    
    stats = {'total_source_rows': total_images, 'converted_files': 0, 'skipped_by_N': 0, 'no_bboxes': 0, 'file_error': 0}

    # Используем tqdm для визуализации прогресса
    for index, row in tqdm(df.iterrows(), total=total_images, desc="Converting to YOLO"):
       
        # 0. Логика пропуска N
        if index % FRAME_SKIP_N != 0:
            stats['skipped_by_N'] += 1
            continue
            
        # 1. Определение путей
        relative_path = row['imageFilename']
        
        # Полный путь к исходному файлу
        original_image_full_path = DS_MASTER_DIR / relative_path 
        image_filename = original_image_full_path.name
        label_filename = image_filename.replace('.png', '.txt')
        
        # Пути назначения
        output_label_path = OUTPUT_LABELS_DIR / label_filename
        output_image_path = OUTPUT_IMAGES_DIR / image_filename
        output_viz_path = OUTPUT_VIZ_DIR / f'DRAWN_{image_filename}'

        # Определяем, где лежит файл сейчас (Train_data_V или уже Train/images)
        current_image_path = OUTPUT_IMAGES_DIR / image_filename
        if not current_image_path.exists():
            current_image_path = original_image_full_path

        yolo_labels = []
        bboxes_for_draw = []
        
        # 2. Итерация по классам и BBoxes
        
        # Получаем размеры изображения (для каждой строки!)
        img_dims = cv2.imread(str(current_image_path))
        if img_dims is None: 
            stats['file_error'] += 1
            continue
            
        H, W, _ = img_dims.shape

        for class_name, class_id in CLASS_MAPPING.items():
            for i in range(1, 9): # Идем по AIRPLANE_1, AIRPLANE_5, ... (8 BBoxes на класс)
                x_min_col_idx = (i * 4) - 3
                
                try:
                    x_min = row[f'{class_name}_{x_min_col_idx}']
                    y_min = row[f'{class_name}_{x_min_col_idx+1}']
                    w = row[f'{class_name}_{x_min_col_idx+2}']
                    h = row[f'{class_name}_{x_min_col_idx+3}']
                except KeyError:
                    continue

                if pd.isna(x_min):
                    continue

                # 3. Конвертация в YOLO формат (Нормализация)
                x_center_norm = (x_min + w / 2) / W
                y_center_norm = (y_min + h / 2) / H
                width_norm = w / W
                height_norm = h / H
                
                yolo_labels.append(f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}")

                # Сбор BBoxes для отрисовки (x_min, y_min, x_max, y_max)
                bboxes_for_draw.append( (int(x_min), int(y_min), int(x_min+w), int(y_min+h), class_name) )


        # 5. Сохранение метки, Копирование PNG и Визуализация
        if yolo_labels:
            # 5.1. Сохранение .txt
            with open(str(output_label_path), 'w') as f:
                f.write('\n'.join(yolo_labels))
            
            # 5.2. Копирование .png (НЕ ПЕРЕМЕЩЕНИЕ!)
            # Пропускаем копирование, если файл уже в целевой папке
            if not output_image_path.exists():
                shutil.copy2(str(current_image_path), str(output_image_path))
                
            stats['converted_files'] += 1
            
            # 5.3. Визуализация (Создается для КАЖДОГО сконвертированного файла)
            img_draw = cv2.imread(str(current_image_path))
            if img_draw is not None:
                for x1, y1, x2, y2, class_name in bboxes_for_draw:
                    color = (0, 255, 0) if class_name == 'AIRPLANE' else (0, 0, 255)
                    cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img_draw, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                cv2.imwrite(str(output_viz_path), img_draw)
            
        else:
            stats['no_bboxes'] += 1


    # 6. Логирование статистики
    with open(STATS_LOG_PATH, 'w') as f:
        f.write("--- YOLO Conversion Statistics ---\n")
        f.write(f"Source CSV Rows: {stats['total_source_rows']}\n")
        f.write(f"Frame Skip N: {FRAME_SKIP_N}\n")
        f.write(f"Files Copied and Labeled: {stats['converted_files']}\n")
        f.write(f"Files Skipped by N: {stats['skipped_by_N']}\n")
        f.write(f"Files Skipped (No BBoxes Found): {stats['no_bboxes']}\n")
        f.write(f"Files Skipped (Read Error/Corrupt): {stats['file_error']}\n")
        f.write("--------------------------------\n")
        
    print("\n--- FINAL RESULT ---")
    print(f"Total images converted and copied: {stats['converted_files']}")
    print(f"Stats saved to: {STATS_LOG_PATH}")
    print(f"YOLO dataset structure created at: {OUTPUT_BASE_DIR}")

if __name__ == '__main__':
    convert_and_draw()