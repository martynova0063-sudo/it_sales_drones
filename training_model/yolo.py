# training_model/yolo.py

import argparse
import os
import sys
import logging
from datetime import datetime

# Добавляем директорию src в sys.path для импорта модулей
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config import Config
from src.logger import Logger
from src.dataset_manager import DatasetManager
from src.yolo_model_trainer import YOLOModelTrainer
from src.prediction_analyzer import PredictionAnalyzer

def main():
    """
    Основная функция-оркестратор для управления жизненным циклом YOLO-проекта:
    загрузка данных, обучение/дообучение модели и тестирование.

    Принимает аргументы командной строки для определения выполняемой операции.
    """
    ##########################################################################
    # Обработка ключей запуска
    ##########################################################################
    parser = argparse.ArgumentParser(description="YOLO Drone Detection Project Orchestrator")
    parser.add_argument(
        "--data",
        action="store_true",
        help="Загружает, организует датасет, генерирует статистику и подготавливает data.yaml."
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Обучает или дообучает модель YOLO, сохраняет лучшую модель и графики обучения."
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Выполняет инференс обученной модели на тестовом датасете."
    )
    parser.add_argument(
        "--predict",
        action="store_true",
        help="Выполняет инференс на тестовом датасете, сравнивает предсказания с метками, собирает статистику и сортирует результаты по папкам."
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Генерирует подробную статистику датасета (количество объектов по классам и размерам) для train, val, test и всего датасета."
    )

    args = parser.parse_args()

    ##########################################################################
    # Инициализация конфигурации и логгера
    ##########################################################################
    config = Config()
    
    # Создаем директорию runs/project_name/timestamp для логов и результатов
    os.makedirs(config.output_dir, exist_ok=True)
    logger = Logger("YOLO_Orchestrator", os.path.join(config.output_dir, "orchestrator_log.log"))
###     logger.set_level(logging.DEBUG) # Устанавливаем уровень логирования в DEBUG для детальной отладки

    logger.info("Запуск проекта по обнаружению дронов с помощью YOLO26.")
    logger.info(f"Директория вывода: {config.output_dir}")

    # Сохраняем текущую конфигурацию для воспроизводимости
    config.save(os.path.join(config.output_dir, "config.yaml"))
    logger.info(f"Конфигурация сохранена в {os.path.join(config.output_dir, 'config.yaml')}")

    dataset_manager = DatasetManager(config, logger)
    model_trainer = YOLOModelTrainer(config, logger)

    ##########################################################################
    # Обработка аргументов командной строки
    ##########################################################################
    # --data: Загрузка и организация датасета
    if args.data:
        logger.info("Выбран режим '--data': Загрузка и организация датасета.")
        # Проверяем, существует ли data.yaml и объединенный датасет, чтобы пропустить загрузку/разбиение
        # для более быстрых повторных запусков, если данные уже готовы.
        if not os.path.exists(config.data_yaml_path) or not os.path.exists(dataset_manager.data_root):
            logger.info("Датасет или data.yaml не найдены, приступаем к загрузке и организации.")
            try:
                dataset_manager.download_and_extract_datasets()
                dataset_manager.organize_and_split_data() # Разделение по умолчанию 80/10/10
            except Exception as e:
                logger.error(f"Ошибка при подготовке датасета: {e}", exc_info=True)
                sys.exit(1)
        else:
            logger.info(f"Найдены существующие data.yaml по пути {config.data_yaml_path} и датасет по пути {dataset_manager.data_root}. Пропуск загрузки и организации.")
        
        dataset_manager.get_dataset_statistics()
        logger.info("Режим '--data' завершен.")

    ##########################################################################
    # --train: Обучение/Дообучение модели
    if args.train:
        logger.info("Выбран режим '--train': Обучение/дообучение модели.")
        if not os.path.exists(config.data_yaml_path):
            logger.error(f"Файл data.yaml не найден по пути {config.data_yaml_path}. Пожалуйста, запустите '--data' сначала.")
            sys.exit(1)
        
        try:
            model_trainer.train()
        except Exception as e:
            logger.error(f"Ошибка при обучении модели: {e}", exc_info=True)
            sys.exit(1)
        logger.info("Режим '--train' завершен.")

    ##########################################################################
    # --test: Инференс на тестовой выборке
    if args.test:
        logger.info("Выбран режим '--test': Инференс на тестовой выборке.")
        try:
            if not os.path.exists(config.data_yaml_path):
                logger.error(f"Файл data.yaml не найден по пути {config.data_yaml_path}. Пожалуйста, запустите '--data' сначала.")
                sys.exit(1)
            if not os.path.exists(config.best_model_path) and not os.path.exists(config.base_model_name): # Проверяем, есть ли хотя бы базовая модель или лучшая
                logger.warning(f"Обученная модель не найдена по пути {config.best_model_path}. Будет использоваться базовая модель {config.base_model_name} для инференса. Рекомендуется сначала обучить модель с помощью '--train'.")
            ### pass
            model_trainer.validate_on_test_set()
        except Exception as e:
            logger.error(f"Ошибка при инференсе модели: {e}", exc_info=True)
            sys.exit(1)
        logger.info("Режим '--test' завершен.")

    ##########################################################################
    # --predict: Предикт на тестовой выборке (картинки и статистика)
    if args.predict:
        logger.info("Выбран режим '--predict': Инференс с анализом результатов.")
        # Проверяем, существует ли data.yaml
        if not os.path.exists(config.data_yaml_path):
            logger.error(f"Файл data.yaml не найден по пути {config.data_yaml_path}. Пожалуйста, запустите '--data' сначала.")
            sys.exit(1)
        # Проверяем, существует ли модель
        if not os.path.exists(config.best_model_path) and not os.path.exists(config.base_model_name):
            logger.error(f"Модель не найдена по пути {config.best_model_path} или {config.base_model_name}. Пожалуйста, обучите модель или проверьте путь.")
            sys.exit(1)
        
        try:
            # Создаем экземпляр анализатора предсказаний
            prediction_analyzer = PredictionAnalyzer(config, logger)
            # Запускаем анализ
            prediction_analyzer.analyze_predictions()
        except Exception as e:
            logger.error(f"Ошибка при анализе предсказаний: {e}", exc_info=True)
            sys.exit(1)
        logger.info("Режим '--predict' завершен.")

    ##########################################################################
    # --stats: Генерация статистики датасета
    if args.stats:
        logger.info("Выбран режим '--stats': Генерация подробной статистики датасета.")
        if not os.path.exists(config.data_yaml_path):
            logger.error(f"Файл data.yaml не найден по пути {config.data_yaml_path}. Пожалуйста, запустите '--data' сначала.")
            sys.exit(1)
        
        try:
            from src.dataset_statistics_generator import DatasetStatisticsGenerator # Импорт здесь для ленивой загрузки
            stats_generator = DatasetStatisticsGenerator(config, logger)
            stats_generator.generate_statistics() # Метод будет генерировать для всех сплитов
        except Exception as e:
            logger.error(f"Ошибка при генерации статистики датасета: {e}", exc_info=True)
            sys.exit(1)
        logger.info("Режим '--stats' завершен.")

    ##########################################################################
    # Если не найден ни один из ключей
    if not any([args.data, args.train, args.test, args.predict, args.stats]):
        logger.warning("Не указан ни один режим (например, --data, --train, --test, --predict, --stats). Никаких действий не выполнено.")
        parser.print_help()

    logger.info("Работа оркестратора завершена.")

if __name__ == "__main__":
    main()