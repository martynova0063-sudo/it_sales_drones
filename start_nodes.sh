#!/bin/bash

# start_nodes.sh
set -x

# Путь к файлу stream_sources.json
CONFIG_FILE="configs/stream_sources.json"

# Базовый порт для стрима MJPEG (каждая следующая нода будет использовать порт +1)
BASE_STREAM_PORT=8001

# Проверяем, существует ли файл конфига
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Ошибка: Файл конфигурации $CONFIG_FILE не найден."
    exit 1
fi

# Читаем ID всех камер из JSON-файла
# Используем jq для парсинга JSON. Если jq не установлен, установите его: sudo apt install jq
CAMERA_IDS=$(jq -r '.cameras[].id' "$CONFIG_FILE")

# Итерируем по каждому ID и запускаем ноду
CURRENT_PORT=$BASE_STREAM_PORT
for ID in $CAMERA_IDS; do
    echo "Запуск ноды для камеры $ID на порту $CURRENT_PORT..."
    
    # Запуск в новой вкладке gnome-terminal
    # Для других терминалов (например, KDE Konsole, xterm) команда будет отличаться.
    # Для Windows: 'start cmd /k python node.py ...'
    gnome-terminal --tab --title="Node-$ID" -- bash -c "python3 node.py --select $ID --stream_port $CURRENT_PORT; exec bash" &
    
    # Увеличиваем порт для следующей ноды
    CURRENT_PORT=$((CURRENT_PORT + 1))
    
    # Небольшая задержка, чтобы дать каждой ноде время на инициализацию
    sleep 8
done

echo "Все ноды запущены."