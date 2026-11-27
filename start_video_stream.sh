#!/bin/bash

# start_video_stream.sh

# --- Настройки ---
OBS_PATH="/usr/bin/obs" # Или "/snap/bin/obs-studio", если вдруг используете Snap, но у вас apt-версия
MEDIAMTX_DIR="$HOME/mediamtx"
MEDIAMTX_BIN="$MEDIAMTX_DIR/mediamtx"
MEDIAMTX_LOG="$MEDIAMTX_DIR/mediamtx.log"

# --- Запуск OBS Studio ---
echo "Запуск OBS Studio..."
# OBS_PATH & отправляет его в фон.
# Может потребоваться небольшая задержка, чтобы OBS успел инициализироваться и запустить RTSP-сервер.
$OBS_PATH &
sleep 10 # Даем OBS 10 секунд на запуск и инициализацию RTSP сервера

# --- Запуск MediaMTX ---
echo "Запуск MediaMTX..."
# Проверяем, не запущен ли MediaMTX уже
if pgrep -x "mediamtx" > /dev/null
then
    echo "MediaMTX уже запущен."
else
    cd "$MEDIAMTX_DIR"
    nohup "$MEDIAMTX_BIN" > "$MEDIAMTX_LOG" 2>&1 &
    echo "MediaMTX запущен в фоновом режиме, логи в $MEDIAMTX_LOG"
fi

echo "Все запущено. Через несколько секунд OBS RTSP -> MediaMTX -> Node.js должны работать."
echo "Не забудьте проверить логи MediaMTX и вашей Node.js-системы."