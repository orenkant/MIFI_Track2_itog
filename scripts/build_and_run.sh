#!/usr/bin/env bash
# Сборка Docker-образа с API и запуск контейнера локально
set -euo pipefail

IMAGE="credit-default-api"
CONTAINER="credit-default-api"
PORT="${PORT:-8000}"

cd "$(dirname "$0")/.."

if [[ ! -f models/model.pkl ]]; then
    echo "Нет models/model.pkl: выполните 'dvc pull' или 'dvc repro'"
    exit 1
fi

docker build -t "$IMAGE" .

docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
docker run -d --name "$CONTAINER" -p "$PORT:8000" "$IMAGE" >/dev/null

echo "Жду запуска API..."
for _ in $(seq 1 30); do
    if curl -sf "http://localhost:$PORT/health" >/dev/null; then
        echo "API запущен: http://localhost:$PORT/docs"
        echo "Проверочный запрос к /predict:"
        curl -s -X POST "http://localhost:$PORT/predict" \
            -H "Content-Type: application/json" \
            -d @scripts/sample_request.json
        echo
        echo "Остановить: docker rm -f $CONTAINER"
        exit 0
    fi
    sleep 1
done

echo "API не ответил за 30 секунд, логи контейнера:"
docker logs "$CONTAINER"
exit 1
