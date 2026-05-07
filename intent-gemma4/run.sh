#!/usr/bin/env sh

cd "$(dirname "$0")"

    # --device-id '0419d58f1f161dfe9e327a2ed9c9f47e' \
    # --hass-api 'http://localhost:8123/api' \
    # --hass-token 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiI4ZThlZWE1NDQ4ZDY0NGJjYjIzZDJlZmVkNjZmZDAyMyIsImlhdCI6MTY5NTMyMjk5MywiZXhwIjoyMDEwNjgyOTkzfQ.t9C8P1HT4xQleyXv8-SQbM_hkZMiIt8HTx0MA6wzIvY' \

python3 src/app.py \
    --uri 'tcp://0.0.0.0:10500' \
    --hass-api 'http://homeassistant.local:8123/api' \
    --hass-token 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiI1N2U3YjlkY2U5MTY0YjhjOGQ5OGQ2NjVjZWYzYjVmYyIsImlhdCI6MTcyNzU2NTY5MCwiZXhwIjoyMDQyOTI1NjkwfQ.vGExzcuHUlvZ66ufZDkWxKictuXVfwaxHVHMb4tvTHY' \
    --tools src/tools.yaml \
    --prime-model \
    --debug "$@"
