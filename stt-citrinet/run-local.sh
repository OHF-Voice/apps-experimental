#!/usr/bin/env bash
cd /usr/src

options=/data/options.json

if [[ ! -f "${options}" ]]; then
    echo "Missing ${options}"
    exit 1
fi

# Get an option value with optional default
config() {
    local key="$1"
    local default="${2:-}"

    if [[ -f "${options}" ]]; then
        jq -r --arg key "${key}" \
            --arg default "${default}" \
            '.[$key] // $default' \
            "${options}"
    else
        printf '%s\n' "${default}"
    fi
}

# Check if option is true
config_true() {
    local key="$1"
    local value
    value="$(config "${key}" "false")"

    [[ "${value}" == "true" ]]
}

# -----------------------------------------------------------------------------

flags=()

if config_true debug_logging; then
    flags+=('--debug')
fi

exec python3 app.py \
    --uri 'tcp://0.0.0.0:10300' \
    --model "$(config model)" \
    --http-host '0.0.0.0' \
    --http-port 5000 \
    --sentences /data/sentences.txt \
    --train-dir /data/train \
    --cache-dir /data/cache \
    ${flags[@]}
