#!/usr/bin/env bash
set -euo pipefail

ADDON_DIR=""
OPTION_OVERRIDES=()

for arg in "$@"; do
  if [[ "$arg" == *=* ]]; then
    OPTION_OVERRIDES+=("$arg")
  elif [[ -z "${ADDON_DIR}" ]]; then
    ADDON_DIR="$arg"
  else
    echo "Unexpected argument: $arg" >&2
    echo "Usage: $0 [addon_dir] [key=value ...]" >&2
    exit 1
  fi
done

if [[ -z "${ADDON_DIR}" ]]; then
  ADDON_DIR="$(pwd)"
fi

CONFIG_YAML="${ADDON_DIR}/config.yaml"

if [[ ! -f "${CONFIG_YAML}" ]]; then
  echo "Missing ${CONFIG_YAML}" >&2
  exit 1
fi

host_arch="$(uname -m)"
case "${host_arch}" in
  x86_64)
    ha_arch="amd64"
    ;;
  aarch64|arm64)
    ha_arch="aarch64"
    ;;
  *)
    echo "Unsupported host architecture: ${host_arch}" >&2
    exit 1
    ;;
esac

slug="$(
  awk -F': *' '$1 == "slug" { print $2; exit }' "${CONFIG_YAML}"
)"

version="$(
  awk -F': *' '$1 == "version" { print $2; exit }' "${CONFIG_YAML}"
)"

if [[ -z "${slug}" || -z "${version}" ]]; then
  echo "Could not read slug/version from ${CONFIG_YAML}" >&2
  exit 1
fi

image_tag="${slug}:${version}-${ha_arch}"
container_name="${slug}-local"
data_dir="${ADDON_DIR}/.addon_data"
options_json="${data_dir}/options.json"

mkdir -p "${data_dir}"

mapfile -t GENERATED < <(
  python3 - "${CONFIG_YAML}" "${options_json}" "${OPTION_OVERRIDES[@]}" <<'PY'
import json
import sys

config_path = sys.argv[1]
output_path = sys.argv[2]
override_args = sys.argv[3:]

with open(config_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

def strip_quotes(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
        return s[1:-1]
    return s

def parse_scalar(value: str):
    value = value.strip()

    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1]

    low = value.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if low in ("null", "~"):
        return None

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        pass

    return value

options = {}
ports = []
section = None
section_indent = None

ingress_port = None

for raw_line in lines:
    line = raw_line.rstrip("\n")
    stripped = line.strip()

    if not stripped or stripped.startswith("#"):
        continue

    indent = len(line) - len(line.lstrip(" "))

    if section is None:
        if stripped == "options:":
            section = "options"
            section_indent = indent
            continue

        if stripped == "ports:":
            section = "ports"
            section_indent = indent
            continue

        if stripped.startswith("ingress_port:"):
            _, value = stripped.split(":", 1)
            ingress_port = parse_scalar(value)
            continue

        continue

    if indent <= section_indent:
        section = None
        section_indent = None
        continue

    if ":" not in stripped:
        continue

    key, value = stripped.split(":", 1)
    key = strip_quotes(key)
    value = value.strip()

    if section == "options":
        options[key] = parse_scalar(value)

    elif section == "ports":
        ports.append((key, parse_scalar(value)))


# Emit normal ports

for container_port, host_binding in ports:
    container_port = strip_quotes(container_port).strip()

    if "/" in container_port:
        container_num, proto = container_port.split("/", 1)
        container_spec = f"{container_num}/{proto}"
    else:
        container_num = container_port
        container_spec = container_port

    if host_binding is None or host_binding == "":
        print(f"PORTMAP={container_num}:{container_spec}")

    elif isinstance(host_binding, int):
        print(f"PORTMAP={host_binding}:{container_spec}")

    else:
        host_binding = str(host_binding).strip()
        print(f"PORTMAP={host_binding}:{container_spec}")


# Emit ingress port

if ingress_port is not None:
    print(f"PORTMAP={ingress_port}:{ingress_port}/tcp")

print(f"OPTIONS_JSON={output_path}")
PY
)

PORT_ARGS=()
for line in "${GENERATED[@]}"; do
  case "${line}" in
    PORTMAP=*)
      mapping="${line#PORTMAP=}"
      PORT_ARGS+=(-p "${mapping}")
      ;;
  esac
done

if ! docker image inspect "${image_tag}" >/dev/null 2>&1; then
  echo "Docker image ${image_tag} does not exist. Build it first." >&2
  exit 1
fi

docker rm -f "${container_name}" >/dev/null 2>&1 || true

echo "Using image: ${image_tag}"
echo "Writing options file to: ${options_json}"
if [[ ${#OPTION_OVERRIDES[@]} -gt 0 ]]; then
  echo "Overrides:"
  printf '  %s\n' "${OPTION_OVERRIDES[@]}"
fi
if [[ ${#PORT_ARGS[@]} -gt 0 ]]; then
  echo "Port mappings:"
  for ((i=0; i<${#PORT_ARGS[@]}; i+=2)); do
    echo "  ${PORT_ARGS[i]} ${PORT_ARGS[i+1]}"
  done
fi
echo

docker run --rm -it \
  --name "${container_name}" \
  "${PORT_ARGS[@]}" \
  -v "${data_dir}:/data" \
  "${image_tag}"
