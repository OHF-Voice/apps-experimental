#!/usr/bin/env bash
set -euo pipefail

ADDON_DIR="${1:-$(pwd)}"
CONFIG_YAML="${ADDON_DIR}/config.yaml"
DOCKERFILE="${ADDON_DIR}/Dockerfile.local"

if [[ ! -f "${CONFIG_YAML}" ]]; then
  echo "Missing ${CONFIG_YAML}" >&2
  exit 1
fi

if [[ ! -f "${DOCKERFILE}" ]]; then
  echo "Missing ${DOCKERFILE}" >&2
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

echo "Building add-on:"
echo "  directory: ${ADDON_DIR}"
echo "  arch:      ${ha_arch}"
echo "  image tag: ${image_tag}"

docker build \
    -f "${DOCKERFILE}" \
    -t "${image_tag}" \
    "${ADDON_DIR}"

echo
echo "Built image ${image_tag}"
