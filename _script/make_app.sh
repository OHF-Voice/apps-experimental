#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 <slug> <name> <port>" >&2
  exit 1
}

escape_sed_replacement() {
  printf '%s' "$1" | sed 's/[&|\\]/\\&/g'
}

[[ $# -eq 3 ]] || usage

SLUG=$1
NAME=$2
PORT=$3

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE_DIR="$SCRIPT_DIR/app_template"
DEST_DIR="$PWD/$SLUG"

[[ -d "$TEMPLATE_DIR" ]] || {
  echo "Template directory not found: $TEMPLATE_DIR" >&2
  exit 1
}

[[ ! -e "$DEST_DIR" ]] || {
  echo "Destination already exists: $DEST_DIR" >&2
  exit 1
}

mkdir -p -- "$DEST_DIR"

# Copy contents of app_template into the new slug directory.
cp -a -- "$TEMPLATE_DIR"/. "$DEST_DIR"/

# Rename files and directories whose names contain @@SLUG@@.
find "$DEST_DIR" -depth -name '*@@SLUG@@*' -exec bash -c '
  set -euo pipefail
  slug=$1
  shift
  for path in "$@"; do
    new_path=${path//@@SLUG@@/$slug}
    if [[ "$path" != "$new_path" ]]; then
      mv -- "$path" "$new_path"
    fi
  done
' bash "$SLUG" {} +

# Escape replacement values for sed.
SLUG_ESCAPED=$(escape_sed_replacement "$SLUG")
NAME_ESCAPED=$(escape_sed_replacement "$NAME")
PORT_ESCAPED=$(escape_sed_replacement "$PORT")

# Replace placeholders in file contents.
find "$DEST_DIR" -type f -exec sed -i \
  -e "s|@@SLUG@@|$SLUG_ESCAPED|g" \
  -e "s|@@NAME@@|$NAME_ESCAPED|g" \
  -e "s|@@PORT@@|$PORT_ESCAPED|g" \
  {} +

echo "Created app at: $DEST_DIR"
