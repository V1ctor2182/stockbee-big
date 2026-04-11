#!/bin/bash
# Convert timeline.yaml → JSON and embed into timeline.html so the page
# works with file:// protocol without any external JS libraries.
# Run this after updating timeline.yaml.
# Usage: bash meta/00-project-room/sync-timeline.sh

set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
HTML="$DIR/timeline.html"
YAML="$DIR/timeline.yaml"

if [ ! -f "$YAML" ]; then
  echo "Error: $YAML not found"
  exit 1
fi

python3 - <<PY
import json
import re
import sys

try:
    import yaml
except ImportError:
    sys.stderr.write(
        "PyYAML not available — install with: pip install pyyaml\n"
    )
    sys.exit(1)

html_path = "$HTML"
yaml_path = "$YAML"

with open(yaml_path, "r") as f:
    data = yaml.safe_load(f)

json_blob = json.dumps(data, ensure_ascii=False, indent=2)

with open(html_path, "r") as f:
    html = f.read()

pattern = re.compile(
    r'(<script id="timeline-data" type="application/json">)\n.*?(</script>\n\n</body>)',
    re.DOTALL,
)
replacement = r"\1\n" + json_blob + "\n" + r"\2"
html_new, n = pattern.subn(replacement, html)

if n == 0:
    # First-time migration: replace the older text/yaml placeholder too.
    legacy = re.compile(
        r'<script id="timeline-data" type="text/yaml">\n.*?</script>',
        re.DOTALL,
    )
    new_block = (
        '<script id="timeline-data" type="application/json">\n'
        + json_blob
        + "\n</script>"
    )
    html_new, n = legacy.subn(new_block, html)

if n == 0:
    sys.stderr.write("Error: could not find embedded timeline-data script tag\n")
    sys.exit(2)

with open(html_path, "w") as f:
    f.write(html_new)

print(f"Synced timeline.yaml → JSON ({len(json_blob)} bytes) into timeline.html")
PY
