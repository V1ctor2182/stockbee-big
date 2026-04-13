#!/bin/bash
# Embed timeline.yaml as a JS variable into timeline.html.
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

python3 -c "
import json, re, sys

try:
    import yaml
except ImportError:
    sys.stderr.write('PyYAML not available — install with: pip install pyyaml\n')
    sys.exit(1)

html_path = '$HTML'
yaml_path = '$YAML'

with open(yaml_path, 'r') as f:
    data = yaml.safe_load(f)

json_blob = json.dumps(data, ensure_ascii=False, indent=2)

with open(html_path, 'r') as f:
    html = f.read()

pattern = re.compile(
    r'(var TIMELINE_RAW_DATA =)\n.*?;\ninit\(\);',
    re.DOTALL,
)
replacement = r'\1\n' + json_blob + ';\ninit();'
html_new, n = pattern.subn(replacement, html)

if n == 0:
    sys.stderr.write('Error: could not find TIMELINE_RAW_DATA block in HTML\n')
    sys.exit(2)

with open(html_path, 'w') as f:
    f.write(html_new)

print(f'Synced timeline.yaml → JS ({len(json_blob)} bytes) into timeline.html')
"
