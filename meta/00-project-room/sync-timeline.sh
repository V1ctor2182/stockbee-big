#!/bin/bash
# Embed timeline.yaml into timeline.html so it works with file:// protocol.
# Run this after updating timeline.yaml.
# Usage: bash meta/00-project-room/sync-timeline.sh

DIR="$(cd "$(dirname "$0")" && pwd)"
HTML="$DIR/timeline.html"
YAML="$DIR/timeline.yaml"

if [ ! -f "$YAML" ]; then
  echo "Error: $YAML not found"
  exit 1
fi

# Replace content between <script id="timeline-data"> and </script>
python3 -c "
import re

with open('$HTML', 'r') as f:
    html = f.read()

with open('$YAML', 'r') as f:
    yaml_content = f.read()

# Replace the embedded YAML block
pattern = r'(<script id=\"timeline-data\" type=\"text/yaml\">)\n.*?(</script>\n\n</body>)'
replacement = r'\1\n' + yaml_content + r'\2'
html_new = re.sub(pattern, replacement, html, flags=re.DOTALL)

with open('$HTML', 'w') as f:
    f.write(html_new)

print(f'Synced timeline.yaml ({len(yaml_content)} bytes) into timeline.html')
"
