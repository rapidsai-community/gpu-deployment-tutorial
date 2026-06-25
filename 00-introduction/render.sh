#!/usr/bin/env bash
# Render the intro deck to PDF + PPTX using the bundled NVIDIA theme.
# Requires Node (npx). Run from anywhere; it cd's into its own folder.
set -euo pipefail
cd "$(dirname "$0")"

DECK="00-introduction-and-gpu-stack.md"
MARP="npx --yes @marp-team/marp-cli@latest"

# Theme, HTML, and local-file access come from .marprc.yml in this folder.
$MARP "$DECK" --pptx -o 00-introduction-and-gpu-stack.pptx

echo "Rendered: 00-introduction-and-gpu-stack.pdf and .pptx"
