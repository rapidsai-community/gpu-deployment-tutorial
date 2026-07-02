#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Render the intro deck to PDF + PPTX using the bundled NVIDIA theme.
# Requires Node (npx). Run from anywhere; it cd's into its own folder.
set -euo pipefail
cd "$(dirname "$0")"

DECK="introduction-to-gpu-stack.md"
MARP="npx --yes @marp-team/marp-cli@latest"

# Theme, HTML, and local-file access come from .marprc.yml in this folder.
$MARP "$DECK" --pdf  -o introduction-to-gpu-stack.pdf
$MARP "$DECK" --pptx -o introduction-to-gpu-stack.pptx

echo "Rendered: introduction-to-gpu-stack.pdf and .pptx"
