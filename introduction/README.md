# 00 · Introduction & the GPU stack

Opening slides: a conceptual tour from the GPU hardware up to RAPIDS.

Built with [Marp](https://marp.app/). `introduction-to-gpu-stack.md` is the
source of truth; the `.pdf` and `.pptx` are rendered outputs.

## Edit

Open the `.md` in VS Code with the **Marp for VS Code** extension for a live,
styled preview. Point it at the theme via your settings:

```json
{
  "markdown.marp.themes": ["./introduction/themes/nvidia.css"],
  "markdown.marp.enableHtml": true
}
```

## Theme

The slides use a specific Marp theme (`themes/nvidia.css`). It is
**bundled in this folder**, so there is no external link to fetch, the settings above and `.marprc.yml` both point at the local copy. The deck-specific helper styles (for the infographic slides) live at the bottom of that file, so the slide Markdown itself stays clean.

## Render

```bash
./render.sh        # regenerates the PDF + PPTX
```

CUDA figures are from the NVIDIA CUDA C++ Programming Guide, attributed on-slide.
