---
name: marimo-to-post
description: Convert a marimo notebook (.py) from ~/projects/nn-playground into a Jekyll post at ~/projects/website/projects/neural-networks/. Provide the notebook filename or full path.
allowed-tools: Read Write Edit Bash
---

Convert a marimo notebook to a Jekyll blog post for dataleek.io. The user provides a filename
or path to a `.py` file in `~/projects/nn-playground`.

## Setup

Derive from user input:
- `NOTEBOOK` — absolute path to the `.py` (e.g. `/home/zoe/projects/nn-playground/mlp.py`)
- `STEM` — filename without extension (e.g. `mlp`)
- `PLAYGROUND` — `/home/zoe/projects/nn-playground`
- `DEST` — `/home/zoe/projects/website/projects/neural-networks`
- `CONVERT` — `/home/zoe/projects/website/.claude/skills/marimo-to-post/convert.py`

---

## Step 1 — Export from marimo

```bash
cd /home/zoe/projects/nn-playground && \
  uv run marimo export md $NOTEBOOK -o /tmp/${STEM}_raw.md
```

---

## Step 2 — Mechanical transforms

```bash
python3 $CONVERT /tmp/${STEM}_raw.md /tmp/${STEM}_body.md
```

This script handles (in order):
- Strip marimo frontmatter (`---…---` header block)
- `{.marimo}` / `{.marimo hide_code="true"}` fences → plain `python` fences
- `mo.mermaid("""…""")` code blocks → ` ```mermaid ` blocks
- Blocks containing `mo.image()`: append `![png](filename.png)` after the closing fence;
  add `{: .code-collapsed}` before it when the block is ≥ 30 lines
- Empty `{.marimo}` blocks (trailing marimo artifacts) → removed
- Single-dollar inline math `$…$` → `$$…$$` (KaTeX requires double-dollar)

---

## Step 3 — Frontmatter

Read `/tmp/${STEM}_body.md`.

1. Extract the first `# Heading` — use it as `title`.
2. Read the first prose paragraph after the heading — write a 1–2 sentence `description`
   that captures what the notebook demonstrates and why it's interesting.
3. Use today's date as `date` (YYYY-MM-DD).

Prepend this Jekyll frontmatter, leaving the body unchanged:

```yaml
---
layout: post
nav-menu: false
show_tile: false
title: "<title>"
description: "<description>"
date: "<YYYY-MM-DD>"
katex: true
---

```

---

## Step 4 — Copy images

Grep `/tmp/${STEM}_body.md` for all `![png](filename.png)` references. Copy each referenced
PNG from `$PLAYGROUND/` to `$DEST/`:

```bash
grep -oP '!\[png\]\(\K[^)]+' /tmp/${STEM}_body.md | \
  xargs -I{} cp $PLAYGROUND/{} $DEST/
```

Warn if any referenced PNG is missing from `$PLAYGROUND/` (it may need the notebook to be
run first).

---

## Step 5 — Install post

```bash
mv /tmp/${STEM}_body.md $DEST/${STEM}.md
```

---

## Step 6 — Report

- Path of the post written
- Images copied (list them)
- Any `mo.image()` block where the PNG filename couldn't be inferred from a
  `notebook_dir()` call (these blocks still render as code but lack an image reference —
  add one manually)
- Any missing PNGs (notebook needs to be run to generate them)
