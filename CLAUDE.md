# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Personal portfolio and blog site for Zoë Farmer (ML/AI Engineer) at dataleek.io. Built with Jekyll using the `andrewbanchich/forty-jekyll-theme` remote theme.

## Development Commands

```bash
# Serve locally with live reload
jekyll serve

# Build static site
jekyll build

# Convert Jupyter notebooks (.ipynb) to markdown posts
python build_notebooks.py

# Convert PDF resume to PNG images (uses Ghostscript)
python assets/build_assets.py
```

## Architecture

- `_config.yml` — Jekyll config: remote theme, KaTeX math engine, social links, site metadata
- `index.md` / `1_resume.md` — Main content pages (home and resume)
- `_layouts/`, `_includes/`, `_sass/` — Theme overrides (header, footer, head, page templates)
- `assets/` — Static files: CSS, JS, images, PDFs, and asset build scripts
- `college/` — Academic notes and projects (some with their own Makefiles and `build.py`)
- `presentations/` — Reveal.js presentations
- `projects/` — Project showcase pages

## Content Conventions

- Pages use Jekyll front matter with `layout`, `title`, `description`, and `nav-menu` fields
- Math rendering via KaTeX (configured in `_config.yml` under `kramdown`)
- Conventional commits are used: `feat:`, `fix:`, `docs:`, `refactor:`
- The site owner's name is **Zoë** (with diaeresis) — use the correct spelling
