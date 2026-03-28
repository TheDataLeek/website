# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Personal portfolio and blog site for Zoë Farmer (ML/AI Engineer) at dataleek.io. Built with Jekyll using a fully custom dark theme (JupyterLab-inspired, no remote theme dependency).

## Development Commands

```bash
# Serve locally with live reload
bundle exec jekyll serve

# Build static site
bundle exec jekyll build

# Convert Jupyter notebooks (.ipynb) to markdown posts
python build_notebooks.py

# Convert PDF resume to PNG images (uses Ghostscript)
python assets/build_assets.py
```

## Architecture

- `_config.yml` — Jekyll config: KaTeX math engine, social links, site metadata (title/subtitle drive the header)
- `index.md` / `1_resume.md` — Main content pages (home and resume)
- `_layouts/` — Layout hierarchy: `default.html` wraps everything; `home.html`, `page.html`, `post.html` extend it
- `_includes/` — `header.html` (sticky nav with pipe separator), `footer.html`, `head.html` (OG/SEO meta)
- `_sass/` — Custom SCSS design system (see below)
- `assets/css/main.scss` — SCSS entry point that imports all partials
- `college/` — Academic notes and projects; `college/code.md`, `college/data.md`, `college/math.md` are section index pages
- `presentations/` — Reveal.js presentations
- `projects/` — Project showcase pages

## SCSS Design System

All design tokens live in `_sass/_variables.scss`:
- **Color palette**: Matplotlib tab10 colors (`$tab-blue`, `$tab-orange`, etc.) as semantic accents (`$accent-primary`, `$accent-code`, `$accent-math`, etc.)
- **Surfaces**: Dark JupyterLab palette (`$surface-page: #1a1a1a`, `$surface-cell`, `$surface-code`)
- **Spacing**: `$space-1` through `$space-16` (rem-based scale, note: `$space-5` does not exist — use `$space-4` or `$space-6`)
- **Typography**: Inter (sans) + JetBrains Mono (mono); `$text-xs` through `$text-4xl`

Key SCSS partials:
- `_notebook.scss` — `.nb-notebook` wrapper gives content the notebook-cell aesthetic; applied automatically in layouts
- `_nav.scss` — Sticky header with pipe separator (`.site-nav__sep`) hidden in mobile drawer
- `_home.scss` — Hero section and image tile grid for the homepage

## Content Conventions

- Pages use Jekyll front matter with `layout`, `title`, `description`, and `nav-menu` fields
- Math rendering via KaTeX (configured in `_config.yml` under `kramdown`)
- Conventional commits: `feat:`, `fix:`, `docs:`, `refactor:`
- The site owner's name is **Zoë** (with diaeresis) — use the correct spelling
