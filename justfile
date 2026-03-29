# List available commands
default:
    @just --list

# Install dependencies and serve with live reload
serve: install
    bundle exec jekyll serve

# Build the static site
build: install
    bundle exec jekyll build

# Install Ruby gem dependencies
install:
    bundle install

# Convert Jupyter notebooks to markdown posts
notebooks:
    uv run python build_notebooks.py

# Convert PDF resume to PNG images (requires Ghostscript)
assets *args='':
    uv run python assets/build_assets.py {{ args }}
