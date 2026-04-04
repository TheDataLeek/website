# List available commands
default:
    @just --list

# Install Node.js dependencies
npm-install:
    npm ci

# Build and minify JS source files
js: npm-install
    npm run build

# Watch JS source files and rebuild on change (run in a second terminal during dev)
js-watch: npm-install
    npm run watch &

# Install dependencies and serve with live reload
serve: install js
    bundle exec jekyll serve

# Build the static site
build: install js
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
