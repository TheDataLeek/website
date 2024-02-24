FILES = $(shell find . -maxdepth 1 -type f)
all:
	jupyter nbconvert --to slides Speed\ Snakes.ipynb --reveal-prefix ./reveal.js-3.6.0

push:
	jupyter nbconvert --to slides Speed\ Snakes.ipynb --reveal-prefix "https://dataleek.io/presentations/reveal.js-3.5.0/"
	rsync -avz --progress $(FILES) zoe@dataleek.io:/var/www/html/dataleek.io/presentations/speedsnakes/
