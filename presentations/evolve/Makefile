FILES = $(shell find . -maxdepth 1 -type f)

slides:
	jupyter nbconvert --to slides ./evolve_interview.ipynb --reveal-prefix ./reveal.js

external:
	jupyter nbconvert --to slides ./evolve_interview.ipynb --reveal-prefix "https://dataleek.io/presentations/reveal.js-3.5.0/"
