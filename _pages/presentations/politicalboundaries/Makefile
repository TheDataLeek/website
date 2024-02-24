
slides:
	jupyter nbconvert --to slides politicalboundaries.ipynb --reveal-prefix './reveal.js-3.5.0'
	sudo cp -rv ../political_boundaries /home/present/

push:
	-rm archive.tgz
	tar -czvf archive.tgz *
	jupyter nbconvert --to slides politicalboundaries.ipynb --reveal-prefix '/presentations/reveal.js-3.5.0'
	rsync -uaz --progress * zoe@dataleek.io:/var/www/html/dataleek.io/presentations/politicalboundaries/
