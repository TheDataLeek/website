TEXDIR = ./tex/

all:
	make lab2

lab2:
	xelatex --shell-escape --interaction=batchmode $(TEXDIR)lab2.tex
	xelatex --shell-escape --interaction=batchmode $(TEXDIR)lab2.tex
	xelatex --shell-escape --interaction=batchmode $(TEXDIR)lab2.tex

lab2-debug:
	xelatex --shell-escape $(TEXDIR)lab2.tex
	xelatex --shell-escape $(TEXDIR)lab2.tex
	xelatex --shell-escape $(TEXDIR)lab2.tex

clean:
	for FILE in `find . -type f | grep -e "\.pyg\|\.aux\|\.log\|\.toc\|\.out\|\.equ\|\.lof\|\.tar\.gz"`; do rm $$FILE; done

push:
	echo "Not Implemented"
