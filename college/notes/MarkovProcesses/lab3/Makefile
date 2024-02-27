TEXDIR = ./tex/

all:
	make lab

lab:
	xelatex --shell-escape --interaction=batchmode $(TEXDIR)lab3.tex
	xelatex --shell-escape --interaction=batchmode $(TEXDIR)lab3.tex
	xelatex --shell-escape --interaction=batchmode $(TEXDIR)lab3.tex

debug:
	xelatex --shell-escape $(TEXDIR)lab3.tex

clean:
	for FILE in `find . -type f | grep -e "\.pyg\|\.aux\|\.log\|\.toc\|\.out\|\.equ\|\.lof\|\.tar\.gz"`; do rm $$FILE; done

push:
	echo "Not Implemented"
