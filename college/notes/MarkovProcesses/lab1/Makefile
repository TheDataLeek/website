TEXDIR = ./tex/

all:
	make lab1

lab1:
	xelatex --shell-escape --interaction=batchmode $(TEXDIR)lab1.tex
	xelatex --shell-escape --interaction=batchmode $(TEXDIR)lab1.tex
	xelatex --shell-escape --interaction=batchmode $(TEXDIR)lab1.tex

lab1-debug:
	xelatex --shell-escape $(TEXDIR)lab1.tex
	xelatex --shell-escape $(TEXDIR)lab1.tex
	xelatex --shell-escape $(TEXDIR)lab1.tex

clean:
	for FILE in `find . -type f | grep -e "\.pyg\|\.aux\|\.log\|\.toc\|\.out\|\.equ\|\.lof\|\.tar\.gz"`; do rm $$FILE; done

push:
	echo "Not Implemented"
