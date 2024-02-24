TEXDIR = ./tex/

all:
	make slides

slides:
	pdflatex --shell-escape --interaction=batchmode $(TEXDIR)PACE.tex

slides-debug:
	pdflatex --shell-escape $(TEXDIR)PACE.tex

clean:
	for FILE in `find . -type f | grep -e "\.pyg\|\.aux\|\.log\|\.toc\|\.out\|\.equ\|\.lof\|\.tar\.gz"`; do rm $$FILE; done

push:
	echo "Not Implemented"
