TEXDIR = ./tex/

all:
	make notes
	make clean

notes:
	xelatex --shell-escape -interaction=batchmode $(TEXDIR)NumericalAnalysis.tex
	xelatex --shell-escape -interaction=batchmode $(TEXDIR)NumericalAnalysis.tex
	xelatex --shell-escape -interaction=batchmode $(TEXDIR)NumericalAnalysis.tex

clean:
	for FILE in `find . -type f | grep -e "\.pyg\|\.aux\|\.log\|\.toc\|\.out\|\.equ\|\.lof\|\.tar\.gz"`; do rm $$FILE; done

push:
	echo "Not Implemented"
