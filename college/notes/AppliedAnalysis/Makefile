TEXDIR = ./tex/

all:
	make notes

notes:
	xelatex --shell-escape -interaction=batchmode $(TEXDIR)AppliedAnalysis1.tex
	xelatex --shell-escape -interaction=batchmode $(TEXDIR)AppliedAnalysis1.tex
	xelatex --shell-escape -interaction=batchmode $(TEXDIR)AppliedAnalysis1.tex

notes-debug:
	xelatex --shell-escape $(TEXDIR)AppliedAnalysis1.tex

clean:
	for FILE in `find . -type f | grep -e "\.pyg\|\.aux\|\.log\|\.toc\|\.out\|\.equ\|\.lof\|\.tar\.gz"`; do rm $$FILE; done
