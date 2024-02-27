TEXDIR = ./tex/

all:
	make notes
	make ComplexAnalysisReview
	make clean

notes:
	xelatex --shell-escape --interaction=batchmode $(TEXDIR)ComplexVariables.tex
	xelatex --shell-escape --interaction=batchmode $(TEXDIR)ComplexVariables.tex
	xelatex --shell-escape --interaction=batchmode $(TEXDIR)ComplexVariables.tex

notes-debug:
	xelatex --shell-escape $(TEXDIR)ComplexVariables.tex

ComplexAnalysisReview:
	xelatex --shell-escape --interaction=batchmode $(TEXDIR)ComplexAnalysisReview.tex
	xelatex --shell-escape --interaction=batchmode $(TEXDIR)ComplexAnalysisReview.tex
	xelatex --shell-escape --interaction=batchmode $(TEXDIR)ComplexAnalysisReview.tex

review-debug:
	xelatex --shell-escape $(TEXDIR)ComplexAnalysisReview.tex

clean:
	for FILE in `find . -type f | grep -e "\.pyg\|\.aux\|\.log\|\.toc\|\.out\|\.equ\|\.lof\|\.tar\.gz"`; do rm $$FILE; done

