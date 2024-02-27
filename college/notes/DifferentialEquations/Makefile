TEXFILE = DifferentialEquationsNotes.tex
TEXDIR = ./tex/
OUTPUT = DifferentialEquationsNotes.pdf

all: FinalReview.pdf
	xelatex --shell-escape $(TEXDIR)$(TEXFILE)
	xelatex --shell-escape $(TEXDIR)$(TEXFILE)
	xelatex --shell-escape $(TEXDIR)$(TEXFILE)
	make clean
	wc $(TEXDIR)m* $(TEXDIR)$(TEXFILE) $(TEXDIR)f*

clean:
	for FILE in `find . -type f | grep -e "\.aux\|\.log\|\.toc\|\.out\|\.equ\|\.lof"`; do rm $$FILE; done

attachments.tar.gz: FinalReview.pdf
	tar -zcf attachments.tar.gz $(TEXDIR)*.tex FinalReview.pdf
	mv attachments.tar.gz $(TEXDIR)

FinalReview.pdf:
	xelatex --shell-escape $(TEXDIR)FinalReview.tex
	xelatex --shell-escape $(TEXDIR)FinalReview.tex
	xelatex --shell-escape $(TEXDIR)FinalReview.tex
