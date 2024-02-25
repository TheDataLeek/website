TEXFILE = DifferentialEquationsNotes.tex
TEXDIR = ./tex/
OUTPUT = DifferentialEquationsNotes.pdf

all: attachments.tar.gz
	pdflatex $(TEXDIR)$(TEXFILE)
	pdflatex $(TEXDIR)$(TEXFILE)
	pdflatex $(TEXDIR)$(TEXFILE)
	make clean
	wc $(TEXDIR)m* $(TEXDIR)$(TEXFILE) $(TEXDIR)f*

clean:
	for FILE in `find . -type f | grep -e "\.aux\|\.log\|\.toc\|\.out\|\.equ\|\.lof"`; do rm $$FILE; done

attachments.tar.gz: FinalReview.pdf
	tar -zcf attachments.tar.gz $(TEXDIR)*.tex FinalReview.pdf
	mv attachments.tar.gz $(TEXDIR)

FinalReview.pdf:
	pdflatex $(TEXDIR)FinalReview.tex
	pdflatex $(TEXDIR)FinalReview.tex
	pdflatex $(TEXDIR)FinalReview.tex
