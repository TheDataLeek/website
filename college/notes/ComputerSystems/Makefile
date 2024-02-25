TEXFILE = ComputerSystemsNotes.tex
TEXDIR = ./tex/
OUTPUT = ComputerSystemsNotes.pdf

all: attachments.tar.gz CSFinalReview.pdf
	pdflatex --shell-escape $(TEXDIR)$(TEXFILE)
	pdflatex --shell-escape $(TEXDIR)$(TEXFILE)
	pdflatex --shell-escape $(TEXDIR)$(TEXFILE)
	make clean
	wc $(TEXDIR)m* $(TEXDIR)$(TEXFILE)

clean:
	for FILE in `find . -type f | grep -e "\.pyg\|\.aux\|\.log\|\.toc\|\.out\|\.equ\|\.lof\|\.tar\.gz"`; do rm $$FILE; done

attachments.tar.gz:
	tar -zcf attachments.tar.gz $(TEXDIR)*.tex
	mv attachments.tar.gz $(TEXDIR)

CSFinalReview.pdf:
	pdflatex --shell-escape $(TEXDIR)CSFinalReview.tex
	pdflatex --shell-escape $(TEXDIR)CSFinalReview.tex
	pdflatex --shell-escape $(TEXDIR)CSFinalReview.tex
