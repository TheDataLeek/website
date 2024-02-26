TEXFILE = ComputerSystemsNotes.tex
TEXDIR = ./tex/
OUTPUT = ComputerSystemsNotes.pdf

all: attachments.tar.gz CSFinalReview.pdf
	xelatex --shell-escape $(TEXDIR)$(TEXFILE)
	xelatex --shell-escape $(TEXDIR)$(TEXFILE)
	xelatex --shell-escape $(TEXDIR)$(TEXFILE)
	make clean
	wc $(TEXDIR)m* $(TEXDIR)$(TEXFILE)

clean:
	for FILE in `find . -type f | grep -e "\.pyg\|\.aux\|\.log\|\.toc\|\.out\|\.equ\|\.lof\|\.tar\.gz"`; do rm $$FILE; done

attachments.tar.gz:
	tar -zcf attachments.tar.gz $(TEXDIR)*.tex
	mv attachments.tar.gz $(TEXDIR)

CSFinalReview.pdf:
	xelatex --shell-escape $(TEXDIR)CSFinalReview.tex
	xelatex --shell-escape $(TEXDIR)CSFinalReview.tex
	xelatex --shell-escape $(TEXDIR)CSFinalReview.tex
