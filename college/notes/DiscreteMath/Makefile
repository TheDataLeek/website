TEXFILE = DiscreteMathNotes.tex
TEXREV = FinalReview.tex
TEXDIR = ./tex/
OUTPUT = DiscreteMathNotes.pdf
REVPUT = FinalReview.pdf

all: attachments.tar.gz
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

FinalReview.pdf:
	xelatex --shell-escape $(TEXDIR)$(TEXREV)
	xelatex --shell-escape $(TEXDIR)$(TEXREV)
	xelatex --shell-escape $(TEXDIR)$(TEXREV)
	make clean
