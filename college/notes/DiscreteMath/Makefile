TEXFILE = DiscreteMathNotes.tex
TEXREV = FinalReview.tex
TEXDIR = ./tex/
OUTPUT = DiscreteMathNotes.pdf
REVPUT = FinalReview.pdf

all: attachments.tar.gz
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

push:
	scp $(OUTPUT) william@will-farmer.com:/var/www/files/
	scp $(REVPUT) william@will-farmer.com:/var/www/files/DiscreteMathReview.pdf

FinalReview.pdf:
	pdflatex --shell-escape $(TEXDIR)$(TEXREV)
	pdflatex --shell-escape $(TEXDIR)$(TEXREV)
	pdflatex --shell-escape $(TEXDIR)$(TEXREV)
	make clean
