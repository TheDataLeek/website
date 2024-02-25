TEXDIR = ./tex/

all:
	make notes
	make ps1
	make ps2
	make ps3
	make ps4
	make ps5
	make ps6
	make ps7
	make ps8
	make ps9
	make clean

notes:
	pdflatex --shell-escape $(TEXDIR)AlgorithmNotes.tex
	pdflatex --shell-escape $(TEXDIR)AlgorithmNotes.tex

ps1:
	pdflatex --shell-escape $(TEXDIR)homework1.tex
	pdflatex --shell-escape $(TEXDIR)homework1.tex
	mv homework1.pdf ps

ps2:
	pdflatex --shell-escape $(TEXDIR)homework2.tex
	pdflatex --shell-escape $(TEXDIR)homework2.tex
	mv homework2.pdf ps

ps3:
	pdflatex --shell-escape $(TEXDIR)homework3.tex
	pdflatex --shell-escape $(TEXDIR)homework3.tex
	mv homework3.pdf ps

ps4:
	pdflatex --shell-escape $(TEXDIR)homework4.tex
	pdflatex --shell-escape $(TEXDIR)homework4.tex
	mv homework4.pdf ps

ps5:
	pdflatex --shell-escape $(TEXDIR)homework5.tex
	pdflatex --shell-escape $(TEXDIR)homework5.tex
	mv homework5.pdf ps

ps6:
	pdflatex --shell-escape $(TEXDIR)homework6.tex
	pdflatex --shell-escape $(TEXDIR)homework6.tex
	mv homework6.pdf ps

ps7:
	pdflatex --shell-escape $(TEXDIR)homework7.tex
	pdflatex --shell-escape $(TEXDIR)homework7.tex
	mv homework7.pdf ps

ps8:
	Rscript -e "library('knitr'); knit('$(TEXDIR)homework8.Rtex')"; mv homework8.tex $(TEXDIR)tmp.tex
	sed -e "s/\\\usepackage\[\]{color}/\\\usepackage\[\]{xcolor}/" $(TEXDIR)tmp.tex > $(TEXDIR)homework8.tex
	rm $(TEXDIR)tmp.tex
	pdflatex --shell-escape $(TEXDIR)homework8.tex
	pdflatex --shell-escape $(TEXDIR)homework8.tex
	mv homework8.pdf ps

ps9:
	Rscript -e "library('knitr'); knit('$(TEXDIR)homework9.Rtex')"; mv homework9.tex $(TEXDIR)tmp.tex
	sed -e "s/\\\usepackage\[\]{color}/\\\usepackage\[\]{xcolor}/" $(TEXDIR)tmp.tex > $(TEXDIR)homework9.tex
	rm $(TEXDIR)tmp.tex
	pdflatex --shell-escape $(TEXDIR)homework9.tex
	pdflatex --shell-escape $(TEXDIR)homework9.tex
	mv homework9.pdf ps

clean:
	for FILE in `find . -type f | grep -e "\.pyg\|\.aux\|\.log\|\.toc\|\.out\|\.equ\|\.lof\|\.tar\.gz"`; do rm $$FILE; done

