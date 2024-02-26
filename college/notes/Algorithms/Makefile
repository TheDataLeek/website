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
	xelatex --shell-escape $(TEXDIR)AlgorithmNotes.tex
	xelatex --shell-escape $(TEXDIR)AlgorithmNotes.tex

ps1:
	xelatex --shell-escape $(TEXDIR)homework1.tex
	xelatex --shell-escape $(TEXDIR)homework1.tex

ps2:
	xelatex --shell-escape $(TEXDIR)homework2.tex
	xelatex --shell-escape $(TEXDIR)homework2.tex

ps3:
	xelatex --shell-escape $(TEXDIR)homework3.tex
	xelatex --shell-escape $(TEXDIR)homework3.tex

ps4:
	xelatex --shell-escape $(TEXDIR)homework4.tex
	xelatex --shell-escape $(TEXDIR)homework4.tex

ps5:
	xelatex --shell-escape $(TEXDIR)homework5.tex
	xelatex --shell-escape $(TEXDIR)homework5.tex

ps6:
	xelatex --shell-escape $(TEXDIR)homework6.tex
	xelatex --shell-escape $(TEXDIR)homework6.tex

ps7:
	xelatex --shell-escape $(TEXDIR)homework7.tex
	xelatex --shell-escape $(TEXDIR)homework7.tex

ps8:
	Rscript -e "library('knitr'); knit('$(TEXDIR)homework8.Rtex')"; mv homework8.tex $(TEXDIR)tmp.tex
	sed -e "s/\\\usepackage\[\]{color}/\\\usepackage\[\]{xcolor}/" $(TEXDIR)tmp.tex > $(TEXDIR)homework8.tex
	rm $(TEXDIR)tmp.tex
	xelatex --shell-escape $(TEXDIR)homework8.tex
	xelatex --shell-escape $(TEXDIR)homework8.tex

ps9:
	Rscript -e "library('knitr'); knit('$(TEXDIR)homework9.Rtex')"; mv homework9.tex $(TEXDIR)tmp.tex
	sed -e "s/\\\usepackage\[\]{color}/\\\usepackage\[\]{xcolor}/" $(TEXDIR)tmp.tex > $(TEXDIR)homework9.tex
	rm $(TEXDIR)tmp.tex
	xelatex --shell-escape $(TEXDIR)homework9.tex
	xelatex --shell-escape $(TEXDIR)homework9.tex

clean:
	for FILE in `find . -type f | grep -e "\.pyg\|\.aux\|\.log\|\.toc\|\.out\|\.equ\|\.lof\|\.tar\.gz"`; do rm $$FILE; done

