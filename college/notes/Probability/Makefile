TEXDIR = ./tex/

all:
	make ps2
	make ps3
	make ps4
	make ps5
	make ps7
	make ps8
	make ps9
	make ps10
	make ps12
	make ps13
	make ps14
	make clean

ps2:
	pdflatex --shell-escape $(TEXDIR)2.tex
	pdflatex --shell-escape $(TEXDIR)2.tex

ps3:
	pdflatex --shell-escape $(TEXDIR)3.tex
	pdflatex --shell-escape $(TEXDIR)3.tex

ps4:
	pdflatex --shell-escape $(TEXDIR)4.tex
	pdflatex --shell-escape $(TEXDIR)4.tex

ps5:
	pdflatex --shell-escape $(TEXDIR)5.tex
	pdflatex --shell-escape $(TEXDIR)5.tex

ps7:
	pdflatex --shell-escape $(TEXDIR)7.tex
	pdflatex --shell-escape $(TEXDIR)7.tex

ps8:
	pdflatex --shell-escape $(TEXDIR)8.tex
	pdflatex --shell-escape $(TEXDIR)8.tex

ps9:
	Rscript -e "library('knitr'); knit('$(TEXDIR)9.Rtex')"; mv 9.tex $(TEXDIR)tmp.tex
	sed -e "s/\\\usepackage\[\]{color}/\\\usepackage\[\]{xcolor}/" $(TEXDIR)tmp.tex > $(TEXDIR)9.tex
	rm $(TEXDIR)tmp.tex
	pdflatex --shell-escape $(TEXDIR)9.tex
	pdflatex --shell-escape $(TEXDIR)9.tex

ps10:
	Rscript -e "library('knitr'); knit('$(TEXDIR)10.Rtex')"; mv 10.tex $(TEXDIR)tmp.tex
	sed -e "s/\\\usepackage\[\]{color}/\\\usepackage\[\]{xcolor}/" $(TEXDIR)tmp.tex > $(TEXDIR)10.tex
	rm $(TEXDIR)tmp.tex
	pdflatex --shell-escape $(TEXDIR)10.tex
	pdflatex --shell-escape $(TEXDIR)10.tex

ps12:
	Rscript -e "library('knitr'); knit('$(TEXDIR)12.Rtex')"; mv 12.tex $(TEXDIR)tmp.tex
	sed -e "s/\\\usepackage\[\]{color}/\\\usepackage\[\]{xcolor}/" $(TEXDIR)tmp.tex > $(TEXDIR)12.tex
	rm $(TEXDIR)tmp.tex
	pdflatex --shell-escape $(TEXDIR)12.tex
	pdflatex --shell-escape $(TEXDIR)12.tex

ps13:
	Rscript -e "library('knitr'); knit('$(TEXDIR)13.Rtex')"; mv 13.tex $(TEXDIR)tmp.tex
	sed -e "s/\\\usepackage\[\]{color}/\\\usepackage\[\]{xcolor}/" $(TEXDIR)tmp.tex > $(TEXDIR)13.tex
	rm $(TEXDIR)tmp.tex
	pdflatex --shell-escape $(TEXDIR)13.tex
	pdflatex --shell-escape $(TEXDIR)13.tex

ps14:
	Rscript -e "library('knitr'); knit('$(TEXDIR)14.Rtex')"; mv 14.tex $(TEXDIR)tmp.tex
	sed -e "s/\\\usepackage\[\]{color}/\\\usepackage\[\]{xcolor}/" $(TEXDIR)tmp.tex > $(TEXDIR)14.tex
	rm $(TEXDIR)tmp.tex
	pdflatex --shell-escape $(TEXDIR)14.tex
	pdflatex --shell-escape $(TEXDIR)14.tex

clean:
	for FILE in `find . -type f | grep -e "\.pyg\|\.aux\|\.log\|\.toc\|\.out\|\.equ\|\.lof\|\.tar\.gz"`; do rm $$FILE; done
