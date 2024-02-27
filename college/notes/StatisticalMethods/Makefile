TEXDIR = ./tex/

all:
	make notes
	make m2
	make final
	make review
	make clean

notes:
	xelatex --shell-escape $(TEXDIR)StatisticalMethods.tex
	xelatex --shell-escape $(TEXDIR)StatisticalMethods.tex

review:
	xelatex --shell-escape $(TEXDIR)StatsFinalReview.tex
	xelatex --shell-escape $(TEXDIR)StatsFinalReview.tex
m2:
	Rscript -e "library('knitr'); knit('$(TEXDIR)m2.Rtex')"; mv m2.tex $(TEXDIR)tmp.tex
	sed -e "s/\\\usepackage\[\]{color}/\\\usepackage\[\]{xcolor}/" $(TEXDIR)tmp.tex > $(TEXDIR)m2.tex
	rm $(TEXDIR)tmp.tex
	xelatex --shell-escape $(TEXDIR)m2.tex
	xelatex --shell-escape $(TEXDIR)m2.tex

final:
	Rscript -e "library('knitr'); knit('$(TEXDIR)takehome_final.Rtex')"
	mv takehome_final.tex $(TEXDIR)tmp.tex
	sed -e "s/\\\usepackage\[\]{color}/\\\usepackage\[\]{xcolor}/" $(TEXDIR)tmp.tex > $(TEXDIR)takehome_final.tex
	rm $(TEXDIR)tmp.tex
	xelatex --shell-escape $(TEXDIR)takehome_final.tex
	xelatex --shell-escape $(TEXDIR)takehome_final.tex

notebook:
	ipython nbconvert --to latex --post pdf ./Takehome_Final.ipynb
	ipython nbconvert --to html ./Takehome_Final.ipynb

clean:
	for FILE in `find . -type f | grep -e "\.pyg\|\.aux\|\.log\|\.toc\|\.out\|\.equ\|\.lof\|\.tar\.gz"`; do rm $$FILE; done

