TEXDIR = ./tex/

all:
	make notes
	make ps1
	make ps2
	make ps3
	make ps4
	make ps5
	make ps6
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

ps1:
	xelatex --shell-escape $(TEXDIR)1.tex
	xelatex --shell-escape $(TEXDIR)1.tex

ps2:
	xelatex --shell-escape $(TEXDIR)2.tex
	xelatex --shell-escape $(TEXDIR)2.tex

ps3:
	xelatex --shell-escape $(TEXDIR)3.tex
	xelatex --shell-escape $(TEXDIR)3.tex

ps4:
	xelatex --shell-escape $(TEXDIR)4.tex
	xelatex --shell-escape $(TEXDIR)4.tex

ps5:
	Rscript -e "library('knitr'); knit('$(TEXDIR)5.Rtex')"; mv 5.tex $(TEXDIR)tmp.tex
	sed -e "s/\\\usepackage\[\]{color}/\\\usepackage\[\]{xcolor}/" $(TEXDIR)tmp.tex > $(TEXDIR)5.tex
	rm $(TEXDIR)tmp.tex
	xelatex --shell-escape $(TEXDIR)5.tex
	xelatex --shell-escape $(TEXDIR)5.tex

ps6:
	Rscript -e "library('knitr'); knit('$(TEXDIR)6.Rtex')"; mv 6.tex $(TEXDIR)tmp.tex
	sed -e "s/\\\usepackage\[\]{color}/\\\usepackage\[\]{xcolor}/" $(TEXDIR)tmp.tex > $(TEXDIR)6.tex
	rm $(TEXDIR)tmp.tex
	xelatex --shell-escape $(TEXDIR)6.tex
	xelatex --shell-escape $(TEXDIR)6.tex

m2:
	Rscript -e "library('knitr'); knit('$(TEXDIR)m2.Rtex')"; mv m2.tex $(TEXDIR)tmp.tex
	sed -e "s/\\\usepackage\[\]{color}/\\\usepackage\[\]{xcolor}/" $(TEXDIR)tmp.tex > $(TEXDIR)m2.tex
	rm $(TEXDIR)tmp.tex
	xelatex --shell-escape $(TEXDIR)m2.tex
	xelatex --shell-escape $(TEXDIR)m2.tex

final:
	Rscript -e "library('knitr'); knit('$(TEXDIR)takehome_final.Rtex')"; mv takehome_final.tex $(TEXDIR)tmp.tex
	sed -e "s/\\\usepackage\[\]{color}/\\\usepackage\[\]{xcolor}/" $(TEXDIR)tmp.tex > $(TEXDIR)takehome_final.tex
	rm $(TEXDIR)tmp.tex
	xelatex --shell-escape $(TEXDIR)takehome_final.tex
	xelatex --shell-escape $(TEXDIR)takehome_final.tex

notebook:
	ipython nbconvert --to latex --post pdf ./Takehome_Final.ipynb
	ipython nbconvert --to html ./Takehome_Final.ipynb

clean:
	for FILE in `find . -type f | grep -e "\.pyg\|\.aux\|\.log\|\.toc\|\.out\|\.equ\|\.lof\|\.tar\.gz"`; do rm $$FILE; done

push:
	scp StatisticalMethods.pdf will-farmer.com:/var/www/files/StatisticalMethods.pdf
	scp StatsFinalReview.pdf will-farmer.com:/var/www/files/StatsFinalReview.pdf
	git add ./Takehome_Final.ipynb
	git commit -m 'Update'
	git push -u gist master
