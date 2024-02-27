TEXDIR = ./tex/

notes:
	xelatex -shell-escape --interaction=batchmode $(TEXDIR)FourierSeries.tex
	xelatex -shell-escape --interaction=batchmode $(TEXDIR)FourierSeries.tex
	xelatex -shell-escape --interaction=batchmode $(TEXDIR)FourierSeries.tex

hw1:
	Pweave --figure-directory=/home/zoe/classwork/2015b/appm4350/homeworks/figures -f texminted $(TEXDIR)hw1.Ptex
	xelatex --shell-escape -interaction=batchmode $(TEXDIR)hw1.tex
	xelatex --shell-escape -interaction=batchmode $(TEXDIR)hw1.tex

hw2:
	Pweave --figure-directory=/home/zoe/classwork/2015b/appm4350/homeworks/figures -f texminted $(TEXDIR)hw2.Ptex
	xelatex --shell-escape -interaction=batchmode $(TEXDIR)hw2.tex
	xelatex --shell-escape -interaction=batchmode $(TEXDIR)hw2.tex

hw3:
	Pweave --figure-directory=/home/zoe/classwork/2015b/appm4350/homeworks/figures -f texminted $(TEXDIR)hw3.Ptex
	xelatex --shell-escape -interaction=batchmode $(TEXDIR)hw3.tex
	xelatex --shell-escape -interaction=batchmode $(TEXDIR)hw3.tex

hw4:
	Pweave --figure-directory=/home/zoe/classwork/2015b/appm4350/homeworks/figures -f texminted $(TEXDIR)hw4.Ptex
	xelatex --shell-escape -interaction=batchmode $(TEXDIR)hw4.tex
	xelatex --shell-escape -interaction=batchmode $(TEXDIR)hw4.tex

hw5:
	Pweave --figure-directory=/home/zoe/classwork/2015b/appm4350/homeworks/figures -f texminted -c $(TEXDIR)hw5.Ptex
	make hw5-tex

hw5-quick:
	Pweave --figure-directory=/home/zoe/classwork/2015b/appm4350/homeworks/figures -f texminted -d $(TEXDIR)hw5.Ptex
	make hw5-tex

hw5-tex:
	xelatex --shell-escape -interaction=batchmode $(TEXDIR)hw5.tex
	xelatex --shell-escape -interaction=batchmode $(TEXDIR)hw5.tex

hw5-debug:
	xelatex --shell-escape $(TEXDIR)hw5.tex

hw6:
	Pweave --figure-directory=/home/zoe/classwork/2015b/appm4350/homeworks/figures -f texminted -c $(TEXDIR)hw6.Ptex
	make hw6-tex

hw6-quick:
	Pweave --figure-directory=/home/zoe/classwork/2015b/appm4350/homeworks/figures -f texminted -d $(TEXDIR)hw6.Ptex
	make hw6-tex

hw6-tex:
	xelatex --shell-escape -interaction=batchmode $(TEXDIR)hw6.tex
	xelatex --shell-escape -interaction=batchmode $(TEXDIR)hw6.tex

hw6-debug:
	xelatex --shell-escape $(TEXDIR)hw6.tex

hw7:
	Pweave --figure-directory=/home/zoe/classwork/2015b/appm4350/homeworks/figures -f texminted -c $(TEXDIR)hw7.Ptex
	make hw7-tex

hw7-quick:
	Pweave --figure-directory=/home/zoe/classwork/2015b/appm4350/homeworks/figures -f texminted -d $(TEXDIR)hw7.Ptex
	make hw7-tex

hw7-tex:
	xelatex --shell-escape -interaction=batchmode $(TEXDIR)hw7.tex
	xelatex --shell-escape -interaction=batchmode $(TEXDIR)hw7.tex

hw7-debug:
	xelatex --shell-escape $(TEXDIR)hw7.tex

hw8:
	Pweave --figure-directory=/home/zoe/classwork/2015b/appm4350/homeworks/figures -f texminted -c $(TEXDIR)hw8.Ptex
	make hw8-tex

hw8-quick:
	Pweave --figure-directory=/home/zoe/classwork/2015b/appm4350/homeworks/figures -f texminted -d $(TEXDIR)hw8.Ptex
	make hw8-tex

hw8-tex:
	xelatex --shell-escape -interaction=batchmode $(TEXDIR)hw8.tex
	xelatex --shell-escape -interaction=batchmode $(TEXDIR)hw8.tex

hw8-debug:
	xelatex --shell-escape $(TEXDIR)hw8.tex

hw9:
	Pweave --figure-directory=/home/zoe/classwork/2015b/appm4350/homeworks/figures -f texminted -c $(TEXDIR)hw9.Ptex
	make hw9-tex

hw9-quick:
	Pweave --figure-directory=/home/zoe/classwork/2015b/appm4350/homeworks/figures -f texminted -d $(TEXDIR)hw9.Ptex
	make hw9-tex

hw9-tex:
	xelatex --shell-escape -interaction=batchmode $(TEXDIR)hw9.tex
	xelatex --shell-escape -interaction=batchmode $(TEXDIR)hw9.tex

hw9-debug:
	xelatex --shell-escape $(TEXDIR)hw9.tex

clean:
	for FILE in `find . -type f | grep -e "\.pyg\|\.aux\|\.log\|\.toc\|\.out\|\.equ\|\.lof\|\.tar\.gz"`; do rm $$FILE; done

push:
	echo "Not Implemented"
