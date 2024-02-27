TEXDIR = ./tex/

all: notes
	echo 'done'

notes:
	xelatex --shell-escape -interaction=batchmode $(TEXDIR)MathematicalStatistics.tex
	xelatex --shell-escape -interaction=batchmode $(TEXDIR)MathematicalStatistics.tex
	xelatex --shell-escape -interaction=batchmode $(TEXDIR)MathematicalStatistics.tex

homework1:
	xelatex --shell-escape -interaction=batchmode $(TEXDIR)homework1.tex
	xelatex --shell-escape -interaction=batchmode $(TEXDIR)homework1.tex
	xelatex --shell-escape -interaction=batchmode $(TEXDIR)homework1.tex

homework2:
	xelatex --shell-escape -interaction=batchmode $(TEXDIR)homework2.tex
	xelatex --shell-escape -interaction=batchmode $(TEXDIR)homework2.tex
	xelatex --shell-escape -interaction=batchmode $(TEXDIR)homework2.tex

homework4:
	xelatex --shell-escape -interaction=batchmode $(TEXDIR)homework4.tex
	xelatex --shell-escape -interaction=batchmode $(TEXDIR)homework4.tex
	xelatex --shell-escape -interaction=batchmode $(TEXDIR)homework4.tex

homework5:
	xelatex --shell-escape -interaction=batchmode $(TEXDIR)homework5.tex
	xelatex --shell-escape -interaction=batchmode $(TEXDIR)homework5.tex
	xelatex --shell-escape -interaction=batchmode $(TEXDIR)homework5.tex

homework6:
	xelatex --shell-escape -interaction=batchmode $(TEXDIR)homework6.tex
	xelatex --shell-escape -interaction=batchmode $(TEXDIR)homework6.tex
	xelatex --shell-escape -interaction=batchmode $(TEXDIR)homework6.tex

homework7:
	xelatex --shell-escape -interaction=batchmode $(TEXDIR)homework7.tex
	xelatex --shell-escape -interaction=batchmode $(TEXDIR)homework7.tex
	xelatex --shell-escape -interaction=batchmode $(TEXDIR)homework7.tex

clean:
	for FILE in `find . -type f | grep -e "\.pyg\|\.aux\|\.log\|\.toc\|\.out\|\.equ\|\.lof\|\.tar\.gz"`; do rm $$FILE; done

push:
	echo "Not Implemented"

pull:
	-rm mathstat_notes_2014.pdf
	wget --user MathStat --password theUMVUEiscoming http://amath.colorado.edu/faculty/corcoran/ProtectDirect/mathstat_notes_2014.pdf
