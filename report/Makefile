PAPER = yelp_report
TEX = $(wildcard *.tex)
BIB = references.bib
FIGS = $(wildcard figs/*.pdf figs/*.png)
DATA = $(wildcard data/*.csv)

.PHONY: all clean

$(PAPER).pdf: $(TEX) $(BIB) $(FIGS)
	latexmk -pdf ${PAPER}.tex

clean:
	latexmk -CA
