#!/usr/bin/env bash
source=report

rm ${source}*out ${source}*aux ${source}*log ${source}*pdf ${source}*~ ${source}*bbl ${source}*blg
pdflatex ${source}.tex
bibtex ${source}.aux
pdflatex ${source}.tex
pdflatex ${source}.tex
