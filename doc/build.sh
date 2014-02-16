#!/bin/sh

set -x
set -e

pdflatex main
# makeindex main.nlo -s nomencl.ist -o main.nls
bibtex main
pdflatex main
pdflatex main
open main.pdf
