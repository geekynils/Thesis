#!/bin/sh

find . -name \*.aux -exec rm {} \;
find . -name \*.log -exec rm {} \;
find . -name \*.out -exec rm {} \;
find . -name \*.toc -exec rm {} \;

