#!/usr/bin/env bash
set -e
TARGET_DIR="../results/isodither"
for file in "$TARGET_DIR"/*.pgf; do
  [ -e "$file" ] || continue
  NAME=$(basename "$file" .pgf)
  echo "Converting $file -> $TARGET_DIR/$NAME.pdf"
  pdflatex -jobname="$NAME" -output-directory="$TARGET_DIR" \
    "\documentclass[tikz]{standalone}\usepackage{amsmath,amssymb}\newcommand{\mathdefault}[1]{#1}\begin{document}\input{$file}\end{document}" > /dev/null
done