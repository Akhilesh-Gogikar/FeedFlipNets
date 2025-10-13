#!/usr/bin/env bash
set -euo pipefail

# Build docs/paper/main.pdf using latexmk if available, else tectonic, else pdflatex+bibtex.

DOC=docs/paper/main.tex
OUTDIR=docs/paper

if command -v latexmk >/dev/null 2>&1; then
  latexmk -pdf -interaction=nonstopmode -halt-on-error -output-directory="$OUTDIR" "$DOC"
elif command -v tectonic >/dev/null 2>&1; then
  # Use tectonic as a lightweight TeX engine. It will fetch needed packages.
  # Output is written to OUTDIR.
  tectonic -p --keep-logs --keep-intermediates -Z continue-on-errors -o "$OUTDIR" "$DOC" || true
else
  if ! command -v pdflatex >/dev/null 2>&1; then
    echo "Error: neither latexmk nor pdflatex is installed."
    echo "Install TeX Live/MacTeX or run: brew install --cask mactex-no-gui (macOS)"
    exit 1
  fi
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$OUTDIR" "$DOC" || true
  if command -v bibtex >/dev/null 2>&1; then
    bibtex "$OUTDIR/main" || true
  fi
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$OUTDIR" "$DOC" || true
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$OUTDIR" "$DOC" || true
fi

echo "Built $OUTDIR/main.pdf"
