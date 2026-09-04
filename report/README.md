# Report

Build with:

```bash
cd report
latexmk -pdf main.tex
```

or, without latexmk:

```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Figures are read from `../experiments/figures/`. Generate them first:

```bash
uv run python -m src.experiments.plot_tradeoff --dataset heart --kwarg batch_size=32 --kwarg epochs=300 --exclude-collapse 4
uv run python -m src.experiments.plot_tradeoff --dataset adult --kwarg batch_size=500
uv run python -m src.experiments.plot_tradeoff --dataset diabetes --kwarg batch_size=500
```

Open items are marked `\todo{...}` in red in the PDF. Search `main.tex` for
`\todo` to find them all.

Before submission, verify every entry in `references.bib` against the
original source. The entries were written from memory and the venues, years
and page numbers should be checked.
