# Manuscript source

`main.tex`, `refs.bib`, `jmlr2e.sty`. Appendix C of `main.tex` contains the
supplementary derivations (S.2-S.9), placed after the conclusion and before
`\bibliography{refs}` as JMLR requires. There is no separately compiled
supplementary document.

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

`\graphicspath{{figs/}{../figs/}}` finds the eight JPEG figures both when
compiling from this directory and from the repository root. On Overleaf,
place `main.tex` and `figs/` at the project root of a single project and set
`main.tex` as the main document.

Notes:

- `jmlr2e.sty` is the official unmodified style file, loaded with `preprint`.
- 34 BibTeX entries; 61 numbered display equations in total -- the main
  text uses `eq:e01`-`eq:e50`, Appendix C continues with (51)-(61);
  cross-references via `\eqref`. Only (35) and (43) wrap onto multiple lines.
- All eight figures are 6.0 x 3.2 in, 600 dpi JPEG (3600 x 1920 px), embedded 1:1 at `width=\textwidth`; every text element is >= 9.5 pt at its final printed size. Legends are horizontal strips below each subplot, outside the plotting area. Glyphs are rasterised by Agg, so the final PDF contains no nested CID font subsets or transparency groups; JPEG uses DCTDecode pass-through, unlike PNG which pdfTeX re-encodes.
- Hyperref bookmarks are disabled and `lastpage` is not loaded; the figures
  keep a single free-tier Overleaf compile within its time limit.
- `\acks{}` records the no-external-funding / no-competing-interests
  statement supplied by the author; `\editor{}` is empty, which is normal at
  submission. No placeholder text remains.

The LaTeX source is a byte-identical mirror of the Overleaf
submission source. If the manuscript is revised, update both copies together
(or remove `paper/` and keep this repository code/data only).
