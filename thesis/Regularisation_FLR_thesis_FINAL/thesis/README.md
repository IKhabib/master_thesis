# Regularisation in Functional Linear Regression

**Author:** Khabibullo Ibadullaev  
**Supervisor:** Prof. Dr. Dominik Liebl

Open `main.pdf` to read the revised thesis. **Main text: 34 pages; 35 including the abstract**, within the 40-page allowance. The six main chapters present the questions, methods, simulation, inference, empirical application, and conclusions. The original background chapter is now Appendix A; the extended methods, proofs, results, and diagnostics remain in Appendices B–K.

## Compile the PDF

Install a reasonably complete TeX Live or MiKTeX distribution and Python 3. From this folder, run:

```bash
python3 build.py
```

On Windows, use `python build.py`. The helper chooses Biber when available, otherwise BibTeX; clears stale bibliography files; compiles until references settle; and reports the main-text page count. It stops with an error if the main text plus the one-page abstract exceeds 40 pages.

To reproduce the bibliography backend used for the supplied PDF:

```bash
python3 build.py --backend bibtex
```

With Biber installed, the usual alternative is:

```bash
latexmk -pdf main.tex
```

The input is **`main.tex`**, not `main.pdf`. If switching bibliography backends, use `build.py` to avoid the previous “main.bbl not created by biblatex” error. On Overleaf, choose `main.tex` as the main document, pdfLaTeX as the compiler, and recompile from scratch after replacing the project.

## Files

| Path | Purpose |
| --- | --- |
| `main.tex` | Document order, metadata, bibliography, and page-count marker |
| `author_details.tex` | Author, supervisor, degree, and submission date |
| `bonn_layout.tex` | Font sizes, line spacing, margins, headings, and table layout |
| `concise/` | Six main chapters |
| `extended/` | Appendices A–K containing the detailed material |
| `frontmatter/` | Abstract, AI-assistance disclosure, and unsigned declaration |
| `figures/` | All figures needed to compile the thesis |
| `figure_scripts/` | The four figure-generation scripts supplied in the uploaded thesis archive |
| `thesis_bibliography.bib` | Bibliographic sources |
| `REVISION_NOTES.md` | Chapter mapping, checks, and remaining submission tasks |
| `TEMPLATE_NOTES.md` | Layout settings and template attribution |
| `original_upload.zip` | Unchanged copy of the source ZIP supplied for this revision |
