# Viewing and compiling the thesis

Open `main.pdf` to read the supplied thesis. Its editable entry point is
`main.tex`; all figures needed to compile it are included. Building the PDF
does not rerun any statistical analysis.

## Complete your personal details

Edit `author_details.tex` before submission. The author name, matriculation
number, supervisor, university and submission month/year are still marked
with brackets because those details have not been supplied. Check the degree
and department against your university's template as well.

The abstract is included in `frontmatter/abstract.tex`. Institution-specific
declarations, acknowledgements, logos and required title-page wording should
be completed according to your university's requirements.

## Files to keep together

Upload or copy the **entire `thesis` folder**, preserving its structure:

```text
thesis/
  main.tex
  author_details.tex
  thesis_bibliography.bib
  build.py
  frontmatter/
  chapters/
  appendices/
  figures/
  figure_scripts/
  README_BUILD.md
  REPRODUCIBILITY.md
  REVISION_NOTES.md
```

The document contains the title page, abstract, contents and figure/table
lists, nine chapters, references and four appendices. The PDF figures are
required inputs; the Python figure scripts are provided for reproducibility.
The empirical Python code, dataset and frozen results are supplied separately
in the complete package's `empirical_application` folder.

## Overleaf or another LaTeX editor

1. Upload the complete `thesis` folder, including `figures`, `frontmatter` and
   `author_details.tex`.
2. Select `main.tex` as the main document and pdfLaTeX as the compiler.
3. Use BibLaTeX/Biber for the bibliography. A full TeX Live installation,
   including TeX Live 2024, supports this preferred route when the relevant
   packages are installed.
4. Compile until citations, contents and cross-references have resolved.

When updating an existing project, delete its old generated `main.bbl` and
other build files listed below, or use the editor's **Recompile from scratch**
option. Do not upload a `.bbl` from an earlier build using the other backend.

## Local build with Python

Install Python 3 and a LaTeX distribution containing pdfLaTeX and the chosen
bibliography processor. From the `thesis` directory, run:

```bash
python3 build.py
```

On Windows, the Python command may be `py` or `python` instead of `python3`.
No third-party Python packages are required by this build helper.

The helper checks for both the `biber` executable and `biblatex.sty` using
`kpsewhich`. If both are available, it selects Biber. Otherwise it explicitly
selects the Natbib/BibTeX fallback. You can also select a backend yourself:

```bash
python3 build.py --backend biber
python3 build.py --backend bibtex
```

The helper removes only `main`'s generated auxiliary files before building,
preventing reuse of a bibliography file from the wrong backend. It then runs
pdfLaTeX, the chosen bibliography processor and enough further pdfLaTeX passes
to settle citations and cross-references. External shell execution by LaTeX
is disabled. The result is `main.pdf`; failures are reported with a nonzero
exit status and details in `main.log`.

## Manual Biber build

The preferred route requires `biblatex.sty` and Biber. After clearing any old
generated files, run these commands in the `thesis` directory:

```bash
pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error main.tex
biber main
pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error main.tex
pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error main.tex
```

Run another pdfLaTeX pass if the log requests one. A single pdfLaTeX pass does
not produce a complete bibliography and resolved cross-references.

## Switching bibliography backends

Before a manual build after changing backends, remove only the generated
files that exist from this list:

```text
main.aux          main.bbl       main.blg       main.bcf
main.run.xml      main.toc       main.lof       main.lot
main.out          main.fls       main.fdb_latexmk
main.log          main.synctex.gz
```

Keep all `.tex` and `.bib` sources, figures and Python files. `build.py` performs
this cleanup automatically. A BibTeX-generated `.bbl` is incompatible with
BibLaTeX and can cause the error “File 'main.bbl' not created by biblatex.”

For the fallback, `build.py --backend bibtex` defines the `\UseBibTeX` flag
before loading `main.tex`. This also works when BibLaTeX is installed but
Biber is unavailable; merely running `bibtex main` after a BibLaTeX pass
does not switch the document to Natbib.

## Validation and reproducibility

The available build environment has pdfLaTeX and BibTeX but no Biber. The
Natbib/BibTeX route was validated locally; a Biber rendering cannot be claimed
as locally tested in that environment. The preferred Biber configuration is
retained for full TeX installations. The two backends can produce small
differences in bibliography formatting and pagination.

Read `REPRODUCIBILITY.md` for the statistical workflows and the complete
package's README for the Python code, dataset and frozen empirical outputs.
The already supplied figures are sufficient for compiling the thesis; figure
regeneration is a separate operation.
