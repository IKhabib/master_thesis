#!/usr/bin/env python3
"""Build the thesis with a clean, explicitly selected bibliography backend."""

import argparse
from pathlib import Path
import shutil
import subprocess
import sys


THESIS_DIR = Path(__file__).resolve().parent
GENERATED_SUFFIXES = (
    ".aux", ".bbl", ".blg", ".bcf", ".run.xml", ".toc", ".lof",
    ".lot", ".out", ".fls", ".fdb_latexmk", ".log", ".synctex.gz",
)
REFERENCE_SUFFIXES = (".aux", ".toc", ".lof", ".lot", ".out")


def required_program(name):
    program = shutil.which(name)
    if program is None:
        raise RuntimeError("Required program is not on PATH: " + name)
    return program


def biblatex_available():
    kpsewhich = shutil.which("kpsewhich")
    if kpsewhich is None:
        return False
    result = subprocess.run(
        [kpsewhich, "--progname=pdflatex", "biblatex.sty"],
        cwd=THESIS_DIR,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0 and bool(result.stdout.strip())


def run(command):
    print("Running: " + " ".join(command), flush=True)
    subprocess.run(command, cwd=THESIS_DIR, check=True)


def reference_state():
    state = []
    for suffix in REFERENCE_SUFFIXES:
        path = THESIS_DIR / ("main" + suffix)
        state.append(path.read_bytes() if path.is_file() else None)
    return tuple(state)


def build(requested_backend):
    if not (THESIS_DIR / "main.tex").is_file():
        raise RuntimeError("main.tex must be beside build.py.")

    pdflatex = required_program("pdflatex")
    has_biber = shutil.which("biber") is not None
    has_biblatex = biblatex_available()
    backend = requested_backend
    if backend == "auto":
        backend = "biber" if has_biber and has_biblatex else "bibtex"
    if backend == "biber" and not has_biblatex:
        raise RuntimeError(
            "Biber was requested, but kpsewhich could not locate biblatex.sty. "
            "Install BibLaTeX or use --backend bibtex."
        )
    bibliography_program = required_program(backend)

    # Remove only this document's generated files, never source, figures or PDF.
    # In particular, a .bbl produced by the other backend must not be reused.
    removed = []
    for suffix in GENERATED_SUFFIXES:
        path = THESIS_DIR / ("main" + suffix)
        if path.is_file():
            path.unlink()
            removed.append(path.name)
    print("Bibliography backend: " + backend, flush=True)
    if removed:
        print("Cleared generated files: " + ", ".join(removed), flush=True)

    tex_input = (
        r"\def\UseBibTeX{1}\input{main.tex}"
        if backend == "bibtex" else "main.tex"
    )
    latex_command = [
        pdflatex, "-no-shell-escape", "-interaction=nonstopmode",
        "-halt-on-error", "-file-line-error", "-jobname=main", tex_input,
    ]
    run(latex_command)
    run([bibliography_program, "main"])

    # At least two further passes resolve citations, lists and cross-references.
    # Continue when pagination changes have not yet settled.
    for pass_number in range(1, 6):
        previous = reference_state()
        run(latex_command)
        if pass_number >= 2 and reference_state() == previous:
            break
    else:
        raise RuntimeError(
            "Cross-references did not settle after five bibliography follow-up "
            "passes. Inspect main.log before using main.pdf."
        )

    log = (THESIS_DIR / "main.log").read_text(encoding="utf-8", errors="replace")
    unresolved = (
        "There were undefined references",
        "There were undefined citations",
        "Please (re)run Biber",
        "Label(s) may have changed",
    )
    if any(message in log for message in unresolved):
        raise RuntimeError(
            "The PDF was generated, but main.log reports unresolved citations "
            "or references. Inspect the log before using the PDF."
        )
    print("Build complete: " + str(THESIS_DIR / "main.pdf"), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", choices=("auto", "biber", "bibtex"), default="auto",
        help="auto prefers Biber when both Biber and BibLaTeX are available",
    )
    args = parser.parse_args()
    try:
        build(args.backend)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print("Build failed: " + str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
