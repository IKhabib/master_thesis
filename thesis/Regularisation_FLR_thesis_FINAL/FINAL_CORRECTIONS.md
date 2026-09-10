# Final corrections and validation

Prepared 10 September 2026. The corrected document contains 154 PDF pages,
including the title page, abstract, contents and lists, nine chapters,
references, and Appendices A–D.

## Corrections completed

- Distinguished the inference study's 20-component variance pilot and
  stabilised moment-residual engine from the estimation study's
  70-component pilot and short recurrence throughout the discussion.
- Derived the correct fixed-grid covariance eigenvalues for the direct-moment
  limiting reference. Retained the archived continuous-design oracle results
  and explicitly separated grid, finite-sample and reference-simulation effects.
- Corrected the Tikhonov filter convention on the null space, the dual-norm
  supremum, the zero-partial-sum case in the Hölder proof, and the completeness
  step in the spectral representation proof.
- Added the appropriate Cardot–Ferraty–Sarda (2003) spline-estimation citation
  and removed the unsupported publication year from the undated Eigenvector
  catalogue entry while retaining its access date.
- Qualified practical interpretation of the small empirical RMSE contrast
  because physical assay units and an application tolerance are unavailable.
  Preserved the designated-benchmark and conditional-bootstrap qualifications.
- Corrected the statements about which outputs and provenance each numerical
  archive contains, and narrowed the reconstruction claim to stored quantities.
- Regenerated 19 simulation and explanatory PDF assets with readable labels
  at their printed size. Ordinary estimation labels are approximately
  8.45–8.95 points in the compiled thesis. Kept all plotted numerical values,
  raw-FPLS tails, instability markers, paired-subset sizes and omitted-tail
  counts. Seven existing spectroscopy assets are unchanged.
- Updated figure captions to match panel arrangements, the moment-residual
  notation, observed-assay residual axes, and the scope of median envelopes.
- Added a substantive abstract and separated author metadata into
  `thesis/author_details.tex`. Shortened list entries while retaining full
  explanatory captions, and made the figure and table lists one page each.
- Improved the methods comparison table placement and removed an isolated
  two-line page at the end of Appendix A.
- Added a portable clean-build helper and corrected complete-folder upload
  instructions. Source ZIPs contain no generated bibliography or auxiliary
  files from an earlier backend.

## Validation performed

- Independently cross-checked the mathematical and interpretation corrections
  against the final review findings; no remaining substantive contradiction
  was found among those findings.
- Compiled the complete thesis from clean auxiliary state using pdfLaTeX and
  the Natbib/BibTeX fallback, with sufficient passes for references and lists
  to settle. The final LaTeX log has no warnings, overfull boxes, undefined
  references or undefined citations. BibTeX's sole warning is the intentionally
  empty year of the undated Eigenvector source.
- Visually reviewed the complete rendered document, with detailed inspection
  of revised mathematical text, figures, captions, front matter and references.
  The final PDF text-boundary check finds no text outside any page.
- Verified byte-for-byte preservation of 30 estimation-archive files,
  12 inference-archive files, 13 optimisation-illustration files and all
  71 files in the empirical application package. Numerical study results and
  the supplied source data were not rerun or changed for these corrections.
- Included the original simulation programs, the exact empirical source
  snapshots, frozen results, source dataset and final presentation generators
  in the complete project. A SHA-256 manifest covers the packaged files.

The preferred BibLaTeX/Biber branch remains available for a full TeX
installation or Overleaf. This environment did not provide Biber, so the
delivered PDF was validated with the documented BibTeX fallback. Use
`python build.py` for local backend selection, or follow `thesis/README_BUILD.md`.

## Information still required from the author

Fill in the five marked personal/institutional fields in
`thesis/author_details.tex`, confirm the degree and department, and add any
declaration required by the university using its prescribed wording. These
details were not supplied and have not been invented. The abstract is included.
