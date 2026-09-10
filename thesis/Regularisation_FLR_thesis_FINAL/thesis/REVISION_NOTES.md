# Earlier editorial, empirical, and production revision through Chapter 9

This file records the preceding checkpoint. For the subsequent final
corrections and their validation, see `FINAL_CORRECTIONS.md` in this directory.
Current compilation instructions are in `README_BUILD.md`.

This checkpoint contains the complete nine-chapter thesis, all four
appendices, the bibliography database, the frozen empirical analysis, and a
verified full-document PDF build.

## What was revised

- Replaced dry end-of-chapter summaries with transitions that carry the
  argument into the next chapter.
- Strengthened the distinction between identification, ill-posedness,
  statistical regularisation, numerical stabilisation, and inference.
- Clarified the fixed-component relationship among NIPALS, APLS, raw FPLS,
  and Arnoldi FPLS, while keeping released-code CG--FPLS distinct through its
  moment-residual objective and stopping rule.
- Standardised notation for the identified parameter space, positive spectral
  index set, Krylov power vectors, condition numbers, and test statistic.
- Added and cross-referenced the spectroscopy application.
- Reworked Chapter 5 into a continuous evidence-based discussion with explicit
  bridges between models, complexity, numerical tails, bias--variance results,
  paired stable fits, slope distributions, inference, and external scope.
- Moved command-line and software-manifest detail from Chapter 4 to
  `REPRODUCIBILITY.md`.
- Corrected the terminology in the coefficient and normal Q--Q figures; no
  simulation was rerun and no estimator logic or archived numerical result was
  changed.
- Updated the bibliography integration and local Biber build instructions.
- Replaced the Chapter 6 placeholder with a complete CG--FPLS inference
  methodology chapter tied to the frozen 5,000-replication configuration.
- Distinguished the identifiable moment null from a literal slope null, the
  finite-iteration statistic from its direct-moment limit, and estimation
  stopping from the stronger inference condition.
- Separated oracle, empirical-null, and feasible calibration concepts and
  documented which one is used in every reported inference branch.
- Derived fixed- and root-$n$ local-alternative behaviour, the analytic
  five-dimensional confidence ellipsoid, and its pointwise envelope, with
  supporting proofs added to Appendix A.
- Recorded the exact inference command, numerical engine, and the differences
  between the frozen Python implementation and the reference notebook.
- Replaced the Chapter 7 placeholder with a complete inference-results chapter
  reconstructed from the frozen 5,000-replication archive.
- Verified every reproducible CSV quantity against the 102 raw NPZ arrays and
  retained all replications without outcome-based filtering.
- Organised the inference evidence in causal order: stopping diagnostics,
  finite-sample null calibration, power, and test-inverted confidence sets.
- Corrected the distinction between the $m_C=10$ confidence branch and the
  $m=70$ size branch, and documented intentional random-stream reuse.
- Inserted four principal inference figures in Chapter 7 and placed the sparse
  5% oracle-size display in a supplementary inference appendix.
- Added exact stopping, null-size, selected-power, and confidence-set tables,
  with Monte Carlo and interpretation qualifications stated in the prose.
- Added a two-panel $\ell^p$ unit-ball, distance, and duality illustration to
  Chapter 2, explicitly connecting $p=2$ self-duality to Parseval's identity,
  Riesz representation, and the Hilbert geometry used by functional linear
  regression.
- Added the corresponding definitions and a proof of H\"older duality in
  Appendix A, including the finite-dimensional polar identities and the
  infinite-dimensional $\ell^\infty$ endpoint qualification.
- Added a deterministic Python generator for the new vector figure and its
  reproduction command.
- Fixed the final architecture as a compact empirical Chapter 8, a final
  Conclusion and Outlook in Chapter 9, and a detailed spectroscopy supplement
  in Appendix D.
- Updated the introduction, background, estimation-results, and
  inference-results transitions so that the main empirical evidence and its
  complete supplementary audit trail have distinct roles.
- Froze and independently audited the pharmaceutical-tablet NIR study using
  instrument 1 and assay as the predeclared primary analysis, with the
  195-tablet development set kept separate from the 460-tablet designated
  benchmark holdout.
- Added the complete Chapter 8 comparison of CG--FPLS, raw FPLS, Arnoldi FPLS,
  and FPCR, including benchmark prediction, fold-based selection, uncertainty,
  numerical-stability diagnostics, and deliberately limited interpretation.
- Added Appendix D with source hashes, preprocessing definitions, exact split
  and tuning rules, observation-level bootstrap comparisons, residual checks,
  complete numerical paths, and three sensitivity analyses.
- Added Chapter 9, which answers the four research questions, separates
  statistical regularisation from numerical representation, records the
  study's limits, and identifies concrete future work.
- Added seven deterministic thesis-scale spectroscopy figures generated only
  from the frozen data and result archive; no fitting or tuning is performed
  by the presentation script.
- Updated `REPRODUCIBILITY.md` with the frozen command, random-stream design,
  source checksums, archive contents, integrity check, and independent replay
  tolerances.
- Added a conditional Natbib/BibTeX fallback while retaining BibLaTeX/Biber as
  the preferred bibliography workflow, then rebuilt and visually checked the
  complete 153-page document.

## Still to be supplied by the author or institution

- Replace the bracketed author, matriculation number, supervisor, university,
  and submission-date placeholders in `author_details.tex`.
- The final correction adds an abstract. Supply any declaration,
  acknowledgements, logos, or wording required by the university.

## Production build

Run `python build.py` from this directory using a TeX installation. The helper
selects an available bibliography backend and clears incompatible generated
files. See `README_BUILD.md` for Overleaf and manual build instructions.
