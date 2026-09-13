# Revision notes — 13 September 2026

## What changed

The original Chapter 2 has become Appendix A. Moving only that chapter would not have brought the original main text within the 40-page allowance, so the remaining material is presented as six concise main chapters with the detailed exposition preserved in appendices. No numerical experiments were changed or rerun for this revision.

| Original material | Revised location |
| --- | --- |
| Chapter 1: introduction | Main Chapter 1, revised |
| Chapter 2: background | Appendix A |
| Appendix A: definitions and proofs | Appendix B |
| Chapter 3: methods | Main Chapter 2; full treatment in Appendix C |
| Chapter 4: simulation design | Main Chapter 3; full treatment in Appendix D |
| Chapter 5: estimation results | Main Chapter 3; full treatment in Appendix E |
| Appendix B: simulation diagnostics | Appendix F |
| Chapter 6: inference methodology | Main Chapter 4; full treatment in Appendix G |
| Chapter 7: inference results | Main Chapter 4; full treatment in Appendix H |
| Appendix C: inference diagnostics | Appendix I |
| Chapter 8: spectroscopy application | Main Chapter 5; full treatment in Appendix J |
| Appendix D: spectroscopy supplement | Appendix K |
| Chapter 9: conclusion | Main Chapter 6, revised |

The main text keeps the essential model assumptions, the four procedures and their different objectives, simulation design and principal risk summaries, inference assumptions and limitations, the empirical protocol and results, and the conclusions. All 322 labels from the relocated technical chapters and appendices are retained, along with their displayed mathematical material, tables, and figure references. The two references to the former introductory test statistic now point to its definition in main Chapter 4. Original introduction and conclusion sources remain in `original_upload.zip`.

Additional corrections clarify the centred PCA reconstruction, squared prediction norm, variance-basis description, empirical contrast signs, wavelength provenance, and the normalisation used in the Arnoldi projection. Dense tables were reformatted at 10 pt; one wide table was divided into panels, and the full error-distribution figure was allowed to float to avoid clipping. The removed reproducibility-manifest section has not been restored.

## Checks and scope

- Built from LaTeX with pdfLaTeX and BibTeX, including repeated passes to settle references.
- Checked the revised numerical summaries against the supplied tables and preserved the saved figures. This is a document revision, not an independent rerun of the analyses.
- Checked the moved labels and equation/figure/table preservation, appendix references, body page count, and table widths.
- Visually reviewed the rendered document, with detailed inspection of mathematical pages, dense tables, figures, title page, and declaration. Corrected the clipping and excessive vertical stretching found during that review.
- Measured body font size and baseline against the requested layout. Times-compatible fonts are embedded in the supplied PDF.
- The undated Eigenvector catalogue remains undated; the recorded access date is not treated as its publication year. BibTeX may issue a harmless “empty year” warning for this entry.
- Biber compilation is supported by the source but was not tested in this environment. Recheck the reported page count after any local changes.

The exact final page count and validation result are recorded in `BUILD_CHECK.txt`.

## Author's remaining submission tasks

1. Read and approve the shortened main text and check the title, degree, personal details, and actual submission date against the registered thesis information.
2. Confirm the AI-assistance arrangements with the supervisor. The university instructions require prior explicit agreement; this revision does not establish that such agreement exists. Review `frontmatter/ai_disclosure.tex` for an accurate account and any required additional detail.
3. Verify the declaration before signing it, including the statement about prior publication. If thesis material is public on GitHub, clarify its treatment with the supervisor or examination office rather than assuming that the declaration is satisfied.
4. Complete the date and original handwritten signature. The supplied declaration is intentionally unsigned and undated.
5. Follow the examination office's required PDF and bound-hardcopy submission procedure by the deadline. Sending a PDF or GitHub link to the supervisor is a review step, not the formal submission.

The page-count and layout checks support compliance with the supplied instructions; they are not a decision by the examination office on the individual submission.
