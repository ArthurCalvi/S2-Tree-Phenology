## Reading the Reliability Diagrams

These panels compare the evergreen posterior produced by the two Random Forest variants:

- **Left:** EMB-14 Random Forest (embedding features).
- **Right:** HARM-14 Random Forest (harmonic features).

### Axes
- **X-axis:** Predicted evergreen probability (`p_evergreen`) emitted by the Random Forest, binned into 10 equal-width intervals (0.0–0.1, 0.1–0.2, …, 0.9–1.0).
- **Y-axis:** Fraction of true evergreens or average confidence for each bin.

### Elements
- **Grey dashed line (Ideal):** Perfect calibration where predicted probabilities match observed evergreen frequency.
- **Coloured line with circles (Mean confidence):** Average `p_evergreen` within each bin. Blue corresponds to EMB-14, orange to HARM-14.
- **Black line with squares (Empirical accuracy):** Observed evergreen fraction among samples in the bin. The closer this line is to the dashed diagonal, the more reliable the probabilities.
- **Shaded histogram bars (Sample fraction):** Proportion of validation pixels that fall into each probability bin. Tall bars indicate where the Random Forest places most of its confidence.

### How to Interpret
1. **Follow a bin on the X-axis.** The blue/orange point tells you what probability the Random Forest assigned to the pixels in that bin. The black square shows how many of those pixels were actually evergreen.
2. **Compare to the diagonal.** Points on the diagonal indicate perfect calibration (e.g., a 0.7 prediction corresponds to 70% evergreen pixels). If the coloured point sits above the black square the forest is slightly over-confident in that bin; if it sits below, it is under-confident. Deviations quantify calibration error (ECE/MCE reported above each panel).
3. **Use the histogram.** Concentration near 0 or 1 means the Random Forest is confident; if the black square is close to the diagonal, this confidence is justified. A mismatch would suggest over- or under-confidence.
4. **Link to threshold tuning.** Moving the evergreen decision threshold (e.g., from 0.50 to 0.55) selects different bins. If those bins lie near the diagonal, the Random Forest probabilities can be treated as real-world frequencies, making threshold adjustments predictable.

### Relation to the Models
- Both panels use out-of-fold probabilities from the same tile-stratified cross-validation scheme. The only difference is the feature set the Random Forest consumes (EMB-14 vs HARM-14).
- The lower Expected Calibration Error (ECE) and Maximum Calibration Error (MCE) values for EMB-14 indicate that its posterior probabilities align more closely with observed outcomes, enabling more reliable post-hoc thresholding than the harmonic baseline.*** End Patch
