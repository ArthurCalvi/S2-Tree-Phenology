# Writing Guidelines

Adapted from “Writing a Good Scientific Paper” (M. D. Black, 2017) for the S2 phenology manuscript. Use this sheet before each drafting pass.

## Story Spine
- **Purpose first:** State the problem, the action we took, and the outcome in that order. Every section must reinforce this arc.
- **One idea per paragraph:** Lead with the key message, follow with evidence, end with the takeaway sentence.
- **Evidence beats opinion:** Replace adjectives with numbers, cite the source, and explain why the result matters.

## IMRaD Checkpoints
- **Introduction:** Move from field context to the exact gap; close with the objective and hypotheses.
- **Methods:** Describe data, preprocessing, models, and evaluation so another group could reproduce the work.
- **Results:** Report findings in the same sequence as the hypotheses. Pair each claim with figures/tables or statistics.
- **Discussion:** Interpret results, note limitations, and anchor implications to real applications—no new data here.

## Sentence Craft
- Keep sentences <22 words when possible. Split long thoughts into declarative statements.
- Prefer active voice (“We trained…”) unless the actor is irrelevant.
- Define acronyms on first mention; reuse the same term afterwards.
- Use precise verbs (e.g., “improves recall by 6%”) instead of vague qualifiers.

## Visuals and References
- Mention each figure/table once in the text and highlight the insight the reader should confirm.
- Ensure captions stand alone, but avoid copying them verbatim into the body.
- Cite only keys present in `article/manuscript/references.bib`; verify numbers against the latest outputs in `results/`.

## Revision Loop
- Check the outline (`article/backbone/outline.yml`) and SPG (`article/backbone/spg.yml`) after each drafting pass.
- Record gaps or future experiments as `\todo{}` comments or outline notes—remove them before submission.
- Reread the abstract and conclusion last to confirm they match the updated evidence.
