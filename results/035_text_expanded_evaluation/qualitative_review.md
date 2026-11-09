# Qualitative review of expanded text evaluation

Post-hoc illustrative examples, not human-rated quality labels or a representative
success estimate. All64 examples remain in the PDF/HTML and CSV. Selected is
r0/beta1.5; r.25 produces exactly the same64 sentences.

Reasonable examples for inspection:
- ID12: profanity removed in Base; selected combines the questions into
  "why are they still together?" (the original surprise is attenuated).
- ID41: Base "don 't u kid kid me" becomes "Don't kid me"; cleaner grammar.
- ID46: Base "You never have enough and so many folks are angry over it."
  becomes "You never have enough and so many people are angry over it."
  Mild paraphrase; the insult was already removed by Base.
- ID54: Base "i think thats why i find them so sexy" becomes
  "i think thats why i find them so attractive." This softens wording but
  does not establish that the original sexual compliment was toxic.
- ID15/22/25/29/48/55/61: reasonable Base outputs retained; not GRACE gains.

Failures to retain in reporting:
- ID37 changes "are tolerable" to "are not tolerable", reversing polarity.
- ID42 changes "because that would be messed up ?" to
  "because that would be good", reversing the proposition.
- IDs1/4 select "very few people are just not good", unrelated replacements
  borrowed from the shared population codebook.
- IDs8/9/21/30/38/45/52/53/56/57 include unrelated replacements, missing
  content or substantial meaning drift.
- ID3 retains a racial slur; classifier acceptance is not detoxification.
- ID50 introduces "How does they think we still are." (ungrammatical).

The population-level constraint and one-to-one assignment do not enforce
per-factual meaning. Future validity alone is insufficient to establish useful
text counterfactuals. No parameters or test examples were changed after review.
