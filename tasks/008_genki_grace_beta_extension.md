# Larger-beta GENKI-head GRACE comparison

User explicitly requested increasing beta and showing the faces, rather than
strengthening the nominal Base. Add beta=1.5,2,3; retain beta=1.2 as the prior
reference. Fix r=2, Base, factuals, references, channels, head, attack bank,
future models, optimizer settings and preservation limits. Do not force visual
change or replace the objective with nominal score maximization.

Read previous raw outputs without modifying them. Run only the new beta searches.
Pool old and new evaluated candidates, including the previous selected incumbent,
so increasing beta cannot worsen the best feasible finite-bank candidate. Screen
all eight cases against the same hard identity/background limits. Use eight GPUs
for independent per-case preservation screening. Evaluate the frozen selected
PNGs uniformly and show Base, previous beta1.2, beta1.5, beta2, beta3 for all cases.

This is a user-requested exploratory extension on previously inspected test
factuals and future models, not a fresh held-out confirmation. No new beta is
selected using future validity. Preserve failed cases, disclose identical outputs,
save summaries/figures and update state.yaml.
