# ABMIL vs CLAM-SB (paired, same test slides)

Test slides compared: **60**

| Model | QWK | Accuracy | Balanced acc. | Macro F1 |
| --- | --- | --- | --- | --- |
| ABMIL | 0.7203 | 0.3667 | 0.3667 | 0.3031 |
| CLAM-SB | 0.7137 | 0.4000 | 0.4000 | 0.3360 |

## Paired statistics

- QWK difference (CLAM-SB - ABMIL): **-0.0066** (95% bootstrap CI [-0.1246, +0.1059], 2000 resamples)
- McNemar exact test: 2 vs 4 discordant slides, p = 0.6875

Interpretation guard: with only 60 test slides a non-significant McNemar test means *inconclusive*, not *equivalent*. Report the CI, not just the point estimate.
