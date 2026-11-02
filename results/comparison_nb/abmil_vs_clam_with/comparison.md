# ABMIL vs CLAM-SB (paired, same test slides)

Test slides compared: **60**

| Model | QWK | Accuracy | Balanced acc. | Macro F1 |
| --- | --- | --- | --- | --- |
| ABMIL | 0.7137 | 0.3667 | 0.3667 | 0.3034 |
| CLAM-SB + instance loss | 0.7137 | 0.4000 | 0.4000 | 0.3360 |

## Paired statistics

- QWK difference (CLAM-SB + instance loss - ABMIL): **+0.0000** (95% bootstrap CI [-0.1122, +0.1078], 2000 resamples)
- McNemar exact test: 1 vs 3 discordant slides, p = 0.6250

Interpretation guard: with only 60 test slides a non-significant McNemar test means *inconclusive*, not *equivalent*. Report the CI, not just the point estimate.
