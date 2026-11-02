# ABMIL vs CLAM-SB (paired, same test slides)

Test slides compared: **60**

| Model | QWK | Accuracy | Balanced acc. | Macro F1 |
| --- | --- | --- | --- | --- |
| CLAM-SB, no instance loss | 0.7023 | 0.4000 | 0.4000 | 0.3583 |
| CLAM-SB + instance loss | 0.7137 | 0.4000 | 0.4000 | 0.3360 |

## Paired statistics

- QWK difference (CLAM-SB + instance loss - CLAM-SB, no instance loss): **+0.0114** (95% bootstrap CI [-0.0930, +0.1141], 2000 resamples)
- McNemar exact test: 4 vs 4 discordant slides, p = 1.0000

Interpretation guard: with only 60 test slides a non-significant McNemar test means *inconclusive*, not *equivalent*. Report the CI, not just the point estimate.
