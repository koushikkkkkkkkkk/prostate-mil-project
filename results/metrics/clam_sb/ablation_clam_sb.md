# CLAM-SB instance-level supervision ablation

Same features, same split, same seeds; only the instance clustering loss changes.

| Setting | QWK (mean) | QWK (sd) | Accuracy | Balanced acc. | Macro F1 | Seeds |
| --- | --- | --- | --- | --- | --- | --- |
| CLAM-SB + instance loss | 0.7035 | 0.0088 | 0.3833 | 0.3833 | 0.3142 | 3 |
| CLAM-SB, no instance loss | 0.7271 | 0.0223 | 0.3944 | 0.3944 | 0.3386 | 3 |
