# Every run on its own test set

Read the `n slides` column: 60 for the ResNet50 cohort, 57 for CHIEF, 53 for GigaPath. **These rows are not all measured on the same slides** — use the common-cohort table below for any cross-encoder claim.

| model | encoder | n slides | seeds | QWK (mean ± sd) | accuracy |
| --- | --- | --- | --- | --- | --- |
| CLAM-SB | CHIEF (768-d) | 57 | 3 | **0.9207 ± 0.0063** | 0.6023 ± 0.0101 |
| ABMIL | CHIEF (768-d) | 57 | 3 | **0.9035 ± 0.0145** | 0.5614 ± 0.0351 |
| ABMIL | GigaPath (1536-d) | 53 | 3 | **0.8984 ± 0.0108** | 0.5849 ± 0.0327 |
| TransMIL | ResNet50 (1024-d) | 60 | 3 | **0.7446 ± 0.0364** | 0.4833 ± 0.0441 |
| ABMIL | ResNet50 (1024-d) | 60 | 3 | **0.7281 ± 0.0132** | 0.3889 ± 0.0255 |
| CLAM-SB, no instance loss | ResNet50 (1024-d) | 60 | 3 | **0.7271 ± 0.0223** | 0.3944 ± 0.0255 |
| CLAM-SB + instance loss | ResNet50 (1024-d) | 60 | 3 | **0.7035 ± 0.0088** | 0.3833 ± 0.0289 |
