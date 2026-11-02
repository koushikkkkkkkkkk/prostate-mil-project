# Every run on the slides all runs share

Restricted to the 50 test slides covered by every run, so each row is measured on identical slides and identical labels. This is the only table that supports a like-for-like comparison.

| model | encoder | n slides | seeds | QWK (mean ± sd) | accuracy |
| --- | --- | --- | --- | --- | --- |
| CLAM-SB | CHIEF (768-d) | 50 | 3 | **0.9278 ± 0.0074** | 0.6067 ± 0.0115 |
| ABMIL | CHIEF (768-d) | 50 | 3 | **0.9115 ± 0.0110** | 0.5667 ± 0.0306 |
| ABMIL | GigaPath (1536-d) | 50 | 3 | **0.8907 ± 0.0132** | 0.5667 ± 0.0306 |
| TransMIL | ResNet50 (1024-d) | 50 | 3 | **0.7113 ± 0.0583** | 0.4733 ± 0.0577 |
| CLAM-SB, no instance loss | ResNet50 (1024-d) | 50 | 3 | **0.7001 ± 0.0313** | 0.3733 ± 0.0306 |
| ABMIL | ResNet50 (1024-d) | 50 | 3 | **0.6995 ± 0.0136** | 0.3667 ± 0.0306 |
| CLAM-SB + instance loss | ResNet50 (1024-d) | 50 | 3 | **0.6680 ± 0.0110** | 0.3600 ± 0.0346 |
