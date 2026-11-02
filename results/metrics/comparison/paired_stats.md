# Paired comparisons (identical slides, identical seeds)

Paired bootstrap CI on ΔQWK (2000 resamples) and the exact McNemar test on per-slide correctness. A CI that spans 0 means *inconclusive at this sample size*, not *equivalent*.

| comparison | seed | n | QWK A | QWK B | ΔQWK | 95% CI | McNemar p | right/wrong |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| abmil → abmil_chief | 0 | 57 | 0.7019 | 0.9163 | +0.2144 | [+0.0949, +0.3747] | 0.0042 | 2/14 |
| abmil → abmil_chief | 1 | 57 | 0.7199 | 0.9119 | +0.1920 | [+0.0746, +0.3521] | 0.0042 | 2/14 |
| abmil → abmil_chief | 2 | 57 | 0.7295 | 0.9092 | +0.1797 | [+0.0635, +0.3278] | 0.0192 | 4/15 |
| abmil → abmil_gigap | 0 | 53 | 0.6999 | 0.9364 | +0.2365 | [+0.1086, +0.4117] | 0.0005 | 3/20 |
| abmil → abmil_gigap | 1 | 53 | 0.7151 | 0.9047 | +0.1896 | [+0.0756, +0.3469] | 0.0266 | 5/16 |
| abmil → abmil_gigap | 2 | 53 | 0.7251 | 0.9104 | +0.1852 | [+0.0648, +0.3561] | 0.0118 | 4/16 |
| abmil → clam_sb/with_instance | 0 | 60 | 0.7137 | 0.7137 | +0.0000 | [-0.1149, +0.0990] | 0.6250 | 1/3 |
| abmil → clam_sb/with_instance | 1 | 60 | 0.7308 | 0.6991 | -0.0317 | [-0.1062, +0.0280] | 0.6250 | 3/1 |
| abmil → clam_sb/with_instance | 2 | 60 | 0.7397 | 0.6978 | -0.0419 | [-0.1219, +0.0259] | 1.0000 | 2/1 |
| abmil → clam_sb/without_instance | 0 | 60 | 0.7137 | 0.7023 | -0.0114 | [-0.0753, +0.0387] | 0.6250 | 1/3 |
| abmil → clam_sb/without_instance | 1 | 60 | 0.7308 | 0.7336 | +0.0029 | [-0.0257, +0.0326] | 1.0000 | 2/1 |
| abmil → clam_sb/without_instance | 2 | 60 | 0.7397 | 0.7454 | +0.0057 | [-0.0099, +0.0255] | 1.0000 | 0/0 |
| abmil → transmil | 0 | 60 | 0.7137 | 0.7030 | -0.0106 | [-0.1488, +0.1048] | 0.0768 | 4/12 |
| abmil → transmil | 1 | 60 | 0.7308 | 0.7602 | +0.0294 | [-0.0632, +0.1218] | 0.0768 | 4/12 |
| abmil → transmil | 2 | 60 | 0.7397 | 0.7706 | +0.0309 | [-0.1076, +0.1752] | 1.0000 | 8/9 |
| abmil_chief → abmil_gigap | 0 | 50 | 0.9227 | 0.9308 | +0.0081 | [-0.0357, +0.0483] | 0.3018 | 5/10 |
| abmil_chief → abmil_gigap | 1 | 50 | 0.9175 | 0.8966 | -0.0209 | [-0.0696, +0.0211] | 1.0000 | 9/8 |
| abmil_chief → abmil_gigap | 2 | 50 | 0.9143 | 0.9024 | -0.0118 | [-0.0720, +0.0413] | 1.0000 | 6/7 |
| abmil_chief → clam_sb_chief | 0 | 57 | 0.9163 | 0.9236 | +0.0073 | [-0.0060, +0.0230] | 0.6875 | 2/4 |
| abmil_chief → clam_sb_chief | 1 | 57 | 0.9119 | 0.9227 | +0.0109 | [-0.0074, +0.0384] | 1.0000 | 2/3 |
| abmil_chief → clam_sb_chief | 2 | 57 | 0.9092 | 0.9209 | +0.0117 | [-0.0150, +0.0436] | 1.0000 | 3/3 |
| clam_sb/with_instance → clam_sb/without_instance | 0 | 60 | 0.7137 | 0.7023 | -0.0114 | [-0.1094, +0.0876] | 1.0000 | 4/4 |
| clam_sb/with_instance → clam_sb/without_instance | 1 | 60 | 0.6991 | 0.7336 | +0.0345 | [-0.0327, +0.1091] | 1.0000 | 2/3 |
| clam_sb/with_instance → clam_sb/without_instance | 2 | 60 | 0.6978 | 0.7454 | +0.0476 | [-0.0148, +0.1252] | 1.0000 | 1/2 |
| clam_sb/with_instance → transmil | 0 | 60 | 0.7137 | 0.7030 | -0.0106 | [-0.1804, +0.1537] | 0.2101 | 5/11 |
| clam_sb/with_instance → transmil | 1 | 60 | 0.6991 | 0.7602 | +0.0611 | [-0.0575, +0.1882] | 0.0309 | 4/14 |
| clam_sb/with_instance → transmil | 2 | 60 | 0.6978 | 0.7706 | +0.0728 | [-0.0839, +0.2322] | 0.8036 | 7/9 |
| clam_sb/without_instance → transmil | 0 | 60 | 0.7023 | 0.7030 | +0.0008 | [-0.1414, +0.1270] | 0.2379 | 6/12 |
| clam_sb/without_instance → transmil | 1 | 60 | 0.7336 | 0.7602 | +0.0266 | [-0.0679, +0.1190] | 0.0225 | 2/11 |
| clam_sb/without_instance → transmil | 2 | 60 | 0.7454 | 0.7706 | +0.0252 | [-0.1209, +0.1729] | 1.0000 | 8/9 |
| abmil → clam_sb_chief | 0 | 57 | 0.7019 | 0.9236 | +0.2217 | [+0.1036, +0.3783] | 0.0005 | 1/15 |
| abmil → clam_sb_chief | 1 | 57 | 0.7199 | 0.9227 | +0.2028 | [+0.0873, +0.3583] | 0.0023 | 2/15 |
| abmil → clam_sb_chief | 2 | 57 | 0.7295 | 0.9209 | +0.1914 | [+0.0709, +0.3452] | 0.0192 | 4/15 |
