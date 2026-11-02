# Paired comparisons (identical slides, identical seeds)

Paired bootstrap CI on ΔQWK (2000 resamples) and the exact McNemar test on per-slide correctness. A CI that spans 0 means *inconclusive at this sample size*, not *equivalent*.

| comparison | seed | n | QWK A | QWK B | ΔQWK | 95% CI | McNemar p | right/wrong |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| abmil → abmil_chief | 0 | 57 | 0.7019 | 0.9168 | +0.2149 | [+0.0965, +0.3718] | 0.0013 | 2/16 |
| abmil → abmil_chief | 1 | 57 | 0.7199 | 0.8880 | +0.1681 | [+0.0550, +0.3228] | 0.0225 | 2/11 |
| abmil → abmil_chief | 2 | 57 | 0.7295 | 0.9056 | +0.1762 | [+0.0592, +0.3277] | 0.0636 | 5/14 |
| abmil → abmil_gigap | 0 | 53 | 0.6999 | 0.8871 | +0.1872 | [+0.0467, +0.3618] | 0.0013 | 2/16 |
| abmil → abmil_gigap | 1 | 53 | 0.7151 | 0.9085 | +0.1934 | [+0.0763, +0.3497] | 0.0414 | 5/15 |
| abmil → abmil_gigap | 2 | 53 | 0.7251 | 0.8995 | +0.1744 | [+0.0539, +0.3449] | 0.0963 | 5/13 |
| abmil → clam_sb/with_instance | 0 | 60 | 0.7137 | 0.7137 | +0.0000 | [-0.1149, +0.0990] | 0.6250 | 1/3 |
| abmil → clam_sb/with_instance | 1 | 60 | 0.7308 | 0.6991 | -0.0317 | [-0.1062, +0.0280] | 0.6250 | 3/1 |
| abmil → clam_sb/with_instance | 2 | 60 | 0.7397 | 0.6978 | -0.0419 | [-0.1219, +0.0259] | 1.0000 | 2/1 |
| abmil → clam_sb/without_instance | 0 | 60 | 0.7137 | 0.7023 | -0.0114 | [-0.0753, +0.0387] | 0.6250 | 1/3 |
| abmil → clam_sb/without_instance | 1 | 60 | 0.7308 | 0.7336 | +0.0029 | [-0.0257, +0.0326] | 1.0000 | 2/1 |
| abmil → clam_sb/without_instance | 2 | 60 | 0.7397 | 0.7454 | +0.0057 | [-0.0099, +0.0255] | 1.0000 | 0/0 |
| abmil → transmil | 0 | 60 | 0.7137 | 0.7030 | -0.0106 | [-0.1488, +0.1048] | 0.0768 | 4/12 |
| abmil → transmil | 1 | 60 | 0.7308 | 0.7602 | +0.0294 | [-0.0632, +0.1218] | 0.0768 | 4/12 |
| abmil → transmil | 2 | 60 | 0.7397 | 0.7706 | +0.0309 | [-0.1076, +0.1752] | 1.0000 | 8/9 |
| abmil_chief → abmil_gigap | 0 | 50 | 0.9231 | 0.8775 | -0.0456 | [-0.1296, +0.0105] | 1.0000 | 7/7 |
| abmil_chief → abmil_gigap | 1 | 50 | 0.9012 | 0.9039 | +0.0027 | [-0.0566, +0.0627] | 1.0000 | 9/10 |
| abmil_chief → abmil_gigap | 2 | 50 | 0.9101 | 0.8908 | -0.0193 | [-0.0753, +0.0292] | 1.0000 | 7/6 |
| abmil_chief → clam_sb_chief | 0 | 57 | 0.9168 | 0.9274 | +0.0107 | [+0.0000, +0.0358] | 1.0000 | 0/1 |
| abmil_chief → clam_sb_chief | 1 | 57 | 0.8880 | 0.9199 | +0.0318 | [+0.0001, +0.0749] | 0.2188 | 1/5 |
| abmil_chief → clam_sb_chief | 2 | 57 | 0.9056 | 0.9148 | +0.0092 | [-0.0079, +0.0349] | 0.6250 | 1/3 |
| clam_sb/with_instance → clam_sb/without_instance | 0 | 60 | 0.7137 | 0.7023 | -0.0114 | [-0.1094, +0.0876] | 1.0000 | 4/4 |
| clam_sb/with_instance → clam_sb/without_instance | 1 | 60 | 0.6991 | 0.7336 | +0.0345 | [-0.0327, +0.1091] | 1.0000 | 2/3 |
| clam_sb/with_instance → clam_sb/without_instance | 2 | 60 | 0.6978 | 0.7454 | +0.0476 | [-0.0148, +0.1252] | 1.0000 | 1/2 |
| clam_sb/with_instance → transmil | 0 | 60 | 0.7137 | 0.7030 | -0.0106 | [-0.1804, +0.1537] | 0.2101 | 5/11 |
| clam_sb/with_instance → transmil | 1 | 60 | 0.6991 | 0.7602 | +0.0611 | [-0.0575, +0.1882] | 0.0309 | 4/14 |
| clam_sb/with_instance → transmil | 2 | 60 | 0.6978 | 0.7706 | +0.0728 | [-0.0839, +0.2322] | 0.8036 | 7/9 |
| clam_sb/without_instance → transmil | 0 | 60 | 0.7023 | 0.7030 | +0.0008 | [-0.1414, +0.1270] | 0.2379 | 6/12 |
| clam_sb/without_instance → transmil | 1 | 60 | 0.7336 | 0.7602 | +0.0266 | [-0.0679, +0.1190] | 0.0225 | 2/11 |
| clam_sb/without_instance → transmil | 2 | 60 | 0.7454 | 0.7706 | +0.0252 | [-0.1209, +0.1729] | 1.0000 | 8/9 |
| abmil → clam_sb_chief | 0 | 57 | 0.7019 | 0.9274 | +0.2255 | [+0.1044, +0.3853] | 0.0007 | 2/17 |
| abmil → clam_sb_chief | 1 | 57 | 0.7199 | 0.9199 | +0.2000 | [+0.0806, +0.3571] | 0.0044 | 3/16 |
| abmil → clam_sb_chief | 2 | 57 | 0.7295 | 0.9148 | +0.1854 | [+0.0681, +0.3401] | 0.0192 | 4/15 |
