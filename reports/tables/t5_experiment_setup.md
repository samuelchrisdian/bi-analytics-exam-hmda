| aspek                            | keterangan                                                                                                                       |
|:---------------------------------|:---------------------------------------------------------------------------------------------------------------------------------|
| Data latih                       | 44.001 baris (2024, 80% stratified), 13 fitur input                                                                              |
| Proporsi kelas positif (ditolak) | 11,75% (rasio negatif:positif 7,51)                                                                                              |
| Validasi                         | StratifiedKFold 5 fold, shuffle, random_state 42                                                                                 |
| Pencarian hyperparameter         | RandomizedSearchCV, n_iter 25, refit pada PR-AUC (average_precision)                                                             |
| Metrik CV                        | average_precision, roc_auc, recall, precision, f1                                                                                |
| Subsample tuning                 | Tidak (data latih 44.001 ≤ ambang 200.000 baris)                                                                                 |
| Penanganan imbalance             | Utama: class weighting (balanced / scale_pos_weight); pembanding: SMOTE di dalam imblearn Pipeline                               |
| Preprocessing                    | Di dalam Pipeline: capping persentil 1–99, log1p (3 fitur nominal), imputasi median (tanpa missing indicator), StandardScaler, one-hot |
| Seed                             | RANDOM_STATE = 42 di semua komponen                                                                                              |
| Library                          | Python 3.11.18, scikit-learn 1.9.1, xgboost 3.2.0, imbalanced-learn 0.14.2, pandas 2.3.3                                         |
| Perangkat keras                  | 13th Gen Intel(R) Core(TM) i7-13620H, 16 thread, RAM 15,3 GB; n_jobs = 4                                                         |
| Test & OOT                       | Tidak dipakai di tahap ini (disimpan untuk Tahap 6–7)                                                                            |
