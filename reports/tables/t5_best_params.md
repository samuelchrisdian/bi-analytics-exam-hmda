| model   | best_params                                                                                                        | waktu_tuning_detik   | n_data_tuning   |
|:--------|:-------------------------------------------------------------------------------------------------------------------|:---------------------|:----------------|
| dummy   | -                                                                                                                  | 0,1                  | 44.001          |
| logreg  | C=0,0060                                                                                                           | 14,4                 | 44.001          |
| rf      | max_depth=14, max_features=0,5000, min_samples_leaf=11, n_estimators=287                                           | 356,8                | 44.001          |
| xgb     | colsample_bytree=0,8737, learning_rate=0,0374, max_depth=6, min_child_weight=8, n_estimators=234, subsample=0,9637 | 58,2                 | 44.001          |
