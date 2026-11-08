from src import config


def test_random_state():
    assert config.RANDOM_STATE == 42


def test_features_exclude_forbidden():
    feats = set(config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES)
    forbidden = set(config.LEAKAGE_COLS + config.PROTECTED_COLS + config.PROXY_COLS + config.ID_COLS)
    assert feats.isdisjoint(forbidden)


def test_log_features_subset():
    assert set(config.LOG_FEATURES) <= set(config.NUMERIC_FEATURES)


def test_every_feature_has_label():
    assert set(config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES) <= set(config.FEATURE_LABELS)


def test_threshold_grid():
    assert config.THRESHOLD_GRID[0] == 0.01 and config.THRESHOLD_GRID[-1] == 0.99


def test_leakage_audit_drops_are_not_features():
    feats = set(config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES)
    assert set(config.LEAKAGE_AUDIT_DROPPED) == {
        "loan_to_value_ratio", "preapproval", "aus-1", "initially_payable_to_institution"}
    assert feats.isdisjoint(config.LEAKAGE_AUDIT_DROPPED)
    assert "ltv_calc" in config.NUMERIC_FEATURES
    assert set(config.LEAKAGE_AUDIT_DROPPED) <= set(config.RAW_USECOLS)
