# train_model.py - Optimized for current data structure
# ADX is correctly dominant at 67% - this is a good sign
# Focus: improve NO_TRADE accuracy from 57% to 65%+

import pandas as pd
import numpy as np
import joblib
import json
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import SelectKBest, f_classif
from config import (
    DATASET_FILE, TRAIN_FILE, TEST_FILE,
    MODEL_FILE, FEATURES, RANDOM_STATE
)

try:
    from xgboost import XGBClassifier
    XGB = True
except ImportError:
    XGB = False
    print("XGBoost not found - using GradientBoosting")


def balance_dataframe(df):
    buy     = df[df["target"] == "BUY"]
    sell    = df[df["target"] == "SELL"]
    notrade = df[df["target"] == "NO_TRADE"]

    print(f"  Before: BUY={len(buy)} SELL={len(sell)} NO_TRADE={len(notrade)}")

    # Use median size to keep more data
    target_size = int(np.median([len(buy), len(sell), len(notrade)]))
    target_size = max(target_size, 2000)

    def sample_class(data, size):
        if len(data) >= size:
            return data.sample(size, random_state=RANDOM_STATE)
        return data.sample(size, random_state=RANDOM_STATE, replace=True)

    balanced = pd.concat([
        sample_class(buy,     target_size),
        sample_class(sell,    target_size),
        sample_class(notrade, target_size),
    ]).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    print(f"  After:  {len(balanced):,} rows ({target_size} per class)")
    return balanced


def walk_forward_validate(df, n_folds=4):
    print(f"\n  Walk-forward validation ({n_folds} folds)...")
    fold_size = len(df) // (n_folds + 1)
    scores    = []

    for i in range(1, n_folds + 1):
        train_end = fold_size * i
        test_end  = train_end + fold_size
        train     = df.iloc[:train_end]
        test      = df.iloc[train_end:test_end]

        if len(test) < 100:
            continue

        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        rf.fit(train[FEATURES], train["target_num"])
        acc = accuracy_score(test["target_num"], rf.predict(test[FEATURES]))
        scores.append(acc)
        print(f"    Fold {i}: {acc*100:.1f}%")

    mean = np.mean(scores)
    std  = np.std(scores)
    print(f"  CV Mean: {mean*100:.1f}% +/- {std*100:.1f}%")
    return mean


def main():
    print("Training AI Model\n")

    df = pd.read_csv(DATASET_FILE)
    print(f"  Loaded {len(df):,} rows")

    # Check features
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        print(f"\n  ERROR - Missing: {missing}")
        print("  Run feature_engineering.py then create_targets.py first!")
        return

    df = df.dropna(subset=FEATURES + ["target"])
    print(f"  After dropna: {len(df):,} rows")

    # Show distribution
    dist = df["target"].value_counts()
    total = len(df)
    print("\n  Target distribution:")
    for label, count in dist.items():
        print(f"    {label:10s}: {count:6d} ({count/total*100:.1f}%)")

    label_map        = {"BUY": 0, "SELL": 1, "NO_TRADE": 2}
    df["target_num"] = df["target"].map(label_map)

    if "open_time" in df.columns:
        df = df.sort_values("open_time").reset_index(drop=True)

    print("\n  Balancing classes...")
    df = balance_dataframe(df)
    df["target_num"] = df["target"].map(label_map)

    cv_score = walk_forward_validate(df)

    split = int(len(df) * 0.70)
    train = df.iloc[:split].copy()
    test  = df.iloc[split:].copy()

    train.to_csv(TRAIN_FILE, index=False)
    test.to_csv(TEST_FILE,   index=False)
    print(f"\n  Train: {len(train):,}  |  Test: {len(test):,}")
    print(f"  Features: {len(FEATURES)}")

    X_train = train[FEATURES]
    y_train = train["target_num"]
    X_test  = test[FEATURES]
    y_test  = test["target_num"]

    # Feature selection
    print("\n  Selecting best features...")
    k        = min(18, len(FEATURES))
    selector = SelectKBest(f_classif, k=k)
    X_tr_sel = selector.fit_transform(X_train, y_train)
    X_te_sel = selector.transform(X_test)
    selected = [FEATURES[i] for i in selector.get_support(indices=True)]
    print(f"  Using {len(selected)} features:")
    for f in selected:
        print(f"    - {f}")

    # XGBoost - tuned for better NO_TRADE detection
    print("\n  Building XGBoost...")
    if XGB:
        xgb = XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.02,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            gamma=0.2,
            reg_alpha=0.1,
            reg_lambda=1.0,
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    else:
        xgb = GradientBoostingClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.02,
            subsample=0.8,
            random_state=RANDOM_STATE,
        )

    # RandomForest with balanced class weights
    print("  Building RandomForest...")
    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=4,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    # GradientBoosting for better NO_TRADE
    print("  Building GradientBoosting...")
    gb = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        random_state=RANDOM_STATE,
    )

    print("\n  Training triple ensemble...")
    print("  This takes 10-15 minutes. Do NOT press Ctrl+C\n")

    ensemble = VotingClassifier(
        estimators=[("xgb", xgb), ("rf", rf), ("gb", gb)],
        voting="soft",
        weights=[3, 2, 1],
    )
    ensemble.fit(X_tr_sel, y_train)

    preds    = ensemble.predict(X_te_sel)
    accuracy = accuracy_score(y_test, preds)

    print(f"\n{'='*50}")
    print(f"  FINAL ACCURACY: {accuracy*100:.1f}%")
    print(f"  CV ACCURACY:    {cv_score*100:.1f}%")
    print(f"{'='*50}")

    print(classification_report(
        y_test, preds,
        target_names=["BUY", "SELL", "NO_TRADE"]
    ))

    print("  Per class accuracy:")
    for label, num in [("BUY", 0), ("SELL", 1), ("NO_TRADE", 2)]:
        mask = y_test == num
        if mask.sum() > 0:
            cacc = accuracy_score(y_test[mask], preds[mask])
            print(f"    {label:10s}: {cacc*100:.1f}%")

    # Feature importance
    rf_model = ensemble.estimators_[1]
    imp = pd.Series(
        rf_model.feature_importances_,
        index=selected
    ).sort_values(ascending=False)

    print("\n  Top 10 most important features:")
    for feat, val in imp.head(10).items():
        bar = "█" * int(val * 100)
        print(f"    {feat:25s} {val:.4f} {bar}")

    # Save pipeline
    pipeline = {
        "ensemble":      ensemble,
        "selector":      selector,
        "best_features": selected,
        "all_features":  FEATURES,
        "label_map":     {"BUY": 0, "SELL": 1, "NO_TRADE": 2},
    }
    joblib.dump(pipeline, MODEL_FILE)
    print(f"\n  Model saved to {MODEL_FILE}")

    with open("model_performance.json", "w") as f:
        json.dump({
            "test_accuracy": round(accuracy, 4),
            "cv_accuracy":   round(cv_score, 4),
            "train_rows":    len(train),
            "test_rows":     len(test),
            "features":      selected,
        }, f, indent=2)

    print(f"\n{'='*50}")
    if accuracy >= 0.70:
        print(f"  EXCELLENT: {accuracy*100:.1f}% - Ready for paper trading!")
        print("  Keep MIN_CONFIDENCE = 65")
    elif accuracy >= 0.65:
        print(f"  GOOD: {accuracy*100:.1f}%")
        print("  Set MIN_CONFIDENCE = 62")
    elif accuracy >= 0.60:
        print(f"  OK: {accuracy*100:.1f}%")
        print("  Set MIN_CONFIDENCE = 60")
    else:
        print(f"  LOW: {accuracy*100:.1f}%")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()