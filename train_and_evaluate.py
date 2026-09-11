"""
Train logistic regression on the synthetic ArthroScreen data and check:
1. Does it actually learn anything (accuracy/precision/recall/AUC)?
2. Which features does it lean on (coefficients = "understand why")?
3. Subject-wise split sanity check vs. a (wrong) bend-wise split, to make
   the data-leakage point concrete with real numbers instead of just theory.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from generate_synthetic_data import generate_dataset, subject_wise_split, FEATURES

df = generate_dataset(n_healthy=30, n_oa=40, bends_per_person=(3, 6))


def fit_and_report(train_df, test_df, tag):
    X_train, y_train = train_df[FEATURES].values, train_df["label"].values
    X_test, y_test = test_df[FEATURES].values, test_df["label"].values

    scaler = StandardScaler().fit(X_train)
    X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_s, y_train)

    y_pred = clf.predict(X_test_s)
    y_prob = clf.predict_proba(X_test_s)[:, 1]

    print(f"\n=== {tag} ===")
    print(f"train rows={len(train_df)} ({train_df['person_id'].nunique()} people)  "
          f"test rows={len(test_df)} ({test_df['person_id'].nunique()} people)")
    print(f"accuracy={accuracy_score(y_test, y_pred):.3f}  "
          f"precision={precision_score(y_test, y_pred):.3f}  "
          f"recall={recall_score(y_test, y_pred):.3f}  "
          f"AUC={roc_auc_score(y_test, y_prob):.3f}")
    return clf, scaler


# --- correct: subject-wise split ---
train_df, test_df = subject_wise_split(df, test_frac=0.25)
clf, scaler = fit_and_report(train_df, test_df, "SUBJECT-WISE split (correct)")

print("\nCoefficients (standardized — bigger |value| = more the model leans on that feature):")
coefs = pd.Series(clf.coef_[0], index=FEATURES).sort_values(key=abs, ascending=False)
for feat, c in coefs.items():
    direction = "-> pushes toward OA" if c > 0 else "-> pushes toward Healthy"
    print(f"  {feat:20s} {c:+.3f}  {direction}")

# --- the leakage demo: bend-wise split on the SAME data ---
bendwise_train, bendwise_test = train_test_split(
    df, test_size=0.25, random_state=42, stratify=df["label"]
)
fit_and_report(bendwise_train, bendwise_test, "BEND-WISE split (the mistake — same person's bends leak across train/test)")

print("\nIf bend-wise AUC looks noticeably higher than subject-wise AUC above, that gap IS the "
      "leakage — the model is partly recognizing a person, not the OA signal.")
