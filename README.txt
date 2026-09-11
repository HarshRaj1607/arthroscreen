ArthroScreen synthetic data — how to run
==========================================

Requirements: python3 with numpy, pandas, scikit-learn
  pip install numpy pandas scikit-learn

Files:
  generate_synthetic_data.py   - the generator (also regenerates the 3 CSVs below if you run it directly)
  train_and_evaluate.py        - trains logistic regression, prints accuracy/precision/recall/AUC + coefficients,
                                  and runs the subject-wise vs bend-wise split comparison
  synthetic_arthroscreen_data.csv  - full dataset (all people, all bends)
  synthetic_train.csv          - subject-wise train split (80%)
  synthetic_test.csv           - subject-wise test split (20%)

To just re-run the experiment as-is:
  python3 train_and_evaluate.py

To regenerate the raw data / CSVs (same seed = same numbers, change params inside the file to try others):
  python3 generate_synthetic_data.py

Known issue (see chat): with 9 correlated features and ~70 people, the model currently
hits ~1.000 accuracy/AUC on both the correct (subject-wise) and deliberately-wrong (bend-wise)
split — meaning the leakage comparison doesn't show a gap yet. See generate_synthetic_data.py's
FEATURE_SPECS dict for per-feature literature grounding notes, and the chat discussion for the
two options (add a person-level confound vs. treat this as pipeline-smoke-test-only) to fix it.
