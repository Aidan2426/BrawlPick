"""
Brawl Stars - Random Forest Model Training
Trains a win-prediction model on training_data_encoded.csv
Saves model + metadata for use by the recommender

Usage: python train_model.py
"""

import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import cross_val_score, train_test_split

DATA_DIR  = Path("data")
MODEL_DIR = Path("model")
MODEL_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────

print("Loading training data...")
df = pd.read_csv(DATA_DIR / "training_data_encoded.csv")
print(f"  {len(df):,} rows, {len(df.columns)} columns")

with open(DATA_DIR / "label_encodings.json") as f:
    label_encodings = json.load(f)

# Target and features
TARGET = "team_a_win"
DROP   = ["mode"]  # redundant with game_mode; drop to reduce noise

feature_cols = [c for c in df.columns if c != TARGET and c not in DROP]
X = df[feature_cols]
y = df[TARGET]

print(f"  Features: {len(feature_cols)}")
print(f"  Target balance: {y.mean():.1%} positive (team A win)")

# ─────────────────────────────────────────────
# Train / test split
# ─────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {len(X_train):,}  |  Test: {len(X_test):,}")

# ─────────────────────────────────────────────
# Train Random Forest
# ─────────────────────────────────────────────

print("\nTraining Random Forest...")

rf = RandomForestClassifier(
    n_estimators=300,      # 300 trees — good balance of accuracy vs speed
    max_depth=None,        # let trees grow fully
    min_samples_leaf=5,    # prevents overfitting on tiny leaf nodes
    max_features="sqrt",   # standard for classification
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,             # use all CPU cores
)

rf.fit(X_train, y_train)
print("  Done.")

# ─────────────────────────────────────────────
# Evaluate
# ─────────────────────────────────────────────

y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:, 1]

test_acc = accuracy_score(y_test, y_pred)
print(f"\n=== TEST SET RESULTS ===")
print(f"Accuracy:  {test_acc:.1%}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Team B wins", "Team A wins"]))

# 5-fold cross-validation on full dataset
print("Running 5-fold cross-validation (this takes ~30s)...")
cv_scores = cross_val_score(rf, X, y, cv=5, scoring="accuracy", n_jobs=-1)
print(f"CV Accuracy: {cv_scores.mean():.1%} ± {cv_scores.std():.1%}")
print(f"  Per fold: {[f'{s:.1%}' for s in cv_scores]}")

# ─────────────────────────────────────────────
# Feature importances
# ─────────────────────────────────────────────

importances = pd.Series(rf.feature_importances_, index=feature_cols)
importances = importances.sort_values(ascending=False)

print(f"\n=== TOP 15 FEATURE IMPORTANCES ===")
for feat, imp in importances.head(15).items():
    bar = "█" * int(imp * 300)
    print(f"  {feat:<35} {imp:.4f}  {bar}")

# Plot and save
fig, ax = plt.subplots(figsize=(10, 7))
importances.head(20).sort_values().plot(kind="barh", ax=ax, color="steelblue")
ax.set_title("Top 20 Feature Importances — Brawl Stars RF Model")
ax.set_xlabel("Importance")
plt.tight_layout()
fig.savefig(MODEL_DIR / "feature_importances.png", dpi=150)
plt.close()
print(f"\nSaved feature_importances.png → model/")

# Confusion matrix
fig, ax = plt.subplots(figsize=(5, 4))
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=["B wins", "A wins"]).plot(ax=ax)
ax.set_title("Confusion Matrix (test set)")
plt.tight_layout()
fig.savefig(MODEL_DIR / "confusion_matrix.png", dpi=150)
plt.close()
print("Saved confusion_matrix.png → model/")

# ─────────────────────────────────────────────
# Save model
# ─────────────────────────────────────────────

model_meta = {
    "feature_cols":     feature_cols,
    "target":           TARGET,
    "label_encodings":  label_encodings,
    "test_accuracy":    round(test_acc, 4),
    "cv_accuracy_mean": round(cv_scores.mean(), 4),
    "cv_accuracy_std":  round(cv_scores.std(), 4),
    "n_train":          len(X_train),
    "n_test":           len(X_test),
    "top_features":     importances.head(10).to_dict(),
}

with open(MODEL_DIR / "rf_model.pkl", "wb") as f:
    pickle.dump(rf, f)

with open(MODEL_DIR / "model_meta.json", "w") as f:
    json.dump(model_meta, f, indent=2)

print(f"\nSaved rf_model.pkl  → model/")
print(f"Saved model_meta.json → model/")

print(f"""
=== DONE ===
Model trained on {len(X_train):,} matches, tested on {len(X_test):,}.
Test accuracy:  {test_acc:.1%}
CV accuracy:    {cv_scores.mean():.1%} ± {cv_scores.std():.1%}

What the accuracy means:
  - 50% = coin flip (no better than random)
  - 55%+ = the model is learning real signal
  - 60%+ = strong — draft composition meaningfully predicts outcomes

Next step: run recommend.py to use the model as a brawler picker.
""")
