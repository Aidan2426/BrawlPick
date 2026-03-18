"""
Brawl Stars - Random Forest Model Training
Trains a win-prediction model on training_data.csv

Encoding strategy:
  - Brawler slots: MAP-SPECIFIC TARGET ENCODING
    Each brawler is encoded as their win rate on the specific map being played.
    Falls back to global win rate when fewer than MIN_MAP_APPEARANCES games exist
    for that (map, brawler) pair on the training set.
  - map, game_mode: label encoding
  - All other features dropped (class, rarity, gadgets, etc.)

Usage: python train_model.py
"""

import json
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
from sklearn.preprocessing import LabelEncoder

DATA_DIR  = Path("data")
MODEL_DIR = Path("model")
MODEL_DIR.mkdir(exist_ok=True)

BRAWLER_SLOTS = [
    "team_a_brawler1", "team_a_brawler2", "team_a_brawler3",
    "team_b_brawler1", "team_b_brawler2", "team_b_brawler3",
]

TARGET = "team_a_win"
DROP   = [
    "mode",                              # redundant with game_mode
    "battle_time",                       # timestamp — spurious correlation
    "environment",                       # visual theme — redundant given map
    # Class / rarity / gadgets / star powers — not considered
    "team_a_brawler1_class",   "team_a_brawler2_class",   "team_a_brawler3_class",
    "team_b_brawler1_class",   "team_b_brawler2_class",   "team_b_brawler3_class",
    "team_a_brawler1_rarity",  "team_a_brawler2_rarity",  "team_a_brawler3_rarity",
    "team_b_brawler1_rarity",  "team_b_brawler2_rarity",  "team_b_brawler3_rarity",
    "team_a_brawler1_star_powers_count", "team_a_brawler2_star_powers_count", "team_a_brawler3_star_powers_count",
    "team_b_brawler1_star_powers_count", "team_b_brawler2_star_powers_count", "team_b_brawler3_star_powers_count",
    "team_a_brawler1_gadgets_count",     "team_a_brawler2_gadgets_count",     "team_a_brawler3_gadgets_count",
    "team_b_brawler1_gadgets_count",     "team_b_brawler2_gadgets_count",     "team_b_brawler3_gadgets_count",
]

# ─────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────

print("Loading training data...")
df = pd.read_csv(DATA_DIR / "training_data.csv")
print(f"  {len(df):,} rows, {len(df.columns)} columns")

# Cap per-map rows so no single map dominates training
MAX_PER_MAP = 500
before = len(df)
df = df.groupby("map", group_keys=False).apply(lambda g: g.sample(min(len(g), MAX_PER_MAP), random_state=42)).reset_index(drop=True)
print(f"  After per-map cap ({MAX_PER_MAP}): {len(df):,} rows  (removed {before - len(df):,})")

# Uppercase brawler columns (data should already be uppercase, but be safe)
for col in BRAWLER_SLOTS:
    if col in df.columns:
        df[col] = df[col].str.upper().fillna("UNKNOWN")

df = df.dropna(subset=[TARGET])
y = df[TARGET].astype(int)

feature_cols_raw = [c for c in df.columns if c != TARGET and c not in DROP and c in df.columns]
X_raw = df[feature_cols_raw].copy()

print(f"  Target balance: {y.mean():.1%} team A win")

# ─────────────────────────────────────────────
# Train / test split BEFORE encoding (prevents target leakage)
# ─────────────────────────────────────────────

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {len(X_train_raw):,}  |  Test: {len(X_test_raw):,}")

# ─────────────────────────────────────────────
# Target encoding for brawler slots
# Computed on training set ONLY to prevent leakage.
#
# For each brawler, compute: "when this brawler plays, what % does their team win?"
# - team_a slots: win = team_a_win
# - team_b slots: win = team_b_win = 1 - team_a_win
# ─────────────────────────────────────────────

MIN_MAP_APPEARANCES = 15  # min (map, brawler) games to use map-specific rate

print("\nComputing brawler target encodings (training set only)...")

# ── Global win rates (fallback) ──────────────────────────────────────────────
appearances = []
for col in BRAWLER_SLOTS:
    is_team_a = "team_a" in col
    tmp = pd.DataFrame({
        "brawler": X_train_raw[col].values,
        "win":     y_train.values if is_team_a else (1 - y_train.values),
    })
    appearances.append(tmp)

combined = pd.concat(appearances, ignore_index=True)
combined = combined[
    combined["brawler"].notna() &
    (combined["brawler"] != "") &
    (combined["brawler"] != "UNKNOWN")
]

brawler_win_rates = combined.groupby("brawler")["win"].mean().to_dict()
global_mean = float(combined["win"].mean())

print(f"  Global: {len(brawler_win_rates)} brawlers, mean win rate {global_mean:.3f}")

# ── Map-specific win rates ────────────────────────────────────────────────────
map_appearances = []
for col in BRAWLER_SLOTS:
    is_team_a = "team_a" in col
    tmp = pd.DataFrame({
        "map":     X_train_raw["map"].values,
        "brawler": X_train_raw[col].values,
        "win":     y_train.values if is_team_a else (1 - y_train.values),
    })
    map_appearances.append(tmp)

map_combined = pd.concat(map_appearances, ignore_index=True)
map_combined = map_combined[
    map_combined["brawler"].notna() &
    (map_combined["brawler"] != "") &
    (map_combined["brawler"] != "UNKNOWN") &
    map_combined["map"].notna()
]

map_counts = map_combined.groupby(["map", "brawler"])["win"].count()
map_rates  = map_combined.groupby(["map", "brawler"])["win"].mean()
valid      = map_counts[map_counts >= MIN_MAP_APPEARANCES].index
map_rates  = map_rates[map_rates.index.isin(valid)]

# Store as "MAP|BRAWLER" → win_rate for easy JSON serialisation
brawler_map_win_rates  = {f"{m.upper()}|{b.upper()}": float(wr)  for (m, b), wr  in map_rates.items()}
brawler_map_game_counts = {f"{m.upper()}|{b.upper()}": int(cnt) for (m, b), cnt in map_counts[map_counts >= MIN_MAP_APPEARANCES].items()}

print(f"  Map-specific: {len(brawler_map_win_rates)} (map, brawler) pairs "
      f"with ≥{MIN_MAP_APPEARANCES} games")


def apply_brawler_target_encoding(X_raw, brawler_win_rates, brawler_map_win_rates, global_mean):
    """Encode each brawler slot as their map-specific win rate, falling back to global."""
    X = X_raw.copy()
    map_col = X["map"].astype(str) if "map" in X.columns else pd.Series([""] * len(X))
    for col in BRAWLER_SLOTS:
        if col in X.columns:
            keys        = map_col + "|" + X[col].astype(str)
            map_encoded = keys.map(brawler_map_win_rates)
            global_enc  = X[col].map(brawler_win_rates).fillna(global_mean)
            X[col]      = map_encoded.where(map_encoded.notna(), global_enc)
    return X


X_train = apply_brawler_target_encoding(X_train_raw, brawler_win_rates, brawler_map_win_rates, global_mean)
X_test  = apply_brawler_target_encoding(X_test_raw,  brawler_win_rates, brawler_map_win_rates, global_mean)

# ─────────────────────────────────────────────
# Label encoding for remaining categorical columns
# ─────────────────────────────────────────────

label_encodings = {}

for col in X_train.select_dtypes(include="object").columns:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col].astype(str).fillna("Unknown"))
    # Map test set values, unknown → 0
    mapping = {cls: i for i, cls in enumerate(le.classes_)}
    X_test[col] = X_test[col].astype(str).fillna("Unknown").map(mapping).fillna(0).astype(int)
    label_encodings[col] = list(le.classes_)

feature_cols = list(X_train.columns)
print(f"\n  Final feature count: {len(feature_cols)}")

# ─────────────────────────────────────────────
# Train Random Forest
# ─────────────────────────────────────────────

print("\nTraining Random Forest...")

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=5,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

rf.fit(X_train, y_train)
print("  Done.")

# ─────────────────────────────────────────────
# Evaluate
# ─────────────────────────────────────────────

y_pred = rf.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)

print(f"\n=== TEST SET RESULTS ===")
print(f"Accuracy:  {test_acc:.1%}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Team B wins", "Team A wins"]))

print("Running 5-fold cross-validation...")
cv_scores = cross_val_score(rf, pd.concat([X_train, X_test]), pd.concat([y_train, y_test]),
                             cv=5, scoring="accuracy", n_jobs=-1)
print(f"CV Accuracy: {cv_scores.mean():.1%} ± {cv_scores.std():.1%}")
print(f"  Per fold: {[f'{s:.1%}' for s in cv_scores]}")

# ─────────────────────────────────────────────
# Feature importances
# ─────────────────────────────────────────────

importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)

print(f"\n=== ALL FEATURE IMPORTANCES ===")
for feat, imp in importances.items():
    bar = "█" * int(imp * 300)
    print(f"  {feat:<35} {imp:.4f}  {bar}")

fig, ax = plt.subplots(figsize=(10, 7))
importances.head(20).sort_values().plot(kind="barh", ax=ax, color="steelblue")
ax.set_title("Top 20 Feature Importances — Brawl Stars RF Model")
ax.set_xlabel("Importance")
plt.tight_layout()
fig.savefig(MODEL_DIR / "feature_importances.png", dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(5, 4))
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=["B wins", "A wins"]).plot(ax=ax)
ax.set_title("Confusion Matrix (test set)")
plt.tight_layout()
fig.savefig(MODEL_DIR / "confusion_matrix.png", dpi=150)
plt.close()

# ─────────────────────────────────────────────
# Save model + metadata
# ─────────────────────────────────────────────

model_meta = {
    "feature_cols":           feature_cols,
    "target":                 TARGET,
    "brawler_win_rates":      brawler_win_rates,       # global fallback
    "brawler_map_win_rates":   brawler_map_win_rates,    # map-specific (primary)
    "brawler_map_game_counts": brawler_map_game_counts, # games behind each map rate
    "brawler_global_mean":    global_mean,
    "label_encodings":        label_encodings,      # for non-brawler categoricals
    "test_accuracy":          round(test_acc, 4),
    "cv_accuracy_mean":       round(float(cv_scores.mean()), 4),
    "cv_accuracy_std":        round(float(cv_scores.std()), 4),
    "n_train":                len(X_train),
    "n_test":                 len(X_test),
    "feature_importances":    importances.to_dict(),
}

import joblib
joblib.dump(rf, MODEL_DIR / "rf_model.pkl", compress=3)

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
""")
