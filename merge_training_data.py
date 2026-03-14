"""
Brawl Stars Data Pipeline - Step 5
Merges battle_logs + brawler metadata + map metadata → training_data.csv

Two outputs:
  training_data.csv         — human-readable, one row per match
  training_data_encoded.csv — ML-ready, label-encoded, no string columns
"""

import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import json

RAW = Path("data/raw")
OUT = Path("data")

# ─────────────────────────────────────────────
# Load raw files
# ─────────────────────────────────────────────

print("Loading raw data...")
battles  = pd.read_csv(RAW / "battle_logs.csv")
brawlers = pd.read_csv(RAW / "brawlers.csv")
maps     = pd.read_csv(RAW / "maps_active.csv")

print(f"  battles:  {len(battles):,} rows")
print(f"  brawlers: {len(brawlers):,} rows")
print(f"  maps:     {len(maps):,} rows")

# ─────────────────────────────────────────────
# Normalize brawler names
# Battle logs → uppercase (SHELLY)
# brawlers.csv → proper case (Shelly)
# Normalize both to UPPERCASE for joining
# ─────────────────────────────────────────────

brawlers["brawler_name_upper"] = brawlers["brawler_name"].str.upper()

# Build lookup: BRAWLER_NAME → {class, rarity, star_powers_count, gadgets_count}
# Drop duplicates on the upper name key before indexing
brawler_lookup = (
    brawlers
    .drop_duplicates(subset="brawler_name_upper", keep="first")
    .set_index("brawler_name_upper")[["class", "rarity", "star_powers_count", "gadgets_count"]]
    .to_dict(orient="index")
)

# ─────────────────────────────────────────────
# Normalize map names for joining
# ─────────────────────────────────────────────

maps["map_name_upper"] = maps["map_name"].str.upper()
map_lookup = (
    maps
    .drop_duplicates(subset="map_name_upper", keep="first")
    .set_index("map_name_upper")[["game_mode_name", "environment_name"]]
    .to_dict(orient="index")
)

battles["map_upper"] = battles["map"].str.upper()

# ─────────────────────────────────────────────
# Drop incomplete / malformed rows
# ─────────────────────────────────────────────

brawler_cols = [
    "team_a_brawler1", "team_a_brawler2", "team_a_brawler3",
    "team_b_brawler1", "team_b_brawler2", "team_b_brawler3",
]

before = len(battles)

# Remove rows with empty brawler slots
battles = battles[
    battles[brawler_cols].apply(lambda row: row.str.strip().ne("").all(), axis=1)
]

# Remove rows where team_a_win is unknown
battles = battles[battles["team_a_win"].notna()]
battles["team_a_win"] = battles["team_a_win"].astype(bool)

after = len(battles)
print(f"\nDropped {before - after:,} incomplete rows → {after:,} clean matches")

# ─────────────────────────────────────────────
# Join map metadata
# ─────────────────────────────────────────────

battles["game_mode"] = battles["map_upper"].map(
    lambda m: map_lookup.get(m, {}).get("game_mode_name", "Unknown")
)
battles["environment"] = battles["map_upper"].map(
    lambda m: map_lookup.get(m, {}).get("environment_name", "Unknown")
)

unmapped_maps = battles[battles["game_mode"] == "Unknown"]["map"].unique()
if len(unmapped_maps):
    print(f"\n  Maps not found in maps_active.csv ({len(unmapped_maps)}):")
    for m in unmapped_maps[:10]:
        print(f"    - {m}")

# ─────────────────────────────────────────────
# Join brawler metadata for each slot
# ─────────────────────────────────────────────

def add_brawler_meta(df, col):
    """Add class and rarity columns for a given brawler slot."""
    df[f"{col}_class"]  = df[col].map(lambda b: brawler_lookup.get(b, {}).get("class", "Unknown"))
    df[f"{col}_rarity"] = df[col].map(lambda b: brawler_lookup.get(b, {}).get("rarity", "Unknown"))
    return df

for col in brawler_cols:
    battles = add_brawler_meta(battles, col)

# Check coverage
total_brawler_cells = len(battles) * len(brawler_cols)
unknown_count = sum(
    (battles[f"{col}_class"] == "Unknown").sum() for col in brawler_cols
)
print(f"\nBrawler metadata coverage: {100*(1 - unknown_count/total_brawler_cells):.1f}%")
if unknown_count:
    # Show which brawlers are missing from the lookup
    missing = set()
    for col in brawler_cols:
        missing |= set(battles.loc[battles[f"{col}_class"] == "Unknown", col].unique())
    print(f"  Brawlers missing from brawlers.csv: {sorted(b for b in missing if isinstance(b, str))}")

# ─────────────────────────────────────────────
# Clean up columns for final output
# ─────────────────────────────────────────────

drop_cols = ["map_upper", "result_for_tag", "team_a_tags", "team_b_tags"]
battles = battles.drop(columns=drop_cols, errors="ignore")

# Reorder columns nicely
col_order = (
    ["battle_time", "map", "game_mode", "environment", "mode", "team_a_win"]
    + brawler_cols
    + [f"{c}_class" for c in brawler_cols]
    + [f"{c}_rarity" for c in brawler_cols]
)
col_order = [c for c in col_order if c in battles.columns]
remaining = [c for c in battles.columns if c not in col_order]
battles = battles[col_order + remaining]

out_path = OUT / "training_data.csv"
battles.to_csv(out_path, index=False)
print(f"\nSaved training_data.csv → {out_path}  ({len(battles):,} rows)")

# ─────────────────────────────────────────────
# Encoded version for sklearn
# Label-encode every string column
# ─────────────────────────────────────────────

print("\nBuilding label-encoded version...")

encoded = battles.copy()

# Drop columns we don't want as features
encoded = encoded.drop(columns=["battle_time"], errors="ignore")

# Convert team_a_win bool → int
encoded["team_a_win"] = encoded["team_a_win"].astype(int)

# Label-encode all object (string) columns
label_encoders = {}
string_cols = encoded.select_dtypes(include="object").columns.tolist()

for col in string_cols:
    le = LabelEncoder()
    encoded[col] = le.fit_transform(encoded[col].astype(str))
    label_encoders[col] = list(le.classes_)  # save mapping for later

# Save label encoder mappings (needed to decode predictions)
mappings_path = OUT / "label_encodings.json"
with open(mappings_path, "w") as f:
    json.dump(label_encoders, f, indent=2)

encoded_path = OUT / "training_data_encoded.csv"
encoded.to_csv(encoded_path, index=False)

print(f"Saved training_data_encoded.csv → {encoded_path}  ({len(encoded):,} rows, {len(encoded.columns)} features)")
print(f"Saved label_encodings.json → {mappings_path}  (needed to decode predictions)")

# ─────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────

print("\n=== SUMMARY ===")
print(f"Total matches:     {len(battles):,}")
print(f"Team A win rate:   {battles['team_a_win'].mean():.1%}  (should be ~50%)")
print(f"Unique maps:       {battles['map'].nunique()}")
print(f"Unique brawlers:   {pd.concat([battles[c] for c in brawler_cols]).nunique()}")
print(f"Game modes:        {sorted(battles['game_mode'].unique().tolist())}")
print(f"\nFeature columns:   {len(encoded.columns) - 1}  (excluding target team_a_win)")
print(f"Target column:     team_a_win  (1 = team A won, 0 = team B won)")
print("\nNext step: run train_model.py to train the Random Forest")
