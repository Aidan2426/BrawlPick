"""
Brawl Stars — Brawler Recommender
Given the current draft state, recommends the best brawler to pick next.

Usage (interactive):
    python recommend.py

Usage (CLI args):
    python recommend.py --map "Singed Earth" --my-team SHELLY RICO --enemy-team BIBI --top 10
"""

import argparse
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

MODEL_DIR = Path("model")
DATA_DIR  = Path("data")

# ─────────────────────────────────────────────
# Load model + metadata
# ─────────────────────────────────────────────

with open(MODEL_DIR / "rf_model.pkl", "rb") as f:
    model = pickle.load(f)

with open(MODEL_DIR / "model_meta.json") as f:
    meta = json.load(f)

feature_cols    = meta["feature_cols"]
label_encodings = meta["label_encodings"]  # col → list of classes (index = encoded int)

# Reverse mappings: col → {name: encoded_int}
encoders = {
    col: {name: i for i, name in enumerate(classes)}
    for col, classes in label_encodings.items()
}

# All known brawlers (union of all 6 brawler slot encodings)
brawler_slot_cols = [
    "team_a_brawler1", "team_a_brawler2", "team_a_brawler3",
    "team_b_brawler1", "team_b_brawler2", "team_b_brawler3",
]
all_brawlers = sorted(set(
    name
    for col in brawler_slot_cols
    for name in label_encodings.get(col, [])
    if name not in ("", "nan")
))


# ─────────────────────────────────────────────
# Encoding helpers
# ─────────────────────────────────────────────

def encode_value(col: str, value: str) -> int:
    """Encode a string value for a given column. Returns 0 if unknown."""
    val_upper = str(value).upper()
    mapping = encoders.get(col, {})
    # Try exact match first, then case-insensitive
    if val_upper in mapping:
        return mapping[val_upper]
    for k, v in mapping.items():
        if k.upper() == val_upper:
            return v
    return 0  # fallback: encode as first class


def get_brawler_class_rarity(brawler: str) -> tuple[int, int]:
    """Return (class_encoded, rarity_encoded) for a brawler name."""
    # Look up from training data brawler → class mapping
    # We'll use brawler1 slot as the reference encoder
    cls    = encode_value("team_a_brawler1_class", "Unknown")
    rarity = encode_value("team_a_brawler1_rarity", "Unknown")

    # Try to find class from any slot's known brawlers
    raw_df = pd.read_csv(DATA_DIR / "training_data.csv")
    b_upper = brawler.upper()
    for slot in brawler_slot_cols:
        class_col  = f"{slot}_class"
        rarity_col = f"{slot}_rarity"
        match = raw_df[raw_df[slot] == b_upper]
        if not match.empty:
            cls    = encode_value(class_col,  match.iloc[0][class_col])
            rarity = encode_value(rarity_col, match.iloc[0][rarity_col])
            break

    return cls, rarity


# Cache class/rarity lookups
_brawler_meta_cache: dict[str, tuple[int, int]] = {}

def brawler_meta(brawler: str, team: str, slot_num: int) -> tuple[int, int]:
    """Cached class/rarity lookup."""
    key = f"{team}_{slot_num}_{brawler}"
    if key not in _brawler_meta_cache:
        col_prefix = f"team_{team}_brawler{slot_num}"
        raw_df = pd.read_csv(DATA_DIR / "training_data.csv")
        b_upper = brawler.upper()
        match = raw_df[raw_df[col_prefix] == b_upper]
        if not match.empty:
            cls    = encode_value(f"{col_prefix}_class",  match.iloc[0][f"{col_prefix}_class"])
            rarity = encode_value(f"{col_prefix}_rarity", match.iloc[0][f"{col_prefix}_rarity"])
        else:
            cls    = encode_value(f"{col_prefix}_class",  "Unknown")
            rarity = encode_value(f"{col_prefix}_rarity", "Unknown")
        _brawler_meta_cache[key] = (cls, rarity)
    return _brawler_meta_cache[key]


# ─────────────────────────────────────────────
# Core recommendation function
# ─────────────────────────────────────────────

def build_row(
    map_name: str,
    my_team: list[str],       # up to 3 brawlers already picked for team A
    enemy_team: list[str],    # up to 3 brawlers already picked for team B
    candidate: str,           # the brawler we're evaluating for the next pick
) -> dict:
    """
    Build a feature row for the model.
    Team A = your team (you are always team A for recommendation purposes).
    Fills unknown slots with the most common brawler in training data (slot 1 = SHELLY as fallback).
    """
    # Pad teams to 3, inserting candidate into the next open slot
    my_picks = list(my_team) + [candidate]
    my_picks = (my_picks + ["SHELLY"] * 3)[:3]

    enemy_picks = list(enemy_team)
    enemy_picks = (enemy_picks + ["SHELLY"] * 3)[:3]

    row = {}

    # Map + game mode
    row["map"]         = encode_value("map",         map_name)
    row["game_mode"]   = encode_value("game_mode",   "Unknown")  # will be overridden
    row["environment"] = encode_value("environment", "Unknown")

    # Try to resolve game_mode from map name
    raw_maps = pd.read_csv(DATA_DIR / "raw/maps_active.csv")
    map_match = raw_maps[raw_maps["map_name"].str.upper() == map_name.upper()]
    if not map_match.empty:
        row["game_mode"]   = encode_value("game_mode",   map_match.iloc[0]["game_mode_name"])
        row["environment"] = encode_value("environment", map_match.iloc[0]["environment_name"])

    # Brawler slots
    for i, brawler in enumerate(my_picks, 1):
        col = f"team_a_brawler{i}"
        row[col] = encode_value(col, brawler)
        cls, rar = brawler_meta(brawler, "a", i)
        row[f"{col}_class"]  = cls
        row[f"{col}_rarity"] = rar

    for i, brawler in enumerate(enemy_picks, 1):
        col = f"team_b_brawler{i}"
        row[col] = encode_value(col, brawler)
        cls, rar = brawler_meta(brawler, "b", i)
        row[f"{col}_class"]  = cls
        row[f"{col}_rarity"] = rar

    return row


def recommend(
    map_name: str,
    my_team: list[str],
    enemy_team: list[str],
    top_n: int = 10,
    exclude: list[str] | None = None,
) -> pd.DataFrame:
    """
    Returns a DataFrame of recommended brawlers ranked by predicted win probability.
    """
    picked = set(b.upper() for b in my_team + enemy_team)
    if exclude:
        picked |= set(b.upper() for b in exclude)

    candidates = [b for b in all_brawlers if b.upper() not in picked]

    if not candidates:
        print("No candidates left!")
        return pd.DataFrame()

    rows = [build_row(map_name, my_team, enemy_team, c) for c in candidates]
    X = pd.DataFrame(rows)[feature_cols]

    probs = model.predict_proba(X)[:, 1]  # P(team_a_win)

    results = pd.DataFrame({
        "brawler":   candidates,
        "win_prob":  probs,
    }).sort_values("win_prob", ascending=False).reset_index(drop=True)

    results["rank"] = results.index + 1
    results["win_prob_pct"] = (results["win_prob"] * 100).round(1).astype(str) + "%"

    return results[["rank", "brawler", "win_prob_pct", "win_prob"]].head(top_n)


# ─────────────────────────────────────────────
# Display helper
# ─────────────────────────────────────────────

def print_recommendations(df: pd.DataFrame, map_name: str, my_team: list, enemy_team: list):
    print(f"\n{'═'*50}")
    print(f"  Map:       {map_name}")
    print(f"  My team:   {', '.join(my_team) if my_team else '(none yet)'}")
    print(f"  Enemy:     {', '.join(enemy_team) if enemy_team else '(none yet)'}")
    print(f"{'═'*50}")
    print(f"  {'#':<4} {'Brawler':<20} {'Win Prob'}")
    print(f"  {'-'*35}")
    for _, row in df.iterrows():
        bar = "█" * int(row["win_prob"] * 20)
        print(f"  {int(row['rank']):<4} {row['brawler']:<20} {row['win_prob_pct']:<8} {bar}")
    print()


# ─────────────────────────────────────────────
# Interactive mode
# ─────────────────────────────────────────────

def interactive():
    print("\nBrawl Stars Brawler Recommender")
    print("Type brawler names in CAPS (e.g. SHELLY). Press Enter to skip a slot.\n")

    map_name = input("Map name: ").strip()

    print("\nYour team picks (up to 2 already picked, leave blank if none):")
    my_team = []
    for i in range(1, 3):
        pick = input(f"  Your pick {i}: ").strip().upper()
        if pick:
            my_team.append(pick)

    print("\nEnemy team picks (up to 3 already picked, leave blank if none):")
    enemy_team = []
    for i in range(1, 4):
        pick = input(f"  Enemy pick {i}: ").strip().upper()
        if pick:
            enemy_team.append(pick)

    top_n = input("\nHow many recommendations? (default 10): ").strip()
    top_n = int(top_n) if top_n.isdigit() else 10

    recs = recommend(map_name, my_team, enemy_team, top_n=top_n)
    print_recommendations(recs, map_name, my_team, enemy_team)


# ─────────────────────────────────────────────
# CLI mode
# ─────────────────────────────────────────────

def cli():
    parser = argparse.ArgumentParser(description="Brawl Stars brawler recommender")
    parser.add_argument("--map",        required=True, help="Map name (e.g. 'Singed Earth')")
    parser.add_argument("--my-team",    nargs="*", default=[], help="Your already-picked brawlers")
    parser.add_argument("--enemy-team", nargs="*", default=[], help="Enemy already-picked brawlers")
    parser.add_argument("--top",        type=int,  default=10,  help="Number of recommendations")
    args = parser.parse_args()

    recs = recommend(args.map, args.my_team, args.enemy_team, top_n=args.top)
    print_recommendations(recs, args.map, args.my_team, args.enemy_team)


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        cli()
    else:
        interactive()
