"""
BrawlPick — Data Explorer
Run: python diagnose.py

Shows raw win rates from training data for any map/draft state,
so you can see exactly what the model is working with.

Commands:
  map <map name>               — all brawler win rates on that map
  pick <map> | <my> | <enemy>  — model predictions for a draft state
  quit                         — exit

Examples:
  map Hard Rock Mine
  pick Hard Rock Mine | SHELLY CROW | EDGAR
"""

import json
import random
import warnings
from pathlib import Path

import joblib
import pandas as pd

warnings.filterwarnings("ignore")

MODEL_DIR = Path("model")
DATA_DIR  = Path("data")

BRAWLER_SLOTS = [
    "team_a_brawler1", "team_a_brawler2", "team_a_brawler3",
    "team_b_brawler1", "team_b_brawler2", "team_b_brawler3",
]

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading model and data...")
model = joblib.load(MODEL_DIR / "rf_model.pkl")
with open(MODEL_DIR / "model_meta.json") as f:
    meta = json.load(f)

feature_cols          = meta["feature_cols"]
brawler_win_rates     = meta["brawler_win_rates"]
brawler_map_win_rates = meta.get("brawler_map_win_rates", {})
brawler_global_mean   = meta["brawler_global_mean"]
label_encodings       = meta["label_encodings"]
all_brawlers          = sorted(brawler_win_rates.keys())

train_df = pd.read_csv(DATA_DIR / "training_data.csv")
for col in BRAWLER_SLOTS:
    train_df[col] = train_df[col].str.upper()

maps_df  = pd.read_csv(DATA_DIR / "raw/maps_active.csv")
map_mode = dict(zip(maps_df["map_name"].str.upper(), maps_df["game_mode_name"]))

label_encoders = {
    col: {cls: i for i, cls in enumerate(classes)}
    for col, classes in label_encodings.items()
}

print("Ready.\n")


# ── Encoding helpers ──────────────────────────────────────────────────────────

def encode_brawler(brawler, map_name=""):
    key = f"{map_name}|{brawler.upper()}"
    if key in brawler_map_win_rates:
        return brawler_map_win_rates[key]
    return brawler_win_rates.get(brawler.upper(), brawler_global_mean)

def encode_label(col, value):
    val = str(value).upper()
    for k, v in label_encoders.get(col, {}).items():
        if k.upper() == val:
            return v
    return 0

def build_row(map_name, my_picks_3, enemy_picks_3):
    row = {
        "map":       encode_label("map", map_name),
        "game_mode": encode_label("game_mode", map_mode.get(map_name.upper(), "Unknown")),
    }
    for i, b in enumerate(my_picks_3, 1):
        row[f"team_a_brawler{i}"] = encode_brawler(b, map_name)
    for i, b in enumerate(enemy_picks_3, 1):
        row[f"team_b_brawler{i}"] = encode_brawler(b, map_name)
    return row


# ── Commands ──────────────────────────────────────────────────────────────────

def show_map_winrates(map_name):
    """All brawlers ranked by win rate on this specific map from training data."""
    map_upper = map_name.upper()
    results = []
    for brawler in all_brawlers:
        key = f"{map_upper}|{brawler}"
        map_wr    = brawler_map_win_rates.get(key)
        global_wr = brawler_win_rates.get(brawler, brawler_global_mean)
        count = sum(
            len(train_df[(train_df["map"].str.upper() == map_upper) & (train_df[col] == brawler)])
            for col in BRAWLER_SLOTS
        )
        results.append({"brawler": brawler, "map_wr": map_wr, "global_wr": global_wr, "n_games": count})

    df = pd.DataFrame(results)
    df_with    = df[df["map_wr"].notna()].sort_values("map_wr", ascending=False)
    df_without = df[df["map_wr"].isna()].sort_values("global_wr", ascending=False)

    mode = map_mode.get(map_upper, "Unknown")
    print(f"\n{'═'*70}")
    print(f"  MAP: {map_name}  |  MODE: {mode}")
    print(f"  (* = fewer than 15 games on this map, less reliable)")
    print(f"{'═'*70}")
    print(f"  {'#':<4} {'Brawler':<22} {'Map WR':>7}  {'Global WR':>9}  {'Games':>6}")
    print(f"  {'-'*60}")

    for rank, (_, row) in enumerate(df_with.iterrows(), 1):
        flag = " *" if row["n_games"] < 15 else ""
        print(f"  {rank:<4} {row['brawler'].title():<22} {row['map_wr']:.1%}     {row['global_wr']:.1%}     {int(row['n_games']):>5}{flag}")

    if not df_without.empty:
        print(f"\n  [No map-specific data — global rate only]")
        for _, row in df_without.iterrows():
            print(f"  {'—':<4} {row['brawler'].title():<22} {'—':>7}     {row['global_wr']:.1%}")
    print()


def show_model_predict(map_name, my_team, enemy_team, n_samples=30):
    """Show model predicted win% for all brawlers given a draft state."""
    if not my_team and not enemy_team:
        print("\n[First pick — using map win rates directly, no model needed]")
        show_map_winrates(map_name)
        return

    picked     = set(b.upper() for b in my_team + enemy_team)
    candidates = [b for b in all_brawlers if b not in picked]
    n_my_unknown    = 2 - len(my_team)
    n_enemy_unknown = 3 - len(enemy_team)
    rng = random.Random(42)
    all_rows = []

    for candidate in candidates:
        candidate_set = picked | {candidate}
        fill_pool = [b for b in all_brawlers if b not in candidate_set]
        for _ in range(n_samples):
            fill       = rng.sample(fill_pool, min(n_my_unknown + n_enemy_unknown, len(fill_pool)))
            my_full    = (list(my_team) + [candidate] + fill[:n_my_unknown])[:3]
            enemy_full = (list(enemy_team) + fill[n_my_unknown: n_my_unknown + n_enemy_unknown])[:3]
            all_rows.append(build_row(map_name, my_full, enemy_full))

    X     = pd.DataFrame(all_rows)[feature_cols]
    probs = model.predict_proba(X)[:, 1]

    scores = [
        {
            "brawler":   c,
            "model_win": float(probs[i * n_samples:(i + 1) * n_samples].mean()),
            "map_wr":    brawler_map_win_rates.get(f"{map_name.upper()}|{c}"),
            "global_wr": brawler_win_rates.get(c, brawler_global_mean),
        }
        for i, c in enumerate(candidates)
    ]

    df = pd.DataFrame(scores).sort_values("model_win", ascending=False).reset_index(drop=True)
    mode = map_mode.get(map_name.upper(), "Unknown")

    print(f"\n{'═'*80}")
    print(f"  MAP: {map_name}  |  MODE: {mode}")
    print(f"  My team:    {', '.join(t.title() for t in my_team) or '(none)'}")
    print(f"  Enemy team: {', '.join(t.title() for t in enemy_team) or '(none)'}")
    print(f"  Model win% = avg predicted win probability over {n_samples} random draft completions")
    print(f"{'═'*80}")
    print(f"  {'#':<4} {'Brawler':<22} {'Model Win%':>10}  {'Map WR':>8}  {'Global WR':>9}")
    print(f"  {'-'*65}")
    for rank, (_, row) in enumerate(df.iterrows(), 1):
        map_str = f"{row['map_wr']:.1%}" if row["map_wr"] is not None else "  —  "
        print(f"  {rank:<4} {row['brawler'].title():<22} {row['model_win']:.1%}       {map_str:>7}   {row['global_wr']:.1%}")
    print()


# ── Interactive loop ──────────────────────────────────────────────────────────

def main():
    print("Commands:")
    print("  map <map name>                  — all brawler win rates on that map")
    print("  pick <map> | <my> | <enemy>     — model predictions for a draft state")
    print("  quit\n")
    print("Examples:")
    print("  map Hard Rock Mine")
    print("  pick Hard Rock Mine | SHELLY CROW | EDGAR")
    print("  pick Hard Rock Mine |  | EDGAR  (enemy first pick, your team empty)\n")

    while True:
        try:
            raw = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not raw:
            continue
        if raw.lower() in ("quit", "exit", "q"):
            break

        if raw.lower().startswith("map "):
            show_map_winrates(raw[4:].strip())

        elif raw.lower().startswith("pick "):
            parts      = [p.strip() for p in raw[5:].split("|")]
            map_name   = parts[0] if parts else ""
            my_team    = [b.upper() for b in parts[1].split()] if len(parts) > 1 and parts[1].strip() else []
            enemy_team = [b.upper() for b in parts[2].split()] if len(parts) > 2 and parts[2].strip() else []
            show_model_predict(map_name, my_team, enemy_team)

        else:
            print("  Unknown command. Try: map <name>  or  pick <map> | <my picks> | <enemy picks>")

    print("Bye.")


if __name__ == "__main__":
    main()
