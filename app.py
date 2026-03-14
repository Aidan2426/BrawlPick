"""
BrawlPick — Brawl Stars Draft Recommender
Snake draft format: A, B, B, A, A, B
"""

import json
import pickle
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

st.set_page_config(page_title="BrawlPick", page_icon="🎮", layout="wide")

# ─────────────────────────────────────────────
# Snake draft order: which team picks at each slot (0-indexed)
# Slot:  0  1  2  3  4  5
# Team:  A  B  B  A  A  B
# ─────────────────────────────────────────────
DRAFT_ORDER = ["A", "B", "B", "A", "A", "B"]  # whose pick at each position

# ─────────────────────────────────────────────
# Load model + data
# ─────────────────────────────────────────────

@st.cache_resource
def load_model():
    with open("model/rf_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("model/model_meta.json") as f:
        meta = json.load(f)
    return model, meta

@st.cache_data
def load_data():
    brawlers = pd.read_csv("data/raw/brawlers.csv")
    maps     = pd.read_csv("data/raw/maps_active.csv")
    train    = pd.read_csv("data/training_data.csv")
    with open("data/label_encodings.json") as f:
        label_encodings = json.load(f)
    return brawlers, maps, train, label_encodings

model, meta = load_model()
brawlers_df, maps_df, train_df, label_encodings = load_data()

feature_cols = meta["feature_cols"]

encoders = {
    col: {name: i for i, name in enumerate(classes)}
    for col, classes in label_encodings.items()
}

BRAWLER_SLOTS = [
    "team_a_brawler1", "team_a_brawler2", "team_a_brawler3",
    "team_b_brawler1", "team_b_brawler2", "team_b_brawler3",
]
all_brawlers = sorted(set(
    name for col in BRAWLER_SLOTS
    for name in label_encodings.get(col, [])
    if name not in ("", "nan")
))

brawlers_df["name_upper"] = brawlers_df["brawler_name"].str.upper()
brawler_icons = dict(zip(brawlers_df["name_upper"], brawlers_df["icon_url"]))

# Only show maps we have actual match data for, grouped by game mode
maps_with_data = set(train_df["map"].dropna().str.strip().unique())
maps_df_filtered = maps_df[maps_df["map_name"].isin(maps_with_data)].copy()

# Build a sorted list: "Game Mode — Map Name" so the dropdown is grouped/readable
maps_df_filtered = maps_df_filtered.sort_values(["game_mode_name", "map_name"])
map_display_options = [
    f"{row['game_mode_name']} — {row['map_name']}"
    for _, row in maps_df_filtered.iterrows()
]
# Also keep maps that are in training data but not in maps_active (edge case)
known_in_active = set(maps_df_filtered["map_name"])
extra_maps = sorted(maps_with_data - known_in_active - {"nan"})
map_display_options += [f"Unknown — {m}" for m in extra_maps]

# Lookup: display string → actual map name
display_to_map = {
    f"{row['game_mode_name']} — {row['map_name']}": row["map_name"]
    for _, row in maps_df_filtered.iterrows()
}
display_to_map.update({f"Unknown — {m}": m for m in extra_maps})

map_mode = dict(zip(maps_df["map_name"].str.upper(), maps_df["game_mode_name"]))
map_env  = dict(zip(maps_df["map_name"].str.upper(), maps_df["environment_name"]))

# ─────────────────────────────────────────────
# Session state — draft picks: list of 6 items (brawler name or None)
# ─────────────────────────────────────────────

if "draft" not in st.session_state:
    st.session_state.draft = [None] * 6
if "map_name" not in st.session_state:
    st.session_state.map_name = display_to_map[map_display_options[0]]
if "rec_history" not in st.session_state:
    # List of {pick_num, recs (DataFrame), chosen} — one entry per your team's pick turn
    st.session_state.rec_history = []

# ─────────────────────────────────────────────
# Helper: derive my_team / enemy_team from draft state
# ─────────────────────────────────────────────

def get_teams():
    my_team    = [st.session_state.draft[i] for i, t in enumerate(DRAFT_ORDER) if t == "A" and st.session_state.draft[i]]
    enemy_team = [st.session_state.draft[i] for i, t in enumerate(DRAFT_ORDER) if t == "B" and st.session_state.draft[i]]
    return my_team, enemy_team

def next_pick_slot():
    """Returns index of the next unfilled pick slot, or None if draft is complete."""
    for i, pick in enumerate(st.session_state.draft):
        if pick is None:
            return i
    return None

def picked_brawlers():
    return [b for b in st.session_state.draft if b is not None]

# ─────────────────────────────────────────────
# Encoding / model helpers
# ─────────────────────────────────────────────

def encode(col, value):
    val = str(value).upper()
    for k, v in encoders.get(col, {}).items():
        if k.upper() == val:
            return v
    return 0

def brawler_class_rarity(brawler, slot_col):
    match = train_df[train_df[slot_col] == brawler.upper()]
    if not match.empty:
        cls    = encode(f"{slot_col}_class",  match.iloc[0][f"{slot_col}_class"])
        rarity = encode(f"{slot_col}_rarity", match.iloc[0][f"{slot_col}_rarity"])
    else:
        cls    = encode("team_a_brawler1_class",  "Unknown")
        rarity = encode("team_a_brawler1_rarity", "Unknown")
    return cls, rarity

def build_row_full(map_name, my_picks_3, enemy_picks_3):
    """Build a model row given exactly 3 brawlers per side."""
    row = {
        "map":         encode("map",         map_name),
        "game_mode":   encode("game_mode",   map_mode.get(map_name.upper(), "Unknown")),
        "environment": encode("environment", map_env.get(map_name.upper(),  "Unknown")),
    }
    for i, b in enumerate(my_picks_3, 1):
        col = f"team_a_brawler{i}"
        row[col] = encode(col, b)
        cls, rar = brawler_class_rarity(b, col)
        row[f"{col}_class"]  = cls
        row[f"{col}_rarity"] = rar
    for i, b in enumerate(enemy_picks_3, 1):
        col = f"team_b_brawler{i}"
        row[col] = encode(col, b)
        cls, rar = brawler_class_rarity(b, col)
        row[f"{col}_class"]  = cls
        row[f"{col}_rarity"] = rar
    return row


def recommend(map_name, my_team, enemy_team, top_n=10, n_samples=30):
    """
    Score each candidate by averaging model win-probability over n_samples
    random completions of the unknown draft slots.
    This removes the padding bias and gives context-aware scores.
    """
    picked      = set(b.upper() for b in my_team + enemy_team)
    candidates  = [b for b in all_brawlers if b not in picked]

    n_my_unknown    = 2 - len(my_team)
    n_enemy_unknown = 3 - len(enemy_team)

    rng = random.Random(42)
    scores = []

    for candidate in candidates:
        candidate_set = picked | {candidate}
        fill_pool = [b for b in all_brawlers if b not in candidate_set]

        sample_rows = []
        for _ in range(n_samples):
            fill       = rng.sample(fill_pool, min(n_my_unknown + n_enemy_unknown, len(fill_pool)))
            my_fill    = fill[:n_my_unknown]
            enemy_fill = fill[n_my_unknown: n_my_unknown + n_enemy_unknown]
            my_full    = (list(my_team) + [candidate] + my_fill)[:3]
            enemy_full = (list(enemy_team) + enemy_fill)[:3]
            sample_rows.append(build_row_full(map_name, my_full, enemy_full))

        X     = pd.DataFrame(sample_rows)[feature_cols]
        probs = model.predict_proba(X)[:, 1]
        scores.append({"brawler": candidate, "win_prob": float(np.mean(probs))})

    return (
        pd.DataFrame(scores)
        .sort_values("win_prob", ascending=False)
        .reset_index(drop=True)
        .head(top_n)
    )

# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────

st.title("🎮 BrawlPick")
st.caption(f"Draft recommender · {meta['n_train']:,} ranked matches · {meta['cv_accuracy_mean']:.0%} CV accuracy")

# Map selector + reset
col_map, col_reset = st.columns([3, 1])
with col_map:
    # Find current map's display string for the default index
    current_display = next(
        (k for k, v in display_to_map.items() if v == st.session_state.map_name),
        map_display_options[0]
    )
    selected_display = st.selectbox(
        f"Map ({len(map_display_options)} maps with match data)",
        map_display_options,
        index=map_display_options.index(current_display),
    )
    selected_map = display_to_map[selected_display]
    if selected_map != st.session_state.map_name:
        st.session_state.map_name = selected_map

with col_reset:
    st.write("")
    st.write("")
    if st.button("Reset Draft", use_container_width=True):
        st.session_state.draft = [None] * 6
        st.session_state.rec_history = []
        st.rerun()

st.divider()

# ─────────────────────────────────────────────
# Draft board — 6 slots in a row
# ─────────────────────────────────────────────

next_slot = next_pick_slot()

st.subheader("Draft Board")
slot_cols = st.columns(6)

TEAM_COLOR = {"A": "🟦", "B": "🟥"}
TEAM_LABEL = {"A": "You", "B": "Enemy"}

for i, col in enumerate(slot_cols):
    with col:
        team      = DRAFT_ORDER[i]
        pick_num  = i + 1
        is_next   = (i == next_slot)
        pick      = st.session_state.draft[i]

        label = f"{TEAM_COLOR[team]} Pick {pick_num} · {TEAM_LABEL[team]}"

        if pick:
            # Filled slot
            url = brawler_icons.get(pick)
            if url:
                st.image(url, width=64)
            st.markdown(f"**{pick.title()}**")
            st.caption(label)
        elif is_next:
            # Current pick — highlighted
            st.markdown("### ⬇️")
            st.markdown(f"**{label}**")
            st.caption("_Now picking_")
        else:
            # Future slot
            st.markdown("◻️")
            st.caption(label)

st.divider()

# ─────────────────────────────────────────────
# Pick input — only show if draft not complete
# ─────────────────────────────────────────────

if next_slot is not None:
    team_picking = DRAFT_ORDER[next_slot]
    already_picked = set(b.upper() for b in picked_brawlers())
    available = [b for b in all_brawlers if b not in already_picked]

    if team_picking == "A":
        st.subheader(f"🟦 Pick {next_slot + 1} — Your Pick")
        my_team, enemy_team = get_teams()

        # Auto-show recommendations
        with st.spinner("Calculating recommendations..."):
            recs = recommend(st.session_state.map_name, my_team, enemy_team)

        # Top 5 icons
        top5 = recs.head(5)
        st.markdown("**Recommended picks:**")
        rec_cols = st.columns(5)
        for j, (_, row) in enumerate(top5.iterrows()):
            with rec_cols[j]:
                url = brawler_icons.get(row["brawler"])
                if url:
                    st.image(url, width=64)
                st.metric(row["brawler"].title(), f"{row['win_prob']:.1%}")

        st.write("")
        chosen = st.selectbox(
            "Confirm your pick:",
            options=[""] + available,
            format_func=lambda x: x.title() if x else "— Select brawler —",
            key=f"pick_{next_slot}",
        )
        if chosen:
            if st.button(f"Lock in {chosen.title()}", type="primary"):
                # Save recommendations for this pick to history before advancing
                st.session_state.rec_history.append({
                    "pick_num": next_slot + 1,
                    "recs": recs.copy(),
                    "chosen": chosen,
                })
                st.session_state.draft[next_slot] = chosen
                st.rerun()

    else:
        st.subheader(f"🟥 Pick {next_slot + 1} — Enemy Pick")
        st.caption("Enter the enemy's pick to continue.")
        chosen = st.selectbox(
            "Enemy picked:",
            options=[""] + available,
            format_func=lambda x: x.title() if x else "— Select brawler —",
            key=f"pick_{next_slot}",
        )
        if chosen:
            if st.button(f"Lock in {chosen.title()} (enemy)", type="primary"):
                st.session_state.draft[next_slot] = chosen
                st.rerun()

else:
    # Draft complete
    st.subheader("✅ Draft Complete")
    my_team, enemy_team = get_teams()

    cola, colb = st.columns(2)
    with cola:
        st.markdown("**🟦 Your Team**")
        for b in my_team:
            c1, c2 = st.columns([1, 3])
            with c1:
                url = brawler_icons.get(b)
                if url:
                    st.image(url, width=48)
            with c2:
                st.write(b.title())
    with colb:
        st.markdown("**🟥 Enemy Team**")
        for b in enemy_team:
            c1, c2 = st.columns([1, 3])
            with c1:
                url = brawler_icons.get(b)
                if url:
                    st.image(url, width=48)
            with c2:
                st.write(b.title())

    # Final win probability
    with st.spinner("Calculating final win probability..."):
        rows  = [build_row_full(st.session_state.map_name, my_team[:3], enemy_team[:3])]
        X     = pd.DataFrame(rows)[feature_cols]
        prob  = model.predict_proba(X)[0][1]

    st.divider()
    st.metric("Estimated Win Probability (Your Team)", f"{prob:.1%}")
    if prob >= 0.55:
        st.success("Strong draft! You have a composition advantage.")
    elif prob >= 0.50:
        st.info("Roughly even draft.")
    else:
        st.warning("Tough matchup — the enemy draft has the edge.")

# ─────────────────────────────────────────────
# Recommendation history
# ─────────────────────────────────────────────

if st.session_state.rec_history:
    st.divider()
    st.subheader("📋 Recommendation History")
    st.caption("What the model suggested at each of your pick turns.")

    for entry in st.session_state.rec_history:
        pick_num = entry["pick_num"]
        chosen   = entry["chosen"]
        recs     = entry["recs"]

        chosen_rank = recs[recs["brawler"] == chosen].index
        rank_label  = f"(you picked #{int(chosen_rank[0]) + 1} on the list)" if len(chosen_rank) else ""

        with st.expander(f"Pick {pick_num} — You chose **{chosen.title()}** {rank_label}", expanded=False):
            top = recs.head(8)
            cols = st.columns([1, 3, 2])
            cols[0].markdown("**#**")
            cols[1].markdown("**Brawler**")
            cols[2].markdown("**Win Prob**")
            for _, row in top.iterrows():
                c0, c1, c2 = st.columns([1, 3, 2])
                is_chosen = row["brawler"] == chosen
                rank_str  = f"**{int(row.name)+1}**" if is_chosen else str(int(row.name)+1)
                name_str  = f"**{row['brawler'].title()} ✓**" if is_chosen else row["brawler"].title()
                prob_str  = f"**{row['win_prob']:.1%}**" if is_chosen else f"{row['win_prob']:.1%}"
                c0.markdown(rank_str)
                c1.markdown(name_str)
                c2.markdown(prob_str)

# Sidebar
with st.sidebar:
    st.header("Model Info")
    st.metric("Test Accuracy",     f"{meta['test_accuracy']:.1%}")
    st.metric("CV Accuracy",       f"{meta['cv_accuracy_mean']:.1%} ± {meta['cv_accuracy_std']:.1%}")
    st.metric("Training matches",  f"{meta['n_train']:,}")
    st.divider()
    st.markdown("**Draft Order**")
    for i, team in enumerate(DRAFT_ORDER):
        icon = "🟦" if team == "A" else "🟥"
        label = "You" if team == "A" else "Enemy"
        filled = st.session_state.draft[i]
        name = filled.title() if filled else "_empty_"
        st.caption(f"Pick {i+1}: {icon} {label} — {name}")
