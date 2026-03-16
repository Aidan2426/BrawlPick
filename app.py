"""
BrawlPick — Brawl Stars Draft Recommender
Snake draft format: A, B, B, A, A, B
"""

import json
import joblib
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

st.set_page_config(page_title="BrawlPick", page_icon="logo.jpg", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nougat&display=swap');
html, body, [class*="css"], h1, h2, h3, h4, h5, h6, p, div, span, button, label, input, select, textarea {
    font-family: 'Nougat', sans-serif !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Snake draft order
# Slot:  0  1  2  3  4  5
# Team:  A  B  B  A  A  B
# ─────────────────────────────────────────────
DRAFT_ORDER = ["A", "B", "B", "A", "A", "B"]

BRAWLER_SLOTS = [
    "team_a_brawler1", "team_a_brawler2", "team_a_brawler3",
    "team_b_brawler1", "team_b_brawler2", "team_b_brawler3",
]

# ─────────────────────────────────────────────
# Load model + data
# ─────────────────────────────────────────────

@st.cache_resource
def load_model():
    model = joblib.load("model/rf_model.pkl")
    with open("model/model_meta.json") as f:
        meta = json.load(f)
    return model, meta

@st.cache_data
def load_data():
    brawlers = pd.read_csv("data/raw/brawlers.csv")
    maps     = pd.read_csv("data/raw/maps_active.csv")
    train    = pd.read_csv("data/training_data.csv")
    for col in BRAWLER_SLOTS:
        train[col] = train[col].str.upper()
    return brawlers, maps, train

model, meta = load_model()
brawlers_df, maps_df, train_df = load_data()

feature_cols      = meta["feature_cols"]
brawler_win_rates = meta["brawler_win_rates"]       # brawler → win rate (target encoding)
brawler_global_mean = meta["brawler_global_mean"]   # fallback for unknown brawlers
label_encodings   = meta["label_encodings"]         # col → list of classes

# Label encoding lookup: col → {value: int}
label_encoders = {
    col: {cls: i for i, cls in enumerate(classes)}
    for col, classes in label_encodings.items()
}

# All known brawlers
all_brawlers = sorted(brawler_win_rates.keys())

# Brawler icons
brawlers_df["name_upper"] = brawlers_df["brawler_name"].str.upper()
brawler_icons = dict(zip(brawlers_df["name_upper"], brawlers_df["icon_url"]))

# Map dropdown — only maps with at least 10 matches in training data
map_counts = train_df["map"].dropna().str.strip().value_counts()
maps_with_data = set(map_counts[map_counts >= 10].index)
maps_df_filtered = maps_df[maps_df["map_name"].isin(maps_with_data)].copy()
maps_df_filtered = maps_df_filtered.sort_values(["game_mode_name", "map_name"])
map_display_options = [
    f"{row['game_mode_name']} — {row['map_name']}"
    for _, row in maps_df_filtered.iterrows()
]
known_in_active = set(maps_df_filtered["map_name"])
extra_maps = sorted(maps_with_data - known_in_active - {"nan"})
map_display_options += [f"Unknown — {m}" for m in extra_maps]

display_to_map = {
    f"{row['game_mode_name']} — {row['map_name']}": row["map_name"]
    for _, row in maps_df_filtered.iterrows()
}
display_to_map.update({f"Unknown — {m}": m for m in extra_maps})

map_mode = dict(zip(maps_df["map_name"].str.upper(), maps_df["game_mode_name"]))
map_env  = dict(zip(maps_df["map_name"].str.upper(), maps_df["environment_name"]))

# ─────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────

if "draft" not in st.session_state:
    st.session_state.draft = [None] * 6
if "map_name" not in st.session_state:
    st.session_state.map_name = display_to_map[map_display_options[0]]
if "rec_history" not in st.session_state:
    st.session_state.rec_history = []
if "map_locked" not in st.session_state:
    st.session_state.map_locked = False

# ─────────────────────────────────────────────
# Draft helpers
# ─────────────────────────────────────────────

def get_teams():
    my_team    = [st.session_state.draft[i] for i, t in enumerate(DRAFT_ORDER) if t == "A" and st.session_state.draft[i]]
    enemy_team = [st.session_state.draft[i] for i, t in enumerate(DRAFT_ORDER) if t == "B" and st.session_state.draft[i]]
    return my_team, enemy_team

def next_pick_slot():
    for i, pick in enumerate(st.session_state.draft):
        if pick is None:
            return i
    return None

def picked_brawlers():
    return [b for b in st.session_state.draft if b is not None]

# ─────────────────────────────────────────────
# Encoding helpers
# ─────────────────────────────────────────────

def encode_brawler(brawler: str) -> float:
    """Target encode: return this brawler's historical win rate."""
    return brawler_win_rates.get(brawler.upper(), brawler_global_mean)

def encode_label(col: str, value: str) -> int:
    """Label encode a non-brawler categorical."""
    val = str(value).upper()
    mapping = label_encoders.get(col, {})
    for k, v in mapping.items():
        if k.upper() == val:
            return v
    return 0

def build_row_full(map_name: str, my_picks_3: list, enemy_picks_3: list) -> dict:
    """Build one feature row for the model given exactly 3 brawlers per side."""
    row = {
        "map":       encode_label("map",       map_name),
        "game_mode": encode_label("game_mode", map_mode.get(map_name.upper(), "Unknown")),
    }
    for i, b in enumerate(my_picks_3, 1):
        row[f"team_a_brawler{i}"] = encode_brawler(b)
    for i, b in enumerate(enemy_picks_3, 1):
        row[f"team_b_brawler{i}"] = encode_brawler(b)
    return row


# ─────────────────────────────────────────────
# Recommendation engine
# ─────────────────────────────────────────────

def recommend(map_name: str, my_team: list, enemy_team: list, top_n: int = 10, n_samples: int = 30) -> pd.DataFrame:
    """
    Score each candidate brawler by averaging the model's win probability over
    n_samples random completions of the unknown draft slots.
    All candidates are scored in a single batched predict_proba call for speed.
    """
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
            all_rows.append(build_row_full(map_name, my_full, enemy_full))

    # Single batched predict_proba call across all candidates × samples
    X     = pd.DataFrame(all_rows)[feature_cols]
    probs = model.predict_proba(X)[:, 1]

    scores = [
        {"brawler": c, "win_prob": float(probs[i * n_samples:(i + 1) * n_samples].mean())}
        for i, c in enumerate(candidates)
    ]

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

col_map, col_lock, col_reset = st.columns([3, 1, 1])
with col_map:
    current_display = next(
        (k for k, v in display_to_map.items() if v == st.session_state.map_name),
        map_display_options[0]
    )
    if st.session_state.map_locked:
        st.markdown(f"**Map:** {current_display}")
        st.caption("Map is locked for this draft.")
    else:
        selected_display = st.selectbox(
            f"Map ({len(map_display_options)} maps with 10+ matches)",
            map_display_options,
            index=map_display_options.index(current_display),
        )
        selected_map = display_to_map[selected_display]
        if selected_map != st.session_state.map_name:
            st.session_state.map_name = selected_map

with col_lock:
    st.write("")
    st.write("")
    if st.session_state.map_locked:
        if st.button("🔓 Change Map", use_container_width=True):
            st.session_state.map_locked = False
            st.session_state.draft = [None] * 6
            st.session_state.rec_history = []
            st.rerun()
    else:
        if st.button("🔒 Lock Map", use_container_width=True, type="primary"):
            st.session_state.map_locked = True
            st.rerun()

with col_reset:
    st.write("")
    st.write("")
    if st.button("Reset Draft", use_container_width=True):
        st.session_state.draft = [None] * 6
        st.session_state.rec_history = []
        st.session_state.map_locked = False
        st.rerun()

st.divider()

if not st.session_state.map_locked:
    st.info("Select a map and click **Lock Map** to start the draft.")
    st.stop()

# ─────────────────────────────────────────────
# Draft board
# ─────────────────────────────────────────────

next_slot = next_pick_slot()

st.subheader("Draft Board")
slot_cols = st.columns(6)

TEAM_COLOR = {"A": "🟦", "B": "🟥"}
TEAM_LABEL = {"A": "You", "B": "Enemy"}

for i, col in enumerate(slot_cols):
    with col:
        team     = DRAFT_ORDER[i]
        pick_num = i + 1
        is_next  = (i == next_slot)
        pick     = st.session_state.draft[i]
        label    = f"{TEAM_COLOR[team]} Pick {pick_num} · {TEAM_LABEL[team]}"

        if pick:
            url = brawler_icons.get(pick)
            if url:
                st.image(url, width=64)
            st.markdown(f"**{pick.title()}**")
            st.caption(label)
        elif is_next:
            st.markdown("### ⬇️")
            st.markdown(f"**{label}**")
            st.caption("_Now picking_")
        else:
            st.markdown("◻️")
            st.caption(label)

st.divider()

# ─────────────────────────────────────────────
# Pick input
# ─────────────────────────────────────────────

if next_slot is not None:
    team_picking   = DRAFT_ORDER[next_slot]
    already_picked = set(b.upper() for b in picked_brawlers())
    available      = [b for b in all_brawlers if b not in already_picked]

    if team_picking == "A":
        st.subheader(f"🟦 Pick {next_slot + 1} — Your Pick")
        my_team, enemy_team = get_teams()

        with st.spinner("Calculating recommendations..."):
            recs = recommend(st.session_state.map_name, my_team, enemy_team)

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
                st.session_state.rec_history.append({
                    "pick_num": next_slot + 1,
                    "recs":     recs.copy(),
                    "chosen":   chosen,
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

    my_avg    = sum(encode_brawler(b) for b in my_team)    / len(my_team)
    enemy_avg = sum(encode_brawler(b) for b in enemy_team) / len(enemy_team)
    edge = my_avg - enemy_avg

    st.divider()
    m1, m2 = st.columns(2)
    m1.metric("Your Team Avg Win Rate",    f"{my_avg:.1%}")
    m2.metric("Enemy Team Avg Win Rate",   f"{enemy_avg:.1%}")
    st.caption("Based on each brawler's historical win rate across all matches in the dataset.")

    if edge >= 0.03:
        st.success("Strong draft! Your brawlers have a clear historical advantage.")
    elif edge >= 0.01:
        st.info("Slight edge to your team.")
    elif edge >= -0.01:
        st.info("Roughly even draft.")
    else:
        st.warning("Tough matchup — the enemy's brawlers have the historical edge.")

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
    st.metric("Test Accuracy",    f"{meta['test_accuracy']:.1%}")
    st.metric("CV Accuracy",      f"{meta['cv_accuracy_mean']:.1%} ± {meta['cv_accuracy_std']:.1%}")
    st.metric("Training matches", f"{meta['n_train']:,}")
    st.divider()
    st.markdown("**Draft Order**")
    for i, team in enumerate(DRAFT_ORDER):
        icon  = "🟦" if team == "A" else "🟥"
        label = "You" if team == "A" else "Enemy"
        filled = st.session_state.draft[i]
        name   = filled.title() if filled else "_empty_"
        st.caption(f"Pick {i+1}: {icon} {label} — {name}")
