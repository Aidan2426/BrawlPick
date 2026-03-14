"""
BrawlPick — Brawl Stars Draft Recommender
Streamlit app: select map + current picks, get brawler recommendations.
"""

import json
import pickle
import warnings
from pathlib import Path

import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="BrawlPick",
    page_icon="🎮",
    layout="wide",
)

# ─────────────────────────────────────────────
# Load model + data (cached so it only runs once)
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
    brawlers  = pd.read_csv("data/raw/brawlers.csv")
    maps      = pd.read_csv("data/raw/maps_active.csv")
    train     = pd.read_csv("data/training_data.csv")
    with open("data/label_encodings.json") as f:
        label_encodings = json.load(f)
    return brawlers, maps, train, label_encodings


model, meta         = load_model()
brawlers_df, maps_df, train_df, label_encodings = load_data()

feature_cols = meta["feature_cols"]

# Build encoders: col → {name: int}
encoders = {
    col: {name: i for i, name in enumerate(classes)}
    for col, classes in label_encodings.items()
}

# All known brawlers
BRAWLER_SLOTS = [
    "team_a_brawler1", "team_a_brawler2", "team_a_brawler3",
    "team_b_brawler1", "team_b_brawler2", "team_b_brawler3",
]
all_brawlers = sorted(set(
    name
    for col in BRAWLER_SLOTS
    for name in label_encodings.get(col, [])
    if name not in ("", "nan")
))

# Brawler icon URLs: name → URL
brawlers_df["name_upper"] = brawlers_df["brawler_name"].str.upper()
brawler_icons = dict(zip(brawlers_df["name_upper"], brawlers_df["icon_url"]))

# Map list for dropdown — sorted alphabetically
map_names = sorted(maps_df["map_name"].dropna().unique().tolist())

# Map → game_mode lookup
map_mode = dict(zip(
    maps_df["map_name"].str.upper(),
    maps_df["game_mode_name"]
))
map_env = dict(zip(
    maps_df["map_name"].str.upper(),
    maps_df["environment_name"]
))


# ─────────────────────────────────────────────
# Encoding helpers
# ─────────────────────────────────────────────

def encode(col: str, value: str) -> int:
    val = str(value).upper()
    m = encoders.get(col, {})
    for k, v in m.items():
        if k.upper() == val:
            return v
    return 0


def brawler_class_rarity(brawler: str, slot_col: str):
    b = brawler.upper()
    match = train_df[train_df[slot_col] == b]
    if not match.empty:
        cls    = encode(f"{slot_col}_class",  match.iloc[0][f"{slot_col}_class"])
        rarity = encode(f"{slot_col}_rarity", match.iloc[0][f"{slot_col}_rarity"])
    else:
        cls    = encode(f"team_a_brawler1_class",  "Unknown")
        rarity = encode(f"team_a_brawler1_rarity", "Unknown")
    return cls, rarity


def build_row(map_name, my_team, enemy_team, candidate):
    my_picks     = (list(my_team) + [candidate] + ["SHELLY"] * 3)[:3]
    enemy_picks  = (list(enemy_team) + ["SHELLY"] * 3)[:3]

    row = {
        "map":         encode("map",         map_name),
        "game_mode":   encode("game_mode",   map_mode.get(map_name.upper(), "Unknown")),
        "environment": encode("environment", map_env.get(map_name.upper(),  "Unknown")),
    }

    for i, b in enumerate(my_picks, 1):
        col = f"team_a_brawler{i}"
        row[col] = encode(col, b)
        cls, rar = brawler_class_rarity(b, col)
        row[f"{col}_class"]  = cls
        row[f"{col}_rarity"] = rar

    for i, b in enumerate(enemy_picks, 1):
        col = f"team_b_brawler{i}"
        row[col] = encode(col, b)
        cls, rar = brawler_class_rarity(b, col)
        row[f"{col}_class"]  = cls
        row[f"{col}_rarity"] = rar

    return row


def recommend(map_name, my_team, enemy_team, top_n=15):
    picked = set(b.upper() for b in my_team + enemy_team)
    candidates = [b for b in all_brawlers if b not in picked]

    rows = [build_row(map_name, my_team, enemy_team, c) for c in candidates]
    X = pd.DataFrame(rows)[feature_cols]
    probs = model.predict_proba(X)[:, 1]

    return (
        pd.DataFrame({"brawler": candidates, "win_prob": probs})
        .sort_values("win_prob", ascending=False)
        .reset_index(drop=True)
        .head(top_n)
    )


# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────

st.title("🎮 BrawlPick")
st.caption(f"Random Forest draft recommender · {meta['n_train']:,} ranked matches · {meta['cv_accuracy_mean']:.0%} CV accuracy")

st.divider()

# Map selector
col_map, col_spacer = st.columns([2, 3])
with col_map:
    map_name = st.selectbox("Select Map", map_names, index=0)
    mode_label = map_mode.get(map_name.upper(), "")
    if mode_label:
        st.caption(f"Mode: **{mode_label}**")

st.divider()

# Team pickers
col_my, col_enemy = st.columns(2)

with col_my:
    st.subheader("Your Team")
    my_picks = st.multiselect(
        "Your picks so far (up to 2)",
        options=all_brawlers,
        max_selections=2,
        placeholder="Select brawlers...",
        key="my_team",
    )
    if my_picks:
        icon_cols = st.columns(len(my_picks))
        for i, b in enumerate(my_picks):
            with icon_cols[i]:
                url = brawler_icons.get(b)
                if url:
                    st.image(url, width=64)
                st.caption(b.title())

with col_enemy:
    st.subheader("Enemy Team")
    # Filter out already-picked brawlers
    available_for_enemy = [b for b in all_brawlers if b not in my_picks]
    enemy_picks = st.multiselect(
        "Enemy picks (up to 3)",
        options=available_for_enemy,
        max_selections=3,
        placeholder="Select brawlers...",
        key="enemy_team",
    )
    if enemy_picks:
        icon_cols = st.columns(len(enemy_picks))
        for i, b in enumerate(enemy_picks):
            with icon_cols[i]:
                url = brawler_icons.get(b)
                if url:
                    st.image(url, width=64)
                st.caption(b.title())

st.divider()

# Recommend button
if st.button("Get Recommendations", type="primary", use_container_width=True):
    if not map_name:
        st.warning("Please select a map first.")
    else:
        with st.spinner("Running model..."):
            recs = recommend(map_name, my_picks, enemy_picks, top_n=15)

        st.subheader(f"Top Picks for {map_name}")

        # Show top 5 with icons in a row
        st.markdown("**Best options:**")
        top5 = recs.head(5)
        icon_row = st.columns(5)
        for i, (_, row) in enumerate(top5.iterrows()):
            with icon_row[i]:
                url = brawler_icons.get(row["brawler"])
                if url:
                    st.image(url, width=72)
                st.metric(
                    label=row["brawler"].title(),
                    value=f"{row['win_prob']:.1%}",
                )

        st.divider()

        # Full ranked table with inline bar
        st.markdown("**Full rankings:**")
        for _, row in recs.iterrows():
            pct = row["win_prob"]
            bar_filled = int(pct * 30)
            bar = "█" * bar_filled + "░" * (30 - bar_filled)
            cols = st.columns([3, 1, 6])
            with cols[0]:
                url = brawler_icons.get(row["brawler"])
                if url:
                    st.image(url, width=32)
                st.write(f"**{row['brawler'].title()}**")
            with cols[1]:
                st.write(f"`{pct:.1%}`")
            with cols[2]:
                st.write(f"`{bar}`")

# Sidebar: model info
with st.sidebar:
    st.header("Model Info")
    st.metric("Test Accuracy", f"{meta['test_accuracy']:.1%}")
    st.metric("CV Accuracy",   f"{meta['cv_accuracy_mean']:.1%} ± {meta['cv_accuracy_std']:.1%}")
    st.metric("Training matches", f"{meta['n_train']:,}")
    st.divider()
    st.caption("Top features driving predictions:")
    for feat, imp in list(meta["top_features"].items())[:8]:
        st.caption(f"• {feat}: {imp:.3f}")
