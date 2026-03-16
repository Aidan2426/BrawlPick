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

st.set_page_config(page_title="BrawlPick", page_icon="Brawl Stars Logo.png", layout="wide")

def _load_font_b64(path: str) -> str:
    import base64
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

_font_b64 = _load_font_b64("Nougat-ExtraBlack.ttf")

def _load_img_b64(path: str) -> str:
    import base64
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

_logo_b64 = _load_img_b64("Brawl Stars Logo.png")
st.markdown(f"""
<style>
@font-face {{
    font-family: 'Nougat';
    src: url('data:font/truetype;base64,{_font_b64}') format('truetype');
}}
html, body, h1, h2, h3, h4, h5, h6, p, button, label, input, select, textarea,
.stMarkdown, .stMetric, .stButton, .stSelectbox, .stDataFrame, .stExpander p {{
    font-family: 'Nougat', sans-serif !important;
}}
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

feature_cols            = meta["feature_cols"]
brawler_win_rates       = meta["brawler_win_rates"]
brawler_map_win_rates   = meta.get("brawler_map_win_rates", {})
brawler_map_game_counts = meta.get("brawler_map_game_counts", {})
brawler_global_mean     = meta["brawler_global_mean"]
label_encodings         = meta["label_encodings"]

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

# Ranked-eligible game modes only (excludes Duels, 5v5, etc.)
RANKED_MODES = {"Gem Grab", "Brawl Ball", "Heist", "Bounty", "Knockout", "Hot Zone"}

# Map dropdown — ranked modes only, with at least 10 matches in training data
map_counts = train_df["map"].dropna().str.strip().value_counts()
maps_with_data = set(map_counts[map_counts >= 10].index)
maps_df_filtered = maps_df[
    maps_df["map_name"].isin(maps_with_data) &
    maps_df["game_mode_name"].isin(RANKED_MODES)
].copy()
maps_df_filtered = maps_df_filtered.sort_values(["game_mode_name", "map_name"])
map_display_options = [
    f"{row['game_mode_name']} — {row['map_name']}"
    for _, row in maps_df_filtered.iterrows()
]

display_to_map = {
    f"{row['game_mode_name']} — {row['map_name']}": row["map_name"]
    for _, row in maps_df_filtered.iterrows()
}

map_mode      = dict(zip(maps_df["map_name"].str.upper(), maps_df["game_mode_name"]))
map_env       = dict(zip(maps_df["map_name"].str.upper(), maps_df["environment_name"]))
map_image_url = dict(zip(maps_df["map_name"], maps_df["image_url"]))

# Mode icon emoji
MODE_ICONS = {
    "Gem Grab":  "💎",
    "Brawl Ball": "⚽",
    "Heist":     "💰",
    "Bounty":    "⭐",
    "Knockout":  "💀",
    "Hot Zone":  "🔥",
}

# Mode color accents
MODE_COLORS = {
    "Gem Grab":  "#d852ff",
    "Brawl Ball": "#48d848",
    "Heist":     "#f0a830",
    "Bounty":    "#24d6ff",
    "Knockout":  "#ff4444",
    "Hot Zone":  "#ff8800",
}

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

def encode_brawler(brawler: str, map_name: str = "") -> float:
    """Map-specific target encode, falling back to global win rate."""
    key = f"{map_name.upper()}|{brawler.upper()}"
    if key in brawler_map_win_rates:
        return brawler_map_win_rates[key]
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
        row[f"team_a_brawler{i}"] = encode_brawler(b, map_name)
    for i, b in enumerate(enemy_picks_3, 1):
        row[f"team_b_brawler{i}"] = encode_brawler(b, map_name)
    return row


# ─────────────────────────────────────────────
# Recommendation engine
# ─────────────────────────────────────────────

def recommend(map_name: str, my_team: list, enemy_team: list, top_n: int = None, n_samples: int = 30) -> pd.DataFrame:
    """
    Score each candidate brawler.

    First pick (both teams empty): rank directly by map-specific win rate from
    training data — no model needed, cleanest signal available.

    All other picks: average model win-probability over n_samples random
    completions of the unknown draft slots.
    """
    picked     = set(b.upper() for b in my_team + enemy_team)
    candidates = [b for b in all_brawlers if b not in picked]

    # ── First pick shortcut ───────────────────────────────────────────────────
    if not my_team and not enemy_team:
        scores = []
        for c in candidates:
            key = f"{map_name.upper()}|{c.upper()}"
            wr  = brawler_map_win_rates.get(key, brawler_win_rates.get(c, brawler_global_mean))
            scores.append({"brawler": c, "win_prob": wr})
        df = pd.DataFrame(scores).sort_values("win_prob", ascending=False).reset_index(drop=True)
        return df if top_n is None else df.head(top_n)

    # ── Model-based scoring ───────────────────────────────────────────────────
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

    X     = pd.DataFrame(all_rows)[feature_cols]
    probs = model.predict_proba(X)[:, 1]

    scores = [
        {"brawler": c, "win_prob": float(probs[i * n_samples:(i + 1) * n_samples].mean())}
        for i, c in enumerate(candidates)
    ]

    df = pd.DataFrame(scores).sort_values("win_prob", ascending=False).reset_index(drop=True)
    return df if top_n is None else df.head(top_n)

# ─────────────────────────────────────────────
# Theme
# ─────────────────────────────────────────────

st.markdown("""
<style>
.stApp { background: #0d1b3e !important; }
.main .block-container { background: transparent !important; padding-top: 0.5rem; }
[data-testid="stSidebar"] { background: #0a1530 !important; }
[data-testid="stSidebar"] p, [data-testid="stSidebar"] label { color: rgba(255,255,255,0.85) !important; }
h1, h2, h3 { color: #FFE135 !important; text-shadow: 2px 2px 6px rgba(0,0,0,0.8); letter-spacing: 1px; }
p, .stMarkdown p { color: rgba(255,255,255,0.9) !important; }
.stCaption p { color: rgba(255,255,255,0.5) !important; }
.stMetric label { color: rgba(255,255,255,0.7) !important; }
.stMetric [data-testid="stMetricValue"] { color: #FFE135 !important; }
.stButton > button[kind="primary"] {
    background: #FFE135 !important; color: #0d1b3e !important;
    border: none !important; font-weight: bold; border-radius: 8px;
}
.stButton > button {
    background: #1e3a7a !important; color: white !important;
    border: 1px solid #3a5a9a !important; border-radius: 8px;
}
.stSelectbox label { color: white !important; }
[data-baseweb="select"] { background: #1a2a5e !important; border-color: #3a5a9a !important; }
.stDivider { border-color: rgba(255,227,53,0.3) !important; }
.stInfo    { background: rgba(255,255,255,0.05) !important; border: 1px solid rgba(255,255,255,0.1) !important; }
.stSuccess { background: rgba(50,200,50,0.1) !important; }
.stWarning { background: rgba(255,150,0,0.1) !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HTML helpers
# ─────────────────────────────────────────────

def _slot_html(brawler, is_next=False, team="A"):
    border = "#FFE135" if is_next else ("#5b9bd5" if team == "A" else "#d55b5b")
    dash   = "solid" if is_next else "dashed"
    if brawler:
        url = brawler_icons.get(brawler.upper(), "")
        if url:
            img = f'<img src="{url}" style="width:80px;height:80px;border-radius:50%;border:3px solid {border};object-fit:cover;display:block;">'
        else:
            img = f'<div style="width:80px;height:80px;border-radius:50%;border:3px solid {border};background:#1a2a5e;display:flex;align-items:center;justify-content:center;"><span style="color:white;font-size:9px;text-align:center;">{brawler.title()}</span></div>'
        return f'<div style="text-align:center;display:inline-block;margin:0 8px;">{img}<div style="color:white;font-size:11px;margin-top:4px;">{brawler.title()}</div></div>'
    else:
        bc    = "#FFE135" if is_next else "rgba(255,255,255,0.25)"
        inner = "▼" if is_next else "?"
        label = "NOW" if is_next else "EMPTY"
        return f'<div style="text-align:center;display:inline-block;margin:0 8px;"><div style="width:80px;height:80px;border-radius:50%;border:3px {dash} {bc};background:rgba(255,255,255,0.04);display:flex;align-items:center;justify-content:center;"><span style="color:{bc};font-size:22px;">{inner}</span></div><div style="color:rgba(255,255,255,0.35);font-size:10px;margin-top:4px;">{label}</div></div>'

def _draft_board_html(draft, next_slot):
    # Snake order: A=0,3,4  B=1,2,5
    blue_html = "".join(_slot_html(draft[i], i == next_slot, "A") for i in [0, 3, 4])
    red_html  = "".join(_slot_html(draft[i], i == next_slot, "B") for i in [1, 2, 5])
    return f"""
    <div style="display:flex;align-items:center;gap:12px;margin:12px 0;">
        <div style="flex:1;background:rgba(30,90,180,0.2);border:2px solid rgba(91,155,213,0.5);border-radius:14px;padding:16px 12px;">
            <div style="color:#5b9bd5;font-size:13px;letter-spacing:3px;margin-bottom:10px;">BLUE TEAM</div>
            <div style="display:flex;justify-content:center;">{blue_html}</div>
        </div>
        <div style="padding:0 8px;flex-shrink:0;text-align:center;">
            <div style="color:#FFE135;font-size:42px;font-weight:bold;text-shadow:0 0 20px rgba(255,227,53,0.5);">VS</div>
        </div>
        <div style="flex:1;background:rgba(180,30,30,0.2);border:2px solid rgba(213,91,91,0.5);border-radius:14px;padding:16px 12px;">
            <div style="color:#d55b5b;font-size:13px;letter-spacing:3px;margin-bottom:10px;">RED TEAM</div>
            <div style="display:flex;justify-content:center;">{red_html}</div>
        </div>
    </div>
    """

def _rec_cards_html(recs):
    cards = ""
    for idx, (_, row) in enumerate(recs.iterrows()):
        b     = row["brawler"]
        url   = brawler_icons.get(b.upper(), "")
        name  = b.title()
        prob  = row["win_prob"]
        glow  = "box-shadow:0 0 14px rgba(255,227,53,0.8);" if idx == 0 else ""
        bord  = "#FFE135" if idx == 0 else "rgba(255,255,255,0.35)"
        if url:
            img_html = f'<img src="{url}" style="width:76px;height:76px;border-radius:50%;border:3px solid {bord};object-fit:cover;display:block;{glow}">'
        else:
            img_html = f'<div style="width:76px;height:76px;border-radius:50%;border:3px solid {bord};background:#1a2a5e;display:flex;align-items:center;justify-content:center;{glow}"><span style="color:white;font-size:9px;">{name}</span></div>'
        cards += f'<div style="display:inline-block;text-align:center;margin:6px 10px;vertical-align:top;">{img_html}<div style="color:white;font-size:11px;margin-top:4px;">{name}</div><div style="color:#FFE135;font-size:12px;font-weight:bold;">{prob:.1%}</div></div>'
    return f'<div style="text-align:center;padding:8px 0;">{cards}</div>'

# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────

st.markdown(f'<h1 style="text-align:center;margin-bottom:0;"><img src="data:image/png;base64,{_logo_b64}" style="height:48px;vertical-align:middle;margin-right:10px;">BRAWLPICK</h1>', unsafe_allow_html=True)
st.markdown(f'<p style="text-align:center;color:rgba(255,255,255,0.45);font-size:12px;margin-top:2px;">{meta["n_train"]:,} ranked matches · {meta["cv_accuracy_mean"]:.0%} CV accuracy</p>', unsafe_allow_html=True)

# ── Map picker / locked bar ────────────────────────────────────────────────

if st.session_state.map_locked:
    # Locked state: show selected map + action buttons
    mode_label  = map_mode.get(st.session_state.map_name.upper(), "")
    mode_color  = MODE_COLORS.get(mode_label, "#FFE135")
    mode_icon   = MODE_ICONS.get(mode_label, "🗺")
    img_url     = map_image_url.get(st.session_state.map_name, "")
    img_tag     = f'<img src="{img_url}" style="height:70px;border-radius:8px;object-fit:cover;margin-right:16px;">' if img_url else ""
    st.markdown(f"""
    <div style="display:flex;align-items:center;background:rgba(15,30,80,0.85);
                border:2px solid {mode_color};border-radius:12px;padding:10px 18px;margin-bottom:8px;">
        {img_tag}
        <div>
            <div style="color:{mode_color};font-size:11px;letter-spacing:2px;">{mode_icon} {mode_label.upper()}</div>
            <div style="color:#FFE135;font-size:20px;font-weight:bold;">{st.session_state.map_name}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    col_change, col_reset = st.columns([1, 1])
    with col_change:
        if st.button("🔓 Change Map", use_container_width=True):
            st.session_state.map_locked = False
            st.session_state.draft = [None] * 6
            st.session_state.rec_history = []
            st.rerun()
    with col_reset:
        if st.button("↺ Reset Draft", use_container_width=True):
            st.session_state.draft = [None] * 6
            st.session_state.rec_history = []
            st.rerun()

else:
    # Map picker grid grouped by mode
    # Add CSS for map card hover effect
    st.markdown("""
    <style>
    .map-card { cursor:pointer; transition:transform 0.15s; }
    .map-card:hover { transform:scale(1.05); }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<p style="color:rgba(255,255,255,0.6);font-size:13px;margin-bottom:4px;">SELECT A MAP TO START THE DRAFT</p>', unsafe_allow_html=True)

    # Group filtered maps by mode
    modes_order = ["Gem Grab", "Brawl Ball", "Heist", "Bounty", "Knockout", "Hot Zone"]
    for mode in modes_order:
        mode_maps = maps_df_filtered[maps_df_filtered["game_mode_name"] == mode]
        if mode_maps.empty:
            continue
        mode_color = MODE_COLORS.get(mode, "#ffffff")
        mode_icon  = MODE_ICONS.get(mode, "🗺")
        st.markdown(f'<div style="color:{mode_color};font-size:13px;letter-spacing:3px;margin:10px 0 6px;">{mode_icon} {mode.upper()}</div>', unsafe_allow_html=True)

        cols = st.columns(len(mode_maps))
        for col, (_, mrow) in zip(cols, mode_maps.iterrows()):
            map_name = mrow["map_name"]
            img_url  = map_image_url.get(map_name, "")
            n_games  = int(map_counts.get(map_name, 0))
            is_selected = (map_name == st.session_state.map_name)
            border = f"3px solid {mode_color}" if is_selected else "2px solid rgba(255,255,255,0.12)"
            bg     = f"rgba(255,255,255,0.08)" if is_selected else "rgba(255,255,255,0.02)"

            with col:
                if img_url:
                    st.markdown(f'<img src="{img_url}" style="width:100%;border-radius:8px;border:{border};display:block;">', unsafe_allow_html=True)
                st.markdown(f'<div style="color:{"#FFE135" if is_selected else "white"};font-size:11px;text-align:center;margin-top:4px;margin-bottom:2px;">{map_name}</div>', unsafe_allow_html=True)
                st.markdown(f'<div style="color:rgba(255,255,255,0.4);font-size:10px;text-align:center;margin-bottom:6px;">{n_games} games</div>', unsafe_allow_html=True)
                if st.button("Select", key=f"map_{map_name}", use_container_width=True,
                             type="primary" if is_selected else "secondary"):
                    st.session_state.map_name = map_name
                    st.session_state.map_locked = True
                    st.session_state.draft = [None] * 6
                    st.session_state.rec_history = []
                    st.rerun()

    st.stop()

# Draft board
next_slot = next_pick_slot()
st.markdown(_draft_board_html(st.session_state.draft, next_slot), unsafe_allow_html=True)
st.divider()

# ─────────────────────────────────────────────
# Pick section
# ─────────────────────────────────────────────

if next_slot is not None:
    team_picking   = DRAFT_ORDER[next_slot]
    already_picked = set(b.upper() for b in picked_brawlers())
    available      = [b for b in all_brawlers if b not in already_picked]

    if team_picking == "A":
        st.markdown('<h3 style="text-align:center;letter-spacing:3px;">PICK YOUR BRAWLER</h3>', unsafe_allow_html=True)
        my_team, enemy_team = get_teams()

        with st.spinner("Calculating recommendations..."):
            recs = recommend(st.session_state.map_name, my_team, enemy_team)

        st.markdown(_rec_cards_html(recs.head(10)), unsafe_allow_html=True)

        with st.expander("Full rankings"):
            map_upper = st.session_state.map_name.upper()
            rows = []
            for _, row in recs.iterrows():
                b   = row["brawler"]
                key = f"{map_upper}|{b.upper()}"
                map_wr    = brawler_map_win_rates.get(key)
                map_games = brawler_map_game_counts.get(key)
                global_wr = brawler_win_rates.get(b, brawler_global_mean)
                rows.append({
                    "Brawler":           b.title(),
                    "Map Win Rate":      (f"{map_wr:.1%} ({map_games} games)" if map_games is not None else f"{map_wr:.1%}") if map_wr is not None else "— no map data",
                    "Global Win Rate":   f"{global_wr:.1%}",
                })
            table = pd.DataFrame(rows)
            table.index = range(1, len(table) + 1)
            st.dataframe(table, use_container_width=True)
        st.write("")

        col_pick, col_btn = st.columns([3, 1])
        with col_pick:
            chosen = st.selectbox(
                "Confirm your pick:",
                options=[""] + available,
                format_func=lambda x: x.title() if x else "— Select brawler —",
                key=f"pick_{next_slot}",
            )
        with col_btn:
            st.write("")
            st.write("")
            if chosen:
                if st.button(f"Lock In {chosen.title()}", type="primary", use_container_width=True):
                    st.session_state.rec_history.append({
                        "pick_num": next_slot + 1,
                        "recs":     recs.copy(),
                        "chosen":   chosen,
                    })
                    st.session_state.draft[next_slot] = chosen
                    st.rerun()

    else:
        st.markdown('<h3 style="text-align:center;color:#d55b5b;letter-spacing:3px;">ENEMY PICK</h3>', unsafe_allow_html=True)
        col_pick, col_btn = st.columns([3, 1])
        with col_pick:
            chosen = st.selectbox(
                "Enemy picked:",
                options=[""] + available,
                format_func=lambda x: x.title() if x else "— Select brawler —",
                key=f"pick_{next_slot}",
            )
        with col_btn:
            st.write("")
            st.write("")
            if chosen:
                if st.button("Lock In (enemy)", type="primary", use_container_width=True):
                    st.session_state.draft[next_slot] = chosen
                    st.rerun()

else:
    # Draft complete
    st.markdown('<h2 style="text-align:center;">✅ DRAFT COMPLETE</h2>', unsafe_allow_html=True)
    my_team, enemy_team = get_teams()

    my_avg    = sum(encode_brawler(b) for b in my_team)    / len(my_team)
    enemy_avg = sum(encode_brawler(b) for b in enemy_team) / len(enemy_team)
    edge      = my_avg - enemy_avg

    cola, colb = st.columns(2)
    with cola:
        blue_slots = "".join(_slot_html(b, False, "A") for b in my_team)
        st.markdown(f'<div style="color:#5b9bd5;font-size:13px;letter-spacing:3px;margin-bottom:8px;">YOUR TEAM</div><div style="display:flex;justify-content:center;background:rgba(30,90,180,0.2);border:2px solid rgba(91,155,213,0.5);border-radius:14px;padding:16px;">{blue_slots}</div>', unsafe_allow_html=True)
        st.metric("Avg Win Rate", f"{my_avg:.1%}")
    with colb:
        red_slots = "".join(_slot_html(b, False, "B") for b in enemy_team)
        st.markdown(f'<div style="color:#d55b5b;font-size:13px;letter-spacing:3px;margin-bottom:8px;">ENEMY TEAM</div><div style="display:flex;justify-content:center;background:rgba(180,30,30,0.2);border:2px solid rgba(213,91,91,0.5);border-radius:14px;padding:16px;">{red_slots}</div>', unsafe_allow_html=True)
        st.metric("Avg Win Rate", f"{enemy_avg:.1%}")

    st.divider()
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
                c0.markdown(f"**{int(row.name)+1}**" if is_chosen else str(int(row.name)+1))
                c1.markdown(f"**{row['brawler'].title()} ✓**" if is_chosen else row["brawler"].title())
                c2.markdown(f"**{row['win_prob']:.1%}**" if is_chosen else f"{row['win_prob']:.1%}")

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
