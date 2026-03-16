"""
Brawl Stars Data Collection - Steps 3 & 4
Step 3: Pull ranked leaderboard → top_players.csv
Step 4: BFS through player battle logs → battle_logs.csv (ranked matches only)

Run: python collect_battle_logs.py
Requires: BRAWLSTARS_API_KEY in .env
"""

import os
import time
import urllib.parse
from collections import deque
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("BRAWLSTARS_API_KEY", "").strip()
if not API_KEY:
    raise SystemExit("ERROR: BRAWLSTARS_API_KEY not found in .env")

BASE_URL = "https://api.brawlstars.com/v1"
OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Accept": "application/json",
}

# How many ranked match records to collect before stopping
TARGET_BATTLES = 10000

# Regions to pull leaderboards from (more = more seed players)
LEADERBOARD_REGIONS = ["global", "US", "KR", "JP", "GB"]
LEADERBOARD_LIMIT = 200  # max per region


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def api_get(path: str, params: dict = None) -> dict | None:
    """Make a GET request. Returns None on non-fatal errors (404, 429, 503)."""
    url = f"{BASE_URL}/{path}"
    try:
        resp = requests.get(url, headers=HEADERS, params=params, timeout=10)
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 429:
            print("    [rate limited] sleeping 5s...")
            time.sleep(5)
            return None
        if resp.status_code in (404, 503):
            return None
        print(f"    [HTTP {resp.status_code}] {url}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"    [request error] {e}")
        return None


def encode_tag(tag: str) -> str:
    """URL-encode a player tag: #ABC123 → %23ABC123"""
    return urllib.parse.quote(tag, safe="")


# ─────────────────────────────────────────────
# STEP 3: Leaderboard → seed player tags
# ─────────────────────────────────────────────

def collect_top_players() -> list[str]:
    print("\n=== STEP 3: Collecting Leaderboard Players ===")

    all_rows = []
    seen_tags = set()

    for region in LEADERBOARD_REGIONS:
        print(f"  Fetching {region} leaderboard...")
        data = api_get(f"rankings/{region}/players", {"limit": LEADERBOARD_LIMIT})
        if not data:
            print(f"    Skipped (no data)")
            continue

        players = data.get("items", [])
        print(f"    Got {len(players)} players")

        for p in players:
            tag = p.get("tag", "")
            if tag and tag not in seen_tags:
                seen_tags.add(tag)
                all_rows.append({
                    "player_tag":  tag,
                    "player_name": p.get("name"),
                    "trophies":    p.get("trophies"),
                    "rank":        p.get("rank"),
                    "region":      region,
                })

        time.sleep(0.5)

    df = pd.DataFrame(all_rows)
    out_path = OUTPUT_DIR / "top_players.csv"
    df.to_csv(out_path, index=False)
    print(f"\n  Saved {len(df)} unique players → {out_path}")

    return df["player_tag"].tolist()


# ─────────────────────────────────────────────
# STEP 4: Battle logs → ranked match records
# ─────────────────────────────────────────────

def parse_battle(raw: dict, fetched_for_tag: str) -> list[dict]:
    """
    Extract ranked match records from a single battle log entry.
    Returns a list of rows (empty list if not a ranked match).
    """
    event = raw.get("event", {})
    battle = raw.get("battle", {})

    # Only keep ranked matches in the 6 standard ranked game modes.
    # Strict whitelist prevents seasonal modes (Brawl Hockey, Wipeout 5v5, etc.)
    # from leaking in even when the API labels them type="ranked".
    RANKED_MODES = {"gemgrab", "brawlball", "heist", "bounty", "knockout", "hotzone"}

    battle_type = battle.get("type", "")
    battle_mode = battle.get("mode", "")

    if battle_type != "ranked":
        return []

    mode_clean = battle_mode.lower().replace(" ", "").replace("_", "")
    if mode_clean not in RANKED_MODES:
        return []

    map_name = event.get("map", "")
    mode = battle_mode
    battle_time = raw.get("battleTime", "")
    result = battle.get("result", "")  # "victory" / "defeat" / "draw"

    teams = battle.get("teams", [])
    if len(teams) != 2:
        return []

    # Flatten both teams into one row
    # Team 0 = the team that fetched_for_tag belongs to (or just team A)
    team_a = teams[0]
    team_b = teams[1]

    def extract_brawlers(team):
        return [
            p.get("brawler", {}).get("name", "UNKNOWN")
            for p in team
            if isinstance(p, dict)
        ]

    brawlers_a = extract_brawlers(team_a)
    brawlers_b = extract_brawlers(team_b)

    # Pad/truncate to 3 per team (standard ranked team size)
    def pad(lst, n=3):
        return (lst + [""] * n)[:n]

    brawlers_a = pad(brawlers_a)
    brawlers_b = pad(brawlers_b)

    # Determine which team fetched_for_tag is on
    team_a_tags = [p.get("tag", "") for p in team_a if isinstance(p, dict)]
    fetched_tag_clean = fetched_for_tag.lstrip("#")
    team_a_win = None
    if result == "victory":
        team_a_win = fetched_tag_clean in [t.lstrip("#") for t in team_a_tags]
    elif result == "defeat":
        team_a_win = fetched_tag_clean not in [t.lstrip("#") for t in team_a_tags]

    row = {
        "battle_time":      battle_time,
        "map":              map_name,
        "mode":             mode,
        "result_for_tag":   result,
        "team_a_win":       team_a_win,
        # Team A brawlers
        "team_a_brawler1":  brawlers_a[0],
        "team_a_brawler2":  brawlers_a[1],
        "team_a_brawler3":  brawlers_a[2],
        # Team B brawlers
        "team_b_brawler1":  brawlers_b[0],
        "team_b_brawler2":  brawlers_b[1],
        "team_b_brawler3":  brawlers_b[2],
        # Tags for deduplication
        "team_a_tags":      "|".join(team_a_tags),
        "team_b_tags":      "|".join([
            p.get("tag", "") for p in team_b if isinstance(p, dict)
        ]),
    }

    return [row]


def collect_battle_logs(seed_tags: list[str]) -> pd.DataFrame:
    print(f"\n=== STEP 4: Collecting Battle Logs (target: {TARGET_BATTLES} ranked matches) ===")

    all_rows = []
    seen_battle_keys = set()   # deduplicate battles
    visited_tags = set()       # don't re-fetch same player

    # BFS queue — start with seed players, expand via opponents
    queue = deque(seed_tags)

    while queue and len(all_rows) < TARGET_BATTLES:
        tag = queue.popleft()
        if tag in visited_tags:
            continue
        visited_tags.add(tag)

        encoded = encode_tag(tag)
        data = api_get(f"players/{encoded}/battlelog")
        time.sleep(0.3)

        if not data:
            continue

        battles = data.get("items", [])
        new_for_player = 0

        for raw in battles:
            rows = parse_battle(raw, tag)
            for row in rows:
                # Deduplicate: same two teams on same map at same time = same match
                key = (
                    row["battle_time"],
                    row["map"],
                    row["team_a_tags"],
                    row["team_b_tags"],
                )
                if key in seen_battle_keys:
                    continue
                seen_battle_keys.add(key)
                all_rows.append(row)
                new_for_player += 1

                # Add opponent tags to the BFS queue
                for opp_tag in row["team_b_tags"].split("|"):
                    if opp_tag and opp_tag not in visited_tags:
                        queue.append(opp_tag)

        if new_for_player > 0:
            print(
                f"  [{len(visited_tags):>5} visited | {len(all_rows):>6} battles] "
                f"{tag} → +{new_for_player} ranked"
            )

        # Save checkpoint every 500 battles so you don't lose progress
        if len(all_rows) % 500 < new_for_player and len(all_rows) > 0:
            _save_checkpoint(all_rows)

    df = pd.DataFrame(all_rows)
    out_path = OUTPUT_DIR / "battle_logs.csv"
    df.to_csv(out_path, index=False)
    print(f"\n  Done. Saved {len(df)} ranked battles → {out_path}")
    print(f"  Visited {len(visited_tags)} unique players")

    return df


def _save_checkpoint(rows: list[dict]):
    path = OUTPUT_DIR / "battle_logs_checkpoint.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"    [checkpoint] {len(rows)} battles saved to {path.name}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("Brawl Stars Data Collection — Steps 3 & 4")
    print("=" * 45)

    # Step 3 — leaderboard
    try:
        seed_tags = collect_top_players()
    except Exception as e:
        print(f"ERROR in Step 3: {e}")
        seed_tags = []

    if not seed_tags:
        raise SystemExit("No seed players found — check your API key and IP registration.")

    # Step 4 — battle logs
    try:
        df_battles = collect_battle_logs(seed_tags)
    except Exception as e:
        print(f"ERROR in Step 4: {e}")
        df_battles = None

    print("\n=== DONE ===")
    print("Files saved to: data/raw/")
    print("  top_players.csv      → leaderboard seed players")
    print("  battle_logs.csv      → ranked match records (training data)")
    print("\nNext step: run merge_training_data.py to build training_data.csv")
