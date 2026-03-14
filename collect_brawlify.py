"""
Brawl Stars Data Collection - Step 1 & 2
Pulls brawler metadata and map info from Brawlify API (no API key needed)
Run: python collect_brawlify.py
"""

import requests
import pandas as pd
import json
import time
from pathlib import Path

from dotenv import load_dotenv
import os

load_dotenv()  # or load_dotenv(".env.local") if you named it that
API_KEY = os.getenv("BRAWLSTARS_API_KEY")

BASE_URL = "https://api.brawlify.com/v1"
CDN_BASE = "https://cdn.brawlify.com"

OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


HEADERS = {
    "User-Agent": "BrawlStarsMLProject/1.0"
}

def get(endpoint: str) -> dict:
    """Make a GET request to the Brawlify API with basic error handling."""
    url = f"{BASE_URL}/{endpoint}"
    print(f"  Fetching: {url}")
    response = requests.get(url, headers=HEADERS, timeout=10)
    response.raise_for_status()
    time.sleep(0.3)  # be polite, don't hammer the API
    return response.json()


# ─────────────────────────────────────────────
# STEP 1: Brawler Metadata
# ─────────────────────────────────────────────

def collect_brawlers() -> pd.DataFrame:
    print("\n=== STEP 1: Collecting Brawler Metadata ===")

    data = get("brawlers")
    brawlers_raw = data.get("list", [])
    print(f"  Found {len(brawlers_raw)} brawlers")

    rows = []
    for b in brawlers_raw:
        rows.append({
            "brawler_id":       b.get("id"),
            "brawler_name":     b.get("name"),
            "rarity":           b.get("rarity", {}).get("name"),
            "rarity_color":     b.get("rarity", {}).get("color"),
            "class":            b.get("class", {}).get("name"),
            "description":      b.get("description"),
            "star_powers_count": len(b.get("starPowers", [])),
            "gadgets_count":    len(b.get("gadgets", [])),
            "icon_url":         f"{CDN_BASE}/brawlers/borderless/{b.get('id')}.png",
        })

    df = pd.DataFrame(rows)
    out_path = OUTPUT_DIR / "brawlers.csv"
    df.to_csv(out_path, index=False)
    print(f"  Saved {len(df)} brawlers → {out_path}")

    # Print a quick summary
    print(f"\n  Classes found:   {sorted(df['class'].dropna().unique().tolist())}")
    print(f"  Rarities found:  {sorted(df['rarity'].dropna().unique().tolist())}")

    return df


# ─────────────────────────────────────────────
# STEP 2: Maps & Game Modes
# ─────────────────────────────────────────────

def collect_maps() -> pd.DataFrame:
    print("\n=== STEP 2: Collecting Map Info ===")

    data = get("maps")
    maps_raw = data.get("list", [])
    print(f"  Found {len(maps_raw)} maps")

    rows = []
    for m in maps_raw:
        game_mode = m.get("gameMode", {})
        environment = m.get("environment", {})

        rows.append({
            "map_id":           m.get("id"),
            "map_name":         m.get("name"),
            "map_hash":         m.get("hash"),
            "game_mode_id":     game_mode.get("id"),
            "game_mode_name":   game_mode.get("name"),
            "game_mode_color":  game_mode.get("color"),
            "environment_id":   environment.get("id"),
            "environment_name": environment.get("name"),
            "disabled":         m.get("disabled", False),
            "image_url":        m.get("imageUrl"),
        })

    df = pd.DataFrame(rows)

    # Filter to only active (non-disabled) maps
    df_active = df[df["disabled"] == False].copy()

    out_path_all    = OUTPUT_DIR / "maps_all.csv"
    out_path_active = OUTPUT_DIR / "maps_active.csv"
    df.to_csv(out_path_all, index=False)
    df_active.to_csv(out_path_active, index=False)

    print(f"  Saved {len(df)} total maps         → {out_path_all}")
    print(f"  Saved {len(df_active)} active maps  → {out_path_active}")

    # Print breakdown by game mode
    print(f"\n  Active maps by game mode:")
    mode_counts = (
        df_active.groupby("game_mode_name")["map_name"]
        .count()
        .sort_values(ascending=False)
    )
    for mode, count in mode_counts.items():
        print(f"    {mode:<25} {count} maps")

    return df_active


# ─────────────────────────────────────────────
# STEP 2b: Current Event Rotation
# (bonus — tells you which maps are live right now)
# ─────────────────────────────────────────────

def collect_events() -> pd.DataFrame:
    print("\n=== STEP 2b: Current Event Rotation ===")

    data = get("events")
    active   = data.get("active", [])
    upcoming = data.get("upcoming", [])
    print(f"  Active events: {len(active)} | Upcoming: {len(upcoming)}")

    def parse_events(events, status):
        rows = []
        for e in events:
            slot = e.get("slot", {})
            m    = e.get("map") or {}
            gm   = m.get("gameMode", {}) if m else {}
            rows.append({
                "status":         status,
                "slot_id":        slot.get("id"),
                "slot_name":      slot.get("name"),
                "map_name":       m.get("name") if m else None,
                "game_mode":      gm.get("name"),
                "start_time":     e.get("startTime"),
                "end_time":       e.get("endTime"),
            })
        return rows

    rows = parse_events(active, "active") + parse_events(upcoming, "upcoming")
    df = pd.DataFrame(rows)

    out_path = OUTPUT_DIR / "events_current.csv"
    df.to_csv(out_path, index=False)
    print(f"  Saved {len(df)} events → {out_path}")

    print(f"\n  Currently active:")
    for _, row in df[df["status"] == "active"].iterrows():
        print(f"    [{row['game_mode']}] {row['map_name']}")

    return df


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("Brawl Stars Data Collection — Steps 1 & 2")
    print("=" * 45)

    try:
        df_brawlers = collect_brawlers()
    except Exception as e:
        print(f"  ERROR collecting brawlers: {e}")
        df_brawlers = None

    try:
        df_maps = collect_maps()
    except Exception as e:
        print(f"  ERROR collecting maps: {e}")
        df_maps = None

    try:
        df_events = collect_events()
    except Exception as e:
        print(f"  ERROR collecting events: {e}")
        df_events = None

    print("\n=== DONE ===")
    print("Files saved to: data/raw/")
    print("  brawlers.csv        → use as a lookup table when encoding brawler names")
    print("  maps_active.csv     → join on map_name when building training features")
    print("  events_current.csv  → tells you what's in rotation right now")
    print("\nNext step: get your Brawl Stars API key and run the battle log collector.")