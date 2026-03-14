"""Quick debug: shows what battle types/modes are actually in the API response."""
import os, urllib.parse, requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("BRAWLSTARS_API_KEY", "").strip()
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Accept": "application/json"}

# Grab one player from top_players.csv
import pandas as pd
df = pd.read_csv("data/raw/top_players.csv")
tag = df["player_tag"].iloc[0]
encoded = urllib.parse.quote(tag, safe="")

resp = requests.get(
    f"https://api.brawlstars.com/v1/players/{encoded}/battlelog",
    headers=HEADERS, timeout=10
)
print(f"Status: {resp.status_code}")
data = resp.json()

battles = data.get("items", [])
print(f"Total battles returned: {len(battles)}\n")

print(f"{'type':<20} {'mode':<25} {'map'}")
print("-" * 65)
for b in battles:
    battle = b.get("battle", {})
    event  = b.get("event", {})
    print(
        f"{battle.get('type',''):<20} "
        f"{battle.get('mode',''):<25} "
        f"{event.get('map','')}"
    )
