"""Quick diagnosis: check per-brawler win rates in training data."""
import pandas as pd

df = pd.read_csv("data/training_data.csv")

brawler_cols_a = ["team_a_brawler1", "team_a_brawler2", "team_a_brawler3"]
brawler_cols_b = ["team_b_brawler1", "team_b_brawler2", "team_b_brawler3"]

rows = []
for col in brawler_cols_a:
    tmp = df[[col, "team_a_win"]].rename(columns={col: "brawler", "team_a_win": "win"})
    rows.append(tmp)
for col in brawler_cols_b:
    tmp = df[[col, "team_a_win"]].copy()
    tmp["win"] = ~df["team_a_win"]  # invert — this team wins when A loses
    tmp = tmp[[col, "win"]].rename(columns={col: "brawler"})
    rows.append(tmp)

combined = pd.concat(rows)
combined = combined[combined["brawler"].notna() & (combined["brawler"] != "")]

stats = (
    combined.groupby("brawler")["win"]
    .agg(["mean", "count"])
    .rename(columns={"mean": "win_rate", "count": "appearances"})
    .sort_values("win_rate", ascending=False)
)

print("Top 20 brawlers by win rate in training data:")
print(stats.head(20).to_string())
print(f"\n8-BIT win rate: {stats.loc['8-BIT', 'win_rate']:.1%} ({stats.loc['8-BIT', 'appearances']} appearances)")
