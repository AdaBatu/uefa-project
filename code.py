# %%
import json
import os
from math import pi

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Arc, Rectangle

try:
    from mplsoccer import Pitch
except ModuleNotFoundError:
    Pitch = None


# =========================
# CONFIG
# =========================

BASE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "uefa-cl-2025-2026",
    "data",
)

MATCHES_PATH = os.path.join(BASE_DIR, "matches.csv")

MATCH_DIRS = [
    os.path.join(BASE_DIR, "statsbomb", "league_phase"),
    os.path.join(BASE_DIR, "statsbomb", "last16"),
    os.path.join(BASE_DIR, "statsbomb", "playoffs"),
    os.path.join(BASE_DIR, "statsbomb", "quarterfinals"),
    os.path.join(BASE_DIR, "statsbomb", "semifinals"),
]

TARGET_TEAM = "Napoli"
MIN_TEAM_MATCHES = 1
PITCH_LENGTH = 120
PITCH_WIDTH = 80

df_matches = pd.read_csv(MATCHES_PATH)


# =========================
# MATCH HELPERS
# =========================

def get_match_path(match_id):
    filename = f"{match_id}.json"

    for match_dir in MATCH_DIRS:
        path = os.path.join(match_dir, filename)
        if os.path.exists(path):
            return path

    raise FileNotFoundError(f"{filename} not found.")


def get_statsbomb_ids():
    match_ids = df_matches["statsbomb"].dropna().astype(int).tolist()
    return [
        match_id
        for match_id in match_ids
        if any(
            os.path.exists(os.path.join(match_dir, f"{match_id}.json"))
            for match_dir in MATCH_DIRS
        )
    ]


def load_match_data(match_id):
    path = get_match_path(match_id)
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def get_match_label(match_id):
    match_row = df_matches[df_matches["statsbomb"] == match_id].iloc[0]
    return f"{match_row['home']} {match_row['score']} {match_row['away']}"


# =========================
# TEAM ATTACK DATA
# =========================

def empty_team_stats(team_name):
    return {
        "Team": team_name,
        "Match IDs": set(),
        "Events": 0,
        "Shots": 0,
        "Goals": 0,
        "xG": 0.0,
        "Open Play xG": 0.0,
        "Passes": 0,
        "Completed Passes": 0,
        "Progressive Passes": 0,
        "Passes Into Box": 0,
        "Passes Into Final Third": 0,
        "Carries": 0,
        "Progressive Carries": 0,
        "Touches Final Third": 0,
        "Touches Box": 0,
        "Build-up Involvements": 0,
        "Pressures": 0,
        "OBV Total": 0.0,
    }


def is_box_location(x, y):
    return x >= 102 and 18 <= y <= 62


def collect_team_attack_stats():
    teams = {}

    for match_id in get_statsbomb_ids():
        match_data = load_match_data(match_id)

        for event in match_data:
            team = event.get("team", {}).get("name")
            if not team:
                continue

            if team not in teams:
                teams[team] = empty_team_stats(team)

            stats = teams[team]
            stats["Match IDs"].add(match_id)
            stats["Events"] += 1
            stats["OBV Total"] += event.get("obv_total_net") or 0

            event_type = event.get("type", {}).get("name")
            x, y = None, None

            if "location" in event:
                x, y = event["location"][:2]

                if x >= 80:
                    stats["Touches Final Third"] += 1

                if is_box_location(x, y):
                    stats["Touches Box"] += 1

            if event_type in ["Pass", "Carry", "Ball Receipt*"]:
                stats["Build-up Involvements"] += 1

            if event_type == "Pressure":
                stats["Pressures"] += 1

            if event_type == "Shot":
                shot = event.get("shot", {})
                outcome = shot.get("outcome", {}).get("name", "Unknown")
                xg = shot.get("statsbomb_xg", 0) or 0
                shot_type = shot.get("type", {}).get("name", "Open Play")

                stats["Shots"] += 1
                stats["xG"] += xg

                if shot_type == "Open Play":
                    stats["Open Play xG"] += xg

                if outcome == "Goal":
                    stats["Goals"] += 1

            if event_type == "Pass":
                pass_data = event.get("pass", {})
                end_location = pass_data.get("end_location")

                if end_location and x is not None:
                    end_x, end_y = end_location[:2]
                    completed = "outcome" not in pass_data

                    stats["Passes"] += 1

                    if completed:
                        stats["Completed Passes"] += 1

                    if completed and end_x - x >= 10:
                        stats["Progressive Passes"] += 1

                    if completed and is_box_location(end_x, end_y):
                        stats["Passes Into Box"] += 1

                    if completed and x < 80 and end_x >= 80:
                        stats["Passes Into Final Third"] += 1

            if event_type == "Carry":
                carry_data = event.get("carry", {})
                end_location = carry_data.get("end_location")

                if end_location and x is not None:
                    end_x = end_location[0]

                    stats["Carries"] += 1

                    if end_x - x >= 10:
                        stats["Progressive Carries"] += 1

    rows = []
    for stats in teams.values():
        row = stats.copy()
        row["Matches"] = len(row.pop("Match IDs"))
        rows.append(row)

    return pd.DataFrame(rows)


def add_team_rates(df):
    df = df.copy()

    per_match_cols = [
        "Shots",
        "Goals",
        "xG",
        "Open Play xG",
        "Progressive Passes",
        "Passes Into Box",
        "Passes Into Final Third",
        "Carries",
        "Progressive Carries",
        "Touches Final Third",
        "Touches Box",
        "Build-up Involvements",
        "Pressures",
        "OBV Total",
    ]

    for col in per_match_cols:
        df[f"{col} per Match"] = np.where(
            df["Matches"] > 0,
            df[col] / df["Matches"],
            0,
        )

    per_100_event_cols = [
        "Shots",
        "Goals",
        "xG",
        "Open Play xG",
        "Progressive Passes",
        "Passes Into Box",
        "Passes Into Final Third",
        "Progressive Carries",
        "Touches Final Third",
        "Touches Box",
        "Build-up Involvements",
        "Pressures",
        "OBV Total",
    ]

    for col in per_100_event_cols:
        df[f"{col} per 100 Events"] = np.where(
            df["Events"] > 0,
            df[col] / df["Events"] * 100,
            0,
        )

    df["Pass Completion %"] = np.where(
        df["Passes"] > 0,
        df["Completed Passes"] / df["Passes"] * 100,
        0,
    )

    df["xG per Shot"] = np.where(df["Shots"] > 0, df["xG"] / df["Shots"], 0)
    df["Goal Conversion %"] = np.where(df["Shots"] > 0, df["Goals"] / df["Shots"] * 100, 0)

    return df


def get_target_team_row(team_stats):
    team_rows = team_stats[team_stats["Team"].str.casefold() == TARGET_TEAM.casefold()]

    if team_rows.empty:
        available = ", ".join(sorted(team_stats["Team"].unique()))
        raise ValueError(f"{TARGET_TEAM} not found. Available teams: {available}")

    return team_rows.iloc[0]


def get_rank_table(team_stats):
    rank_cols = {
        "xG per Match": False,
        "Open Play xG per Match": False,
        "Shots per Match": False,
        "xG per Shot": False,
        "Touches Box per Match": False,
        "Passes Into Box per Match": False,
        "Progressive Passes per Match": False,
        "OBV Total per Match": False,
    }

    table = team_stats[["Team", "Matches"] + list(rank_cols)].copy()

    for col, ascending in rank_cols.items():
        table[f"{col} Rank"] = table[col].rank(method="min", ascending=ascending).astype(int)

    return table.sort_values("xG per Match", ascending=False)


# =========================
# SHOT DATA
# =========================

def collect_team_shots(team_name):
    rows = []

    for match_id in get_statsbomb_ids():
        match_data = load_match_data(match_id)
        match_label = get_match_label(match_id)

        for event in match_data:
            if event.get("team", {}).get("name") != team_name:
                continue

            if event.get("type", {}).get("name") != "Shot":
                continue

            if "location" not in event:
                continue

            shot = event.get("shot", {})
            x, y = event["location"][:2]
            outcome = shot.get("outcome", {}).get("name", "Unknown")

            rows.append({
                "Match ID": match_id,
                "Match": match_label,
                "Player": event.get("player", {}).get("name", "Unknown"),
                "minute": event.get("minute"),
                "second": event.get("second"),
                "x": x,
                "y": y,
                "xG": shot.get("statsbomb_xg", 0) or 0,
                "outcome": outcome,
                "body_part": shot.get("body_part", {}).get("name", "Unknown"),
                "shot_type": shot.get("type", {}).get("name", "Open Play"),
                "goal": outcome == "Goal",
                "on_target": outcome in ["Goal", "Saved", "Saved to Post"],
                "in_box": is_box_location(x, y),
            })

    return pd.DataFrame(rows)


def collect_team_events_df(team_name):
    frames = []

    for match_id in get_statsbomb_ids():
        match_data = load_match_data(match_id)
        df = pd.json_normalize(match_data, sep=".")
        df["match_id"] = match_id
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    events = pd.concat(frames, ignore_index=True)
    return events.loc[events["team.name"] == team_name].copy()


def split_locations(series):
    coords = series.apply(
        lambda value: value if isinstance(value, list) and len(value) >= 2 else [np.nan, np.nan]
    )
    return pd.DataFrame(coords.tolist(), columns=["x", "y"], index=series.index)


def collect_team_passes(team_name):
    events = collect_team_events_df(team_name)
    passes = events.loc[events["type.name"] == "Pass"].copy()

    if passes.empty:
        return passes

    starts = split_locations(passes["location"])
    ends = split_locations(passes["pass.end_location"])
    passes = passes.join(starts)
    passes = passes.join(ends.rename(columns={"x": "end_x", "y": "end_y"}))

    passes["completed"] = passes["pass.outcome.name"].isna()
    passes["outcome"] = np.where(passes["completed"], "Complete", passes["pass.outcome.name"])
    passes["under_pressure"] = passes["under_pressure"].eq(True)
    passes["length"] = pd.to_numeric(passes["pass.length"], errors="coerce")
    passes["pass_success_probability"] = pd.to_numeric(
        passes.get("pass.pass_success_probability", np.nan),
        errors="coerce",
    )
    passes["height"] = passes["pass.height.name"].fillna("Unknown")
    passes["type"] = passes["pass.type.name"].fillna("Regular")
    passes["into_final_third"] = (passes["x"] < 80) & (passes["end_x"] >= 80)
    passes["into_box"] = (passes["end_x"] >= 102) & (passes["end_y"].between(18, 62))
    passes["progressive"] = (passes["end_x"] - passes["x"]) >= 10
    passes["long_ball"] = passes["length"] >= 30
    passes["lofted_long_ball"] = passes["long_ball"] & (passes["height"] == "High Pass")
    passes["shot_assist"] = passes["pass.shot_assist"].eq(True) if "pass.shot_assist" in passes else False
    passes["goal_assist"] = passes["pass.goal_assist"].eq(True) if "pass.goal_assist" in passes else False

    return passes


def summarize_passing(passes):
    if passes.empty:
        return pd.DataFrame()

    total = len(passes)
    completed = int(passes["completed"].sum())
    under_pressure = passes.loc[passes["under_pressure"]]
    not_under_pressure = passes.loc[~passes["under_pressure"]]

    rows = [{
        "Passes": total,
        "Completed": completed,
        "Completion %": completed / total * 100,
        "Average Length": passes["length"].mean(),
        "Median Length": passes["length"].median(),
        "Progressive Passes": int(passes["progressive"].sum()),
        "Passes Into Final Third": int(passes["into_final_third"].sum()),
        "Passes Into Box": int(passes["into_box"].sum()),
        "Shot Assists": int(passes["shot_assist"].sum()),
        "Goal Assists": int(passes["goal_assist"].sum()),
        "Lofted Long Balls": int(passes["lofted_long_ball"].sum()),
        "Under Pressure Completion %": under_pressure["completed"].mean() * 100 if len(under_pressure) else 0,
        "No Pressure Completion %": not_under_pressure["completed"].mean() * 100 if len(not_under_pressure) else 0,
    }]

    return pd.DataFrame(rows).round(2)


def passing_category_table(passes):
    if passes.empty:
        return pd.DataFrame()

    rows = []
    for category, column in [
        ("Height", "height"),
        ("Type", "type"),
        ("Outcome", "outcome"),
    ]:
        grouped = passes.groupby(column)["completed"].agg(["size", "mean"]).reset_index()
        grouped.columns = ["Value", "Passes", "Completion Rate"]
        grouped["Category"] = category
        grouped["Completion %"] = grouped["Completion Rate"] * 100
        rows.append(grouped[["Category", "Value", "Passes", "Completion %"]])

    return pd.concat(rows, ignore_index=True).sort_values(
        ["Category", "Passes"],
        ascending=[True, False],
    ).round(2)


def passing_loss_table(events, passes):
    if events.empty:
        return pd.DataFrame()

    dispossessed = events["type.name"].eq("Dispossessed")
    miscontrol = events["type.name"].eq("Miscontrol")
    dribble_incomplete = events["type.name"].eq("Dribble") & events.get(
        "dribble.outcome.name", pd.Series(index=events.index, dtype=object)
    ).eq("Incomplete")
    receipt_incomplete = events["type.name"].eq("Ball Receipt*") & events.get(
        "ball_receipt.outcome.name", pd.Series(index=events.index, dtype=object)
    ).eq("Incomplete")
    bad_passes = passes.loc[~passes["completed"]]

    rows = [
        ("Ball losses", len(bad_passes) + dispossessed.sum() + miscontrol.sum() + dribble_incomplete.sum() + receipt_incomplete.sum()),
        ("Unintentional losses", dispossessed.sum() + miscontrol.sum() + dribble_incomplete.sum() + receipt_incomplete.sum()),
        ("Turnovers", dispossessed.sum()),
        ("Bad passes", len(bad_passes)),
        ("Bad passes successful", int((passes["completed"] & (passes["pass_success_probability"] < 0.5)).sum())),
        ("Deliberate losses", events["type.name"].eq("Clearance").sum() + passes["type"].isin(["Free Kick", "Goal Kick", "Throw-in"]).sum()),
        ("Shots", events["type.name"].eq("Shot").sum()),
        ("Clearances", events["type.name"].eq("Clearance").sum()),
        ("Fouls committed", events["type.name"].eq("Foul Committed").sum()),
        ("Offside passes", passes["outcome"].eq("Pass Offside").sum()),
        ("Lofted long balls", passes["lofted_long_ball"].sum()),
    ]

    return pd.DataFrame(rows, columns=["Metric", "Count"]).sort_values("Count", ascending=False)


def draw_pitch_on_axis(ax):
    field_color = "#7ab66f"
    line_color = "#ffffff"
    ax.add_patch(Rectangle((0, 0), 120, 80, facecolor=field_color, edgecolor="none", zorder=0))
    line_kwargs = {"color": line_color, "linewidth": 2.2, "zorder": 2}
    ax.plot([0, 120, 120, 0, 0], [0, 0, 80, 80, 0], **line_kwargs)
    ax.plot([60, 60], [0, 80], **line_kwargs)
    ax.add_patch(plt.Circle((60, 40), 10, fill=False, **line_kwargs))
    ax.plot([0, 18, 18, 0], [18, 18, 62, 62], **line_kwargs)
    ax.plot([102, 120, 120, 102, 102], [18, 18, 62, 62, 18], **line_kwargs)
    ax.plot([0, 6, 6, 0], [30, 30, 50, 50], **line_kwargs)
    ax.plot([114, 120, 120, 114, 114], [30, 30, 50, 50, 30], **line_kwargs)
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 80)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_pass_length_outcome(passes, team_name):
    if passes.empty:
        raise ValueError("No passes available.")

    fig, ax = plt.subplots(figsize=(10, 5))
    complete = passes.loc[passes["completed"], "length"].dropna()
    incomplete = passes.loc[~passes["completed"], "length"].dropna()
    ax.hist([complete, incomplete], bins=18, label=["Complete", "Incomplete"], color=["#2ca02c", "#d62728"], alpha=0.78)
    ax.set_title(f"{team_name} pass length and outcome")
    ax.set_xlabel("Pass length")
    ax.set_ylabel("Passes")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.show()
    return fig, ax


def plot_pass_categories(passes, team_name):
    category_df = passing_category_table(passes)
    plot_df = category_df.loc[category_df["Category"].isin(["Height", "Type"])].copy()
    plot_df = plot_df.sort_values(["Category", "Passes"], ascending=[True, True])

    labels = plot_df["Category"] + ": " + plot_df["Value"].astype(str)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(labels, plot_df["Passes"], color="#4c78a8")
    ax.set_title(f"{team_name} pass categories")
    ax.set_xlabel("Passes")
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.show()
    return fig, ax


def plot_passing_volume_efficiency(passes, team_name):
    groups = {
        "All": passes,
        "Progressive": passes.loc[passes["progressive"]],
        "Into final third": passes.loc[passes["into_final_third"]],
        "Into box": passes.loc[passes["into_box"]],
        "Long balls": passes.loc[passes["long_ball"]],
        "Lofted long balls": passes.loc[passes["lofted_long_ball"]],
    }
    rows = []
    for label, df in groups.items():
        rows.append({
            "Category": label,
            "Passes": len(df),
            "Completion %": df["completed"].mean() * 100 if len(df) else 0,
        })
    plot_df = pd.DataFrame(rows)

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()
    ax1.bar(plot_df["Category"], plot_df["Passes"], color="#4c78a8", alpha=0.78, label="Volume")
    ax2.plot(plot_df["Category"], plot_df["Completion %"], color="#d62728", marker="o", linewidth=2.5, label="Completion %")
    ax1.set_title(f"{team_name} passing volume vs efficiency")
    ax1.set_ylabel("Passes")
    ax2.set_ylabel("Completion %")
    ax1.tick_params(axis="x", rotation=25)
    ax1.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.show()
    return fig, (ax1, ax2)


def plot_pressure_pass_success(passes, team_name):
    plot_df = passes.groupby("under_pressure")["completed"].agg(["size", "mean"]).reset_index()
    plot_df["Pressure"] = np.where(plot_df["under_pressure"], "With pressure", "No pressure")
    plot_df["Completion %"] = plot_df["mean"] * 100

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(plot_df["Pressure"], plot_df["Completion %"], color=["#2ca02c", "#ff7f0e"])
    ax.set_title(f"{team_name} pass success under pressure")
    ax.set_ylabel("Completion %")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25)
    for bar, count in zip(bars, plot_df["size"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"n={count}", ha="center")
    plt.tight_layout()
    plt.show()
    return fig, ax


def _pitch_grid(values_x, values_y, bins=(12, 8)):
    return np.histogram2d(
        values_x,
        values_y,
        bins=bins,
        range=[[0, 120], [0, 80]],
    )


def plot_pass_start_heatmap(passes, team_name, bins=(12, 8)):
    hist, x_edges, y_edges = _pitch_grid(passes["x"].dropna(), passes["y"].dropna(), bins=bins)
    fig, ax = plt.subplots(figsize=(12, 8))
    draw_pitch_on_axis(ax)
    mesh = ax.pcolormesh(x_edges, y_edges, hist.T, cmap="Blues", alpha=0.72, shading="flat", zorder=1)
    fig.colorbar(mesh, ax=ax, fraction=0.03, pad=0.02, label="Pass volume")
    ax.set_title(f"{team_name} pass volume heatmap")
    plt.tight_layout()
    plt.show()
    return fig, ax


def plot_pass_efficiency_heatmap(passes, team_name, bins=(12, 8), min_passes=5):
    attempts, x_edges, y_edges = _pitch_grid(passes["x"].dropna(), passes["y"].dropna(), bins=bins)
    completed = passes.loc[passes["completed"]]
    made, _, _ = _pitch_grid(completed["x"].dropna(), completed["y"].dropna(), bins=bins)
    efficiency = np.full_like(attempts, np.nan, dtype=float)
    np.divide(made, attempts, out=efficiency, where=attempts >= min_passes)

    fig, ax = plt.subplots(figsize=(12, 8))
    draw_pitch_on_axis(ax)
    mesh = ax.pcolormesh(x_edges, y_edges, (efficiency * 100).T, cmap="RdYlGn", vmin=50, vmax=100, alpha=0.75, shading="flat", zorder=1)
    fig.colorbar(mesh, ax=ax, fraction=0.03, pad=0.02, label=f"Completion % (min {min_passes} passes)")
    ax.set_title(f"{team_name} passing efficiency heatmap")
    plt.tight_layout()
    plt.show()
    return fig, ax


def collect_unintentional_losses(events, passes):
    dispossessed = events["type.name"].eq("Dispossessed")
    miscontrol = events["type.name"].eq("Miscontrol")
    dribble_outcome = events.get("dribble.outcome.name", pd.Series(index=events.index, dtype=object))
    receipt_outcome = events.get("ball_receipt.outcome.name", pd.Series(index=events.index, dtype=object))
    dribble_incomplete = events["type.name"].eq("Dribble") & dribble_outcome.eq("Incomplete")
    receipt_incomplete = events["type.name"].eq("Ball Receipt*") & receipt_outcome.eq("Incomplete")
    other_losses = events.loc[dispossessed | miscontrol | dribble_incomplete | receipt_incomplete].copy()
    bad_passes = passes.loc[~passes["completed"]].copy()
    return pd.concat([other_losses, bad_passes], ignore_index=True, sort=False)


def plot_unintentional_losses_heatmap(events, passes, team_name, bins=(12, 8)):
    losses = collect_unintentional_losses(events, passes)
    coords = split_locations(losses["location"])
    hist, x_edges, y_edges = _pitch_grid(coords["x"].dropna(), coords["y"].dropna(), bins=bins)

    fig, ax = plt.subplots(figsize=(12, 8))
    draw_pitch_on_axis(ax)
    mesh = ax.pcolormesh(x_edges, y_edges, hist.T, cmap="Reds", alpha=0.72, shading="flat", zorder=1)
    fig.colorbar(mesh, ax=ax, fraction=0.03, pad=0.02, label="Unintentional losses")
    ax.set_title(f"{team_name} unintentional losses heatmap")
    plt.tight_layout()
    plt.show()
    return fig, ax


def summarize_team_shooting(df_shots, team_name):
    if df_shots.empty:
        return {
            "Team": team_name,
            "Shots": 0,
            "Goals": 0,
            "xG": 0,
            "xG / Shot": 0,
            "Shot Accuracy %": "0.0%",
            "Conversion %": "0.0%",
            "Shots in Box": 0,
            "Box Shot %": "0.0%",
        }

    shots = len(df_shots)
    goals = int(df_shots["goal"].sum())
    xg = df_shots["xG"].sum()
    on_target = df_shots["on_target"].sum()
    in_box = df_shots["in_box"].sum()

    return {
        "Team": team_name,
        "Shots": shots,
        "Goals": goals,
        "xG": round(xg, 2),
        "xG / Shot": round(xg / shots, 3) if shots > 0 else 0,
        "Shot Accuracy %": f"{on_target / shots * 100:.1f}%",
        "Conversion %": f"{goals / shots * 100:.1f}%",
        "Shots in Box": int(in_box),
        "Box Shot %": f"{in_box / shots * 100:.1f}%",
    }


# =========================
# VISUAL STYLE
# =========================

def create_half_pitch():
    if Pitch is None:
        return None

    return Pitch(
        pitch_type="statsbomb",
        half=True,
        pitch_color="#cfe3c4",
        line_color="#ffffff",
        linewidth=2.2,
        goal_type="box",
        stripe=False,
    )


def draw_half_pitch_fallback(figsize=(10, 8)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.set_facecolor("white")
    ax.set_facecolor("#cfe3c4")

    # Attacking half of a StatsBomb 120x80 pitch.
    ax.plot([60, 120, 120, 60, 60], [0, 0, 80, 80, 0], color="white", linewidth=2.2, zorder=1)
    ax.plot([60, 60], [0, 80], color="white", linewidth=2.2, zorder=1)

    # Penalty area, six-yard box, goal, spot, and penalty arc.
    ax.plot([102, 120, 120, 102, 102], [18, 18, 62, 62, 18], color="white", linewidth=2.2, zorder=1)
    ax.plot([114, 120, 120, 114, 114], [30, 30, 50, 50, 30], color="white", linewidth=2.2, zorder=1)
    ax.plot([120, 120], [36, 44], color="white", linewidth=5, zorder=1)
    ax.scatter([108], [40], color="white", s=24, zorder=1)

    penalty_arc = Arc(
        (108, 40),
        width=20,
        height=20,
        theta1=128,
        theta2=232,
        color="white",
        linewidth=2.2,
        zorder=1,
    )
    ax.add_patch(penalty_arc)

    # Hint of the center circle at halfway.
    center_arc = Arc(
        (60, 40),
        width=20,
        height=20,
        theta1=-90,
        theta2=90,
        color="white",
        linewidth=2.2,
        zorder=1,
    )
    ax.add_patch(center_arc)

    ax.set_xlim(60, 121)
    ax.set_ylim(0, 80)
    ax.set_aspect("equal")
    ax.axis("off")

    return fig, ax


def draw_full_pitch_fallback(figsize=(12, 8)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.set_facecolor("white")

    field_color = "#7ab66f"
    line_color = "#ffffff"
    goal_color = "#111111"

    ax.add_patch(
        Rectangle(
            (0, 0),
            120,
            80,
            facecolor=field_color,
            edgecolor="none",
            zorder=0,
        )
    )

    line_kwargs = {"color": line_color, "linewidth": 2.6, "zorder": 2}
    goal_kwargs = {"color": goal_color, "linewidth": 2.8, "zorder": 3}

    # Full StatsBomb 120x80 pitch.
    ax.plot([0, 120, 120, 0, 0], [0, 0, 80, 80, 0], **line_kwargs)
    ax.plot([60, 60], [0, 80], **line_kwargs)

    # Center circle and spot.
    ax.add_patch(plt.Circle((60, 40), 10, fill=False, **line_kwargs))
    ax.scatter([60], [40], color="white", s=24, zorder=1)

    # Both penalty areas.
    ax.plot([0, 18, 18, 0], [18, 18, 62, 62], **line_kwargs)
    ax.plot([102, 120, 120, 102, 102], [18, 18, 62, 62, 18], **line_kwargs)

    # Both six-yard boxes.
    ax.plot([0, 6, 6, 0], [30, 30, 50, 50], **line_kwargs)
    ax.plot([114, 120, 120, 114, 114], [30, 30, 50, 50, 30], **line_kwargs)

    # Goal frames and penalty spots.
    ax.plot([0, -3, -3, 0], [36, 36, 44, 44], **goal_kwargs)
    ax.plot([120, 123, 123, 120], [36, 36, 44, 44], **goal_kwargs)
    ax.scatter([12, 108], [40, 40], color=line_color, s=28, zorder=2)

    # Penalty arcs.
    ax.add_patch(Arc((12, 40), width=20, height=20, theta1=-52, theta2=52, **line_kwargs))
    ax.add_patch(Arc((108, 40), width=20, height=20, theta1=128, theta2=232, **line_kwargs))

    ax.set_xlim(-5, 125)
    ax.set_ylim(-3, 83)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    return fig, ax


def draw_rotated_attacking_half_pitch(figsize=(8, 10)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.set_facecolor("white")

    field_color = "#7ab66f"
    line_color = "#ffffff"
    goal_color = "#111111"

    # Rotated attacking half: x is original pitch width, y is distance from goal.
    ax.add_patch(
        Rectangle(
            (0, 0),
            80,
            60,
            facecolor=field_color,
            edgecolor="none",
            zorder=0,
        )
    )

    line_kwargs = {"color": line_color, "linewidth": 2.6, "zorder": 2}
    goal_kwargs = {"color": goal_color, "linewidth": 2.8, "zorder": 3}

    ax.plot([0, 80, 80, 0, 0], [0, 0, 60, 60, 0], **line_kwargs)
    ax.plot([0, 80], [60, 60], **line_kwargs)

    # Penalty area and six-yard box after rotation.
    ax.plot([18, 62, 62, 18, 18], [0, 0, 18, 18, 0], **line_kwargs)
    ax.plot([30, 50, 50, 30, 30], [0, 0, 6, 6, 0], **line_kwargs)

    # Goal frame, penalty spot, and penalty arc.
    ax.plot([36, 36, 44, 44], [0, -3, -3, 0], **goal_kwargs)
    ax.scatter([40], [12], color=line_color, s=28, zorder=2)
    ax.add_patch(Arc((40, 12), width=20, height=20, theta1=37, theta2=143, **line_kwargs))

    # Hint of the center circle at halfway.
    ax.add_patch(Arc((40, 60), width=20, height=20, theta1=180, theta2=360, **line_kwargs))

    ax.set_xlim(-2, 82)
    ax.set_ylim(-5, 62)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    return fig, ax


def rotate_shot_locations_for_attacking_half(df_shots):
    rotated = df_shots.copy()
    rotated["plot_x"] = rotated["y"]
    rotated["plot_y"] = PITCH_LENGTH - rotated["x"]
    return rotated


# =========================
# TEAM ATTACK VISUALS
# =========================

RADAR_METRICS = {
    "xG": "xG per Match",
    "Open Play xG": "Open Play xG per Match",
    "Shots": "Shots per Match",
    "xG / Shot": "xG per Shot",
    "Box Touches": "Touches Box per Match",
    "Box Passes": "Passes Into Box per Match",
    "Prog Passes": "Progressive Passes per Match",
    "Prog Carries": "Progressive Carries per Match",
    "Final 3rd": "Touches Final Third per Match",
    "OBV": "OBV Total per Match",
}


def percentile_rank(series, value):
    series = series.dropna()

    if series.empty:
        return 0

    return (series <= value).mean() * 100


def get_percentile_values(team_row, comparison_df):
    values = {}

    for label, column in RADAR_METRICS.items():
        values[label] = percentile_rank(comparison_df[column], team_row[column])

    return values


def get_average_percentile_values(comparison_df):
    values = {}

    for label, column in RADAR_METRICS.items():
        avg_value = comparison_df[column].mean()
        values[label] = percentile_rank(comparison_df[column], avg_value)

    return values


def plot_team_attack_radar(team_stats):
    team_row = get_target_team_row(team_stats)
    comparison_df = team_stats[
        (team_stats["Team"] != team_row["Team"]) &
        (team_stats["Matches"] >= MIN_TEAM_MATCHES)
    ].copy()

    team_values = get_percentile_values(team_row, comparison_df)
    avg_values = get_average_percentile_values(comparison_df)

    categories = list(team_values.keys())
    team_data = list(team_values.values()) + [list(team_values.values())[0]]
    avg_data = list(avg_values.values()) + [list(avg_values.values())[0]]

    angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 9), subplot_kw=dict(polar=True))

    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=8)

    ax.plot(angles, avg_data, linewidth=2.5, color="grey", label="Average other teams")
    ax.fill(angles, avg_data, color="grey", alpha=0.2)
    ax.plot(angles, team_data, linewidth=3, color="#1f77b4", label=team_row["Team"])
    ax.fill(angles, team_data, color="#1f77b4", alpha=0.25)

    for angle, value in zip(angles[:-1], team_data[:-1]):
        ax.text(
            angle,
            value + 5,
            f"{value:.0f}",
            ha="center",
            va="center",
            fontsize=9,
            color="#1f77b4",
            fontweight="bold",
        )

    ax.set_title(
        f"{team_row['Team']} Attack Percentiles\nvs Other Champions League Teams",
        fontsize=17,
        fontweight="bold",
        pad=24,
    )
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.18), fontsize=9)
    plt.tight_layout()
    plt.show()


def plot_attack_ranking_bars(team_stats):
    metrics = [
        "xG per Match",
        "Shots per Match",
        "Touches Box per Match",
        "Passes Into Box per Match",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    for ax, metric in zip(axes, metrics):
        plot_df = team_stats.sort_values(metric, ascending=False).head(12).copy()

        if TARGET_TEAM not in plot_df["Team"].values:
            target_row = team_stats[team_stats["Team"] == TARGET_TEAM]
            plot_df = pd.concat([plot_df.iloc[:-1], target_row], ignore_index=True)
            plot_df = plot_df.sort_values(metric, ascending=True)
        else:
            plot_df = plot_df.sort_values(metric, ascending=True)

        colors = ["#1f77b4" if team == TARGET_TEAM else "#9aa0a6" for team in plot_df["Team"]]

        ax.barh(plot_df["Team"], plot_df[metric], color=colors)
        ax.set_title(metric, fontsize=13, fontweight="bold")
        ax.grid(axis="x", alpha=0.25)

        for index, value in enumerate(plot_df[metric]):
            ax.text(value, index, f" {value:.2f}", va="center", fontsize=9)

    fig.suptitle(
        f"{TARGET_TEAM} Attack Compared With Other Teams",
        fontsize=20,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.show()


def plot_team_shot_map(df_shots, team_name):
    if df_shots.empty:
        print(f"No shot data for {team_name}.")
        return

    df_shots = rotate_shot_locations_for_attacking_half(df_shots)
    fig, ax = draw_rotated_attacking_half_pitch(figsize=(8, 10))

    goals = df_shots[df_shots["goal"] == True]
    non_goals = df_shots[df_shots["goal"] == False]

    ax.scatter(
        non_goals["plot_x"],
        non_goals["plot_y"],
        s=non_goals["xG"] * 900 + 80,
        color="black",
        edgecolors="white",
        linewidth=1,
        alpha=0.65,
        label="Shot",
        zorder=3,
    )

    ax.scatter(
        goals["plot_x"],
        goals["plot_y"],
        s=goals["xG"] * 900 + 120,
        color="#d62728",
        edgecolors="white",
        linewidth=1.5,
        alpha=0.9,
        label="Goal",
        zorder=4,
    )

    ax.set_title(
        f"{team_name} Shot Map\nRotated attacking half, circle size = xG",
        fontsize=18,
        fontweight="bold",
    )

    ax.legend(loc="lower left")
    plt.tight_layout()
    plt.show()
    return fig, ax


def plot_team_shooting_summary_table(df_shots, team_name):
    summary = summarize_team_shooting(df_shots, team_name)

    table_df = pd.DataFrame({
        "Metric": list(summary.keys())[1:],
        "Value": list(summary.values())[1:],
    })

    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.axis("off")

    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        cellLoc="left",
        colLoc="left",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.45)

    ax.set_title(
        f"{team_name} Shooting Summary",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    plt.tight_layout()
    plt.show()


# =========================
# RUN
# =========================

if __name__ == "__main__":
    print("StatsBomb match IDs:")
    print(get_statsbomb_ids())

    team_stats = collect_team_attack_stats()
    team_stats = team_stats[team_stats["Matches"] >= MIN_TEAM_MATCHES].copy()
    team_stats = add_team_rates(team_stats)

    rank_table = get_rank_table(team_stats)
    target_row = get_target_team_row(team_stats)

    print(f"\n{TARGET_TEAM} attacking profile:")
    profile_cols = [
        "Team",
        "Matches",
        "Shots",
        "Goals",
        "xG",
        "xG per Match",
        "Open Play xG per Match",
        "Shots per Match",
        "xG per Shot",
        "Touches Box per Match",
        "Passes Into Box per Match",
        "Progressive Passes per Match",
        "OBV Total per Match",
    ]
    print(target_row[profile_cols].to_frame().T.to_string(index=False))

    print("\nTeam attack ranking table:")
    display_cols = [
        "Team",
        "Matches",
        "xG per Match",
        "xG per Match Rank",
        "Shots per Match",
        "Shots per Match Rank",
        "Touches Box per Match",
        "Touches Box per Match Rank",
        "Passes Into Box per Match",
        "Passes Into Box per Match Rank",
        "OBV Total per Match",
        "OBV Total per Match Rank",
    ]
    print(rank_table[display_cols].round(2).to_string(index=False))

    napoli_shots = collect_team_shots(TARGET_TEAM)

    print(f"\n{TARGET_TEAM} shooting summary:")
    print(pd.DataFrame([summarize_team_shooting(napoli_shots, TARGET_TEAM)]).to_string(index=False))

    plot_team_attack_radar(team_stats)
    plot_attack_ranking_bars(team_stats)
    plot_team_shooting_summary_table(napoli_shots, TARGET_TEAM)
    plot_team_shot_map(napoli_shots, TARGET_TEAM)
