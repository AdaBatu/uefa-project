from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.ndimage import gaussian_filter
except ImportError:
    gaussian_filter = None

PITCH_LENGTH = 120
PITCH_WIDTH = 80


def _default_data_dir() -> Path:
    return Path(__file__).resolve().parent / "data" / "statsbomb" / "league_phase"


def load_matches(matches_csv: Optional[str | Path] = None) -> pd.DataFrame:
    """Load matches.csv into a DataFrame."""
    if matches_csv is None:
        matches_csv = Path(__file__).resolve().parent / "data" / "matches.csv"
    return pd.read_csv(matches_csv)


def team_matches(team_name: str, matches_df: pd.DataFrame) -> pd.DataFrame:
    """Return matches where team appears as home or away."""
    mask = (matches_df["home"] == team_name) | (matches_df["away"] == team_name)
    return matches_df.loc[mask].copy()


def load_events(match_id: int | str, data_dir: Optional[str | Path] = None) -> list[dict]:
    """Load raw StatsBomb events JSON for a match."""
    data_dir = Path(data_dir) if data_dir else _default_data_dir()
    path = data_dir / f"{match_id}.json"
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_events_df(match_id: int | str, data_dir: Optional[str | Path] = None) -> pd.DataFrame:
    """Load and flatten StatsBomb events for a match."""
    events = load_events(match_id, data_dir=data_dir)
    return pd.json_normalize(events, sep=".")


def load_lineups(match_id: int | str, data_dir: Optional[str | Path] = None) -> list[dict]:
    """Load raw StatsBomb lineups JSON for a match."""
    data_dir = Path(data_dir) if data_dir else _default_data_dir()
    path = data_dir / f"{match_id}_lineups.json"
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_team_events(
    team_name: str, match_ids: Iterable[int | str], data_dir: Optional[str | Path] = None
) -> pd.DataFrame:
    """Load and combine events for a team across multiple matches."""
    frames: list[pd.DataFrame] = []
    for match_id in match_ids:
        df = load_events_df(match_id, data_dir=data_dir)
        df["match_id"] = match_id
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    events = pd.concat(frames, ignore_index=True)
    return events.loc[events["team.name"] == team_name].copy()


def load_matches_events(
    match_ids: Iterable[int | str], data_dir: Optional[str | Path] = None
) -> pd.DataFrame:
    """Load and combine events for multiple matches (all teams)."""
    frames: list[pd.DataFrame] = []
    for match_id in match_ids:
        df = load_events_df(match_id, data_dir=data_dir)
        df["match_id"] = match_id
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _split_locations(series: pd.Series) -> pd.DataFrame:
    coords = series.apply(
        lambda value: value if isinstance(value, list) and len(value) >= 2 else [np.nan, np.nan]
    )
    return pd.DataFrame(coords.tolist(), columns=["x", "y"], index=series.index)


def _flip_location_value(value: object) -> object:
    if isinstance(value, list) and len(value) >= 2:
        flipped = value.copy()
        flipped[0] = PITCH_LENGTH - flipped[0]
        return flipped
    return value


def _event_time_seconds(df: pd.DataFrame) -> pd.Series:
    offsets = df["period"].map({1: 0, 2: 45 * 60, 3: 90 * 60, 4: 105 * 60, 5: 120 * 60}).fillna(0)
    minutes = pd.to_numeric(df["minute"], errors="coerce").fillna(0)
    seconds = pd.to_numeric(df["second"], errors="coerce").fillna(0)
    return minutes * 60 + seconds + offsets


def _draw_pitch(
    ax: plt.Axes,
    team_name: Optional[str] = None,
    annotate: bool = False,
) -> None:
    ax.set_xlim(0, PITCH_LENGTH)
    ax.set_ylim(0, PITCH_WIDTH)
    ax.set_aspect("equal")
    ax.axis("off")

    # Pitch outline
    ax.plot([0, 0, PITCH_LENGTH, PITCH_LENGTH, 0], [0, PITCH_WIDTH, PITCH_WIDTH, 0, 0], color="black")
    ax.plot([PITCH_LENGTH / 2, PITCH_LENGTH / 2], [0, PITCH_WIDTH], color="black", linewidth=1)

    # Penalty areas
    box_x = 18
    box_y = 44
    ax.plot([0, box_x, box_x, 0], [PITCH_WIDTH / 2 - box_y / 2, PITCH_WIDTH / 2 - box_y / 2, PITCH_WIDTH / 2 + box_y / 2, PITCH_WIDTH / 2 + box_y / 2], color="black")
    ax.plot([PITCH_LENGTH, PITCH_LENGTH - box_x, PITCH_LENGTH - box_x, PITCH_LENGTH], [PITCH_WIDTH / 2 - box_y / 2, PITCH_WIDTH / 2 - box_y / 2, PITCH_WIDTH / 2 + box_y / 2, PITCH_WIDTH / 2 + box_y / 2], color="black")

    # Six-yard boxes
    six_x = 6
    six_y = 20
    ax.plot([0, six_x, six_x, 0], [PITCH_WIDTH / 2 - six_y / 2, PITCH_WIDTH / 2 - six_y / 2, PITCH_WIDTH / 2 + six_y / 2, PITCH_WIDTH / 2 + six_y / 2], color="black")
    ax.plot([PITCH_LENGTH, PITCH_LENGTH - six_x, PITCH_LENGTH - six_x, PITCH_LENGTH], [PITCH_WIDTH / 2 - six_y / 2, PITCH_WIDTH / 2 - six_y / 2, PITCH_WIDTH / 2 + six_y / 2, PITCH_WIDTH / 2 + six_y / 2], color="black")

    # Center circle
    center_circle = plt.Circle((PITCH_LENGTH / 2, PITCH_WIDTH / 2), 10, color="black", fill=False)
    ax.add_artist(center_circle)

    if annotate and team_name:
        ax.text(5, PITCH_WIDTH - 2, f"{team_name} attacking ->", fontsize=9, va="top")
        ax.text(5, 2, f"<- {team_name} defending", fontsize=9, va="bottom")


def _smoothed_histogram(
    x: pd.Series,
    y: pd.Series,
    bins: tuple[int, int] = (30, 20),
    sigma: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a 2D pitch histogram, optionally smoothed with scipy."""
    x_values = pd.to_numeric(x, errors="coerce").to_numpy()
    y_values = pd.to_numeric(y, errors="coerce").to_numpy()
    mask = np.isfinite(x_values) & np.isfinite(y_values)

    hist, x_edges, y_edges = np.histogram2d(
        x_values[mask],
        y_values[mask],
        bins=bins,
        range=[[0, PITCH_LENGTH], [0, PITCH_WIDTH]],
    )

    if gaussian_filter is not None and sigma > 0:
        hist = gaussian_filter(hist, sigma=sigma)

    return hist, x_edges, y_edges


def _plot_pitch_heatmap(
    values: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    team_name: str,
    title: str,
    colorbar_label: str,
    cmap: str = "Reds",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(10, 6))
    mesh = ax.imshow(
        values.T,
        origin="lower",
        extent=(x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        alpha=0.85,
        aspect="auto",
    )
    _draw_pitch(ax, team_name=team_name, annotate=True)
    fig.colorbar(mesh, ax=ax, fraction=0.03, pad=0.02, label=colorbar_label)
    ax.set_title(title)
    return fig, ax


def _safe_rate(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    rate = np.full_like(numerator, np.nan, dtype=float)
    np.divide(numerator, denominator, out=rate, where=denominator > 0)
    return rate


def _without_corner_events(events_df: pd.DataFrame) -> pd.DataFrame:
    """Remove events that are directly tagged as corner situations."""
    mask = pd.Series(False, index=events_df.index)

    if "play_pattern.name" in events_df.columns:
        mask = mask | (events_df["play_pattern.name"] == "From Corner")

    if "pass.type.name" in events_df.columns:
        mask = mask | (events_df["pass.type.name"] == "Corner")

    return events_df.loc[~mask].copy()


def orient_events_for_team(
    events_df: pd.DataFrame,
    team_name: str,
    matches_df: Optional[pd.DataFrame] = None,
    flip_away: bool = False,
) -> pd.DataFrame:
    """Optionally flip locations so team_name attacks left-to-right when away."""
    if events_df.empty or not flip_away:
        return events_df
    if "match_id" not in events_df.columns:
        raise ValueError("events_df must include match_id for orientation.")
    if matches_df is None:
        raise ValueError("matches_df is required when flip_away is True.")

    match_map = matches_df.set_index("statsbomb")[["home", "away"]].copy()
    match_map.index = match_map.index.astype(str)

    events = events_df.copy()
    match_ids = events["match_id"].astype(str)
    away_team = match_ids.map(match_map["away"])
    team_is_away = away_team == team_name

    flip_mask = (events["team.name"] == team_name) & (team_is_away.fillna(False))
    for col in ["location", "pass.end_location", "shot.end_location", "carry.end_location"]:
        if col in events.columns:
            events.loc[flip_mask, col] = events.loc[flip_mask, col].apply(_flip_location_value)

    return events


def ball_loss_events(events_df: pd.DataFrame, team_name: str) -> pd.DataFrame:
    team_df = events_df.loc[events_df["team.name"] == team_name].copy()

    dispossessed = team_df["type.name"] == "Dispossessed"
    miscontrol = team_df["type.name"] == "Miscontrol"
    dribble_incomplete = (team_df["type.name"] == "Dribble") & (
        team_df["dribble.outcome.name"] == "Incomplete"
    )
    pass_incomplete = (team_df["type.name"] == "Pass") & (
        team_df["pass.outcome.name"].notna()
    )
    ball_receipt_incomplete = (team_df["type.name"] == "Ball Receipt") & (
        team_df["ball_receipt.outcome.name"] == "Incomplete"
    )

    mask = dispossessed | miscontrol | dribble_incomplete | pass_incomplete | ball_receipt_incomplete
    return team_df.loc[mask].copy()


def defensive_actions(events_df: pd.DataFrame, team_name: str) -> pd.DataFrame:
    team_df = events_df.loc[events_df["team.name"] == team_name].copy()
    defensive_types = ["Tackle", "Interception", "Pressure", "Ball Recovery", "Clearance"]
    return team_df.loc[team_df["type.name"].isin(defensive_types)].copy()


def plot_ball_loss_heatmap(
    events_df: pd.DataFrame,
    team_name: str,
    title: Optional[str] = None,
    gridsize: int = 30,
) -> tuple[plt.Figure, plt.Axes]:
    df = ball_loss_events(events_df, team_name)
    if df.empty:
        raise ValueError("No ball loss events found for the selected team.")

    coords = _split_locations(df["location"])
    fig, ax = plt.subplots(figsize=(10, 6))
    _draw_pitch(ax, team_name=team_name, annotate=True)
    heat = ax.hexbin(
        coords["x"],
        coords["y"],
        gridsize=gridsize,
        extent=(0, PITCH_LENGTH, 0, PITCH_WIDTH),
        cmap="Reds",
        mincnt=1,
        alpha=0.8,
    )
    fig.colorbar(heat, ax=ax, fraction=0.03, pad=0.02, label="Ball losses")
    ax.set_title(title or f"{team_name} ball-loss heatmap")
    return fig, ax


def plot_defensive_action_map(
    events_df: pd.DataFrame,
    team_name: str,
    title: Optional[str] = None,
) -> tuple[plt.Figure, plt.Axes]:
    df = defensive_actions(events_df, team_name)
    if df.empty:
        raise ValueError("No defensive actions found for the selected team.")

    coords = _split_locations(df["location"])
    df = df.join(coords)

    color_map = {
        "Tackle": "#1f77b4",
        "Interception": "#ff7f0e",
        "Pressure": "#2ca02c",
        "Ball Recovery": "#9467bd",
        "Clearance": "#8c564b",
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    _draw_pitch(ax, team_name=team_name, annotate=True)

    for action_type, group in df.groupby("type.name"):
        ax.scatter(
            group["x"],
            group["y"],
            s=20,
            color=color_map.get(action_type, "#7f7f7f"),
            alpha=0.7,
            label=action_type,
        )

    ax.legend(title="Action", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.set_title(title or f"{team_name} defensive actions")
    return fig, ax


def player_stats(events_df: pd.DataFrame, team_name: str) -> pd.DataFrame:
    team_df = events_df.loc[events_df["team.name"] == team_name].copy()
    player_col = "player.name"

    def _count(type_name: str) -> pd.Series:
        return team_df.loc[team_df["type.name"] == type_name].groupby(player_col).size()

    passes = _count("Pass")
    shots = _count("Shot")
    interceptions = _count("Interception")
    tackles = _count("Tackle")
    pressures = _count("Pressure")
    recoveries = _count("Ball Recovery")

    xg = (
        team_df.loc[team_df["type.name"] == "Shot"]
        .groupby(player_col)["shot.statsbomb_xg"]
        .sum()
    )

    goal_assist = team_df.get("pass.goal_assist")
    if goal_assist is None:
        goal_assist = False
    else:
        goal_assist = goal_assist.fillna(False)

    assist_flag = team_df.get("pass.assist")
    if assist_flag is None:
        assist_flag = False
    else:
        assist_flag = assist_flag.fillna(False)

    assist_mask = (team_df["type.name"] == "Pass") & (goal_assist | assist_flag)
    assists = team_df.loc[assist_mask].groupby(player_col).size()

    stats = pd.DataFrame(
        {
            "passes": passes,
            "shots": shots,
            "xg": xg,
            "assists": assists,
            "interceptions": interceptions,
            "tackles": tackles,
            "pressures": pressures,
            "recoveries": recoveries,
        }
    ).fillna(0)

    stats.index.name = "player"
    return stats.reset_index()


def plot_defensive_action_heatmap(
    events_df: pd.DataFrame,
    team_name: str,
    title: Optional[str] = None,
    gridsize: int = 30,
) -> tuple[plt.Figure, plt.Axes]:
    df = defensive_actions(events_df, team_name)
    if df.empty:
        raise ValueError("No defensive actions found for the selected team.")

    coords = _split_locations(df["location"])
    fig, ax = plt.subplots(figsize=(10, 6))
    _draw_pitch(ax, team_name=team_name, annotate=True)
    heat = ax.hexbin(
        coords["x"],
        coords["y"],
        gridsize=gridsize,
        extent=(0, PITCH_LENGTH, 0, PITCH_WIDTH),
        cmap="Blues",
        mincnt=1,
        alpha=0.8,
    )
    fig.colorbar(heat, ax=ax, fraction=0.03, pad=0.02, label="Defensive actions")
    ax.set_title(title or f"{team_name} defensive action heatmap")
    return fig, ax


def plot_defensive_action_heatmaps_by_type(
    events_df: pd.DataFrame,
    team_name: str,
    gridsize: int = 30,
) -> dict[str, tuple[plt.Figure, plt.Axes]]:
    """Return a heatmap per defensive action type."""
    df = defensive_actions(events_df, team_name)
    if df.empty:
        raise ValueError("No defensive actions found for the selected team.")

    results: dict[str, tuple[plt.Figure, plt.Axes]] = {}
    for action_type in ["Tackle", "Interception", "Pressure", "Ball Recovery"]:
        subset = df.loc[df["type.name"] == action_type]
        if subset.empty:
            continue

        coords = _split_locations(subset["location"])
        fig, ax = plt.subplots(figsize=(10, 6))
        _draw_pitch(ax, team_name=team_name, annotate=True)
        heat = ax.hexbin(
            coords["x"],
            coords["y"],
            gridsize=gridsize,
            extent=(0, PITCH_LENGTH, 0, PITCH_WIDTH),
            cmap="Reds",
            mincnt=1,
            alpha=0.8,
        )
        fig.colorbar(heat, ax=ax, fraction=0.03, pad=0.02, label=f"{action_type} events")
        ax.set_title(f"{team_name} {action_type} heatmap")
        results[action_type] = (fig, ax)

    return results


def plot_defensive_action_density(
    events_df: pd.DataFrame,
    team_name: str,
    action_type: str = "Interception",
    title: Optional[str] = None,
    bins: tuple[int, int] = (30, 20),
    sigma: float = 1.0,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot smoothed spatial density for one defensive action type."""
    df = defensive_actions(events_df, team_name)
    df = df.loc[df["type.name"] == action_type]
    if df.empty:
        raise ValueError(f"No {action_type} events found for {team_name}.")

    coords = _split_locations(df["location"])
    hist, x_edges, y_edges = _smoothed_histogram(
        coords["x"],
        coords["y"],
        bins=bins,
        sigma=sigma,
    )

    return _plot_pitch_heatmap(
        hist,
        x_edges,
        y_edges,
        team_name=team_name,
        title=title or f"{team_name} {action_type.lower()} density",
        colorbar_label=f"Smoothed {action_type.lower()} count",
        cmap="Blues",
    )


def counterattack_outcomes(
    events_df: pd.DataFrame,
    team_name: str,
    window_seconds: int = 10,
) -> pd.DataFrame:
    """Summarize opponent counterattacks after ball losses."""
    if events_df.empty:
        raise ValueError("No events available for counterattack analysis.")

    events = events_df.copy()
    events["time_sec"] = _event_time_seconds(events)

    losses = ball_loss_events(events, team_name)
    if losses.empty:
        raise ValueError("No ball loss events found for the selected team.")

    outcomes: list[str] = []
    for _, loss in losses.iterrows():
        window = events.loc[
            (events["team.name"] != team_name)
            & (events["period"] == loss["period"])
            & (events["time_sec"] > loss["time_sec"])
            & (events["time_sec"] <= loss["time_sec"] + window_seconds)
        ]

        if window.empty:
            outcomes.append("No Shot")
            continue

        shots = window.loc[window["type.name"] == "Shot"]
        if shots.empty:
            outcomes.append("No Shot")
        elif (shots["shot.outcome.name"] == "Goal").any():
            outcomes.append("Goal")
        else:
            outcomes.append("Shot")

    counts = (
        pd.Series(outcomes)
        .value_counts()
        .reindex(["Goal", "Shot", "No Shot"], fill_value=0)
    )
    return counts.rename_axis("outcome").reset_index(name="count")


def plot_counterattack_bar(
    events_df: pd.DataFrame,
    team_name: str,
    window_seconds: int = 10,
    title: Optional[str] = None,
) -> tuple[plt.Figure, plt.Axes]:
    counts = counterattack_outcomes(events_df, team_name, window_seconds=window_seconds)
    bar_df = counts.loc[counts["outcome"].isin(["Shot", "Goal"])]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(bar_df["outcome"], bar_df["count"], color=["#1f77b4", "#d62728"])
    ax.set_ylabel("Count")
    ax.set_title(title or f"{team_name} counterattacks ending in shots/goals")
    return fig, ax


def plot_counterattack_pie(
    events_df: pd.DataFrame,
    team_name: str,
    window_seconds: int = 10,
    title: Optional[str] = None,
) -> tuple[plt.Figure, plt.Axes]:
    counts = counterattack_outcomes(events_df, team_name, window_seconds=window_seconds)
    if counts["count"].sum() == 0:
        raise ValueError("No counterattack outcomes available to plot.")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(
        counts["count"],
        labels=counts["outcome"],
        autopct="%1.0f%%",
        startangle=90,
    )
    ax.set_title(title or f"{team_name} counterattack endings")
    return fig, ax


def recovery_times_after_loss(
    events_df: pd.DataFrame,
    team_name: str,
    max_seconds: int = 60,
) -> pd.DataFrame:
    """Return recovery times (seconds) after each ball loss."""
    if events_df.empty:
        raise ValueError("No events available for recovery-time analysis.")
    if "match_id" not in events_df.columns:
        raise ValueError("events_df must include match_id for recovery-time analysis.")

    events = events_df.copy()
    events["time_sec"] = _event_time_seconds(events)

    losses = ball_loss_events(events, team_name)
    if losses.empty:
        raise ValueError("No ball loss events found for the selected team.")

    results: list[dict] = []
    for _, loss in losses.iterrows():
        window = events.loc[
            (events["match_id"] == loss["match_id"])
            & (events["period"] == loss["period"])
            & (events["time_sec"] > loss["time_sec"])
            & (events["time_sec"] <= loss["time_sec"] + max_seconds)
            & (events["team.name"] == team_name)
        ].sort_values("time_sec")

        if window.empty:
            continue

        recovery_time = float(window.iloc[0]["time_sec"] - loss["time_sec"])
        results.append({"match_id": loss["match_id"], "recovery_seconds": recovery_time})

    if not results:
        raise ValueError("No recovery times found within the specified window.")

    return pd.DataFrame(results)


def plot_recovery_time_boxplot(
    events_df: pd.DataFrame,
    team_name: str,
    max_seconds: int = 60,
    title: Optional[str] = None,
) -> tuple[plt.Figure, plt.Axes]:
    recovery_df = recovery_times_after_loss(
        events_df, team_name, max_seconds=max_seconds
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(recovery_df["recovery_seconds"], vert=True, patch_artist=True)
    ax.set_ylabel("Seconds")
    ax.set_title(title or f"{team_name} recovery time after ball loss")
    return fig, ax


def plot_recovery_time_boxplot_without_worst(
    events_df: pd.DataFrame,
    team_name: str,
    max_seconds: int = 60,
    worst_share: float = 0.10,
    title: Optional[str] = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot recovery times after removing the slowest recovery observations."""
    if not 0 <= worst_share < 1:
        raise ValueError("worst_share must be greater than or equal to 0 and less than 1.")

    recovery_df = recovery_times_after_loss(
        events_df, team_name, max_seconds=max_seconds
    )

    cutoff = recovery_df["recovery_seconds"].quantile(1 - worst_share)
    trimmed = recovery_df.loc[recovery_df["recovery_seconds"] <= cutoff]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(trimmed["recovery_seconds"], vert=True, patch_artist=True)
    ax.set_ylabel("Seconds")
    ax.set_title(
        title
        or f"{team_name} recovery time after ball loss\nwithout slowest {worst_share:.0%}"
    )
    ax.text(
        1.08,
        cutoff,
        f"cutoff: {cutoff:.1f}s",
        va="center",
        fontsize=9,
    )
    return fig, ax


def ball_losses_leading_to_goals_against(
    events_df: pd.DataFrame,
    team_name: str,
    window_seconds: int = 20,
) -> pd.DataFrame:
    """Return ball losses that lead to opponent goals within a time window."""
    if events_df.empty:
        raise ValueError("No events available for goal-against analysis.")
    if "match_id" not in events_df.columns:
        raise ValueError("events_df must include match_id for goal-against analysis.")

    events = events_df.copy()
    events["time_sec"] = _event_time_seconds(events)

    losses = ball_loss_events(events, team_name)
    if losses.empty:
        raise ValueError("No ball loss events found for the selected team.")

    results: list[pd.Series] = []
    for _, loss in losses.iterrows():
        window = events.loc[
            (events["match_id"] == loss["match_id"])
            & (events["period"] == loss["period"])
            & (events["time_sec"] > loss["time_sec"])
            & (events["time_sec"] <= loss["time_sec"] + window_seconds)
        ]

        if window.empty:
            continue

        goals = window.loc[
            (window["team.name"] != team_name)
            & (window["type.name"] == "Shot")
            & (window["shot.outcome.name"] == "Goal")
        ]

        if goals.empty:
            continue

        results.append(loss)

    if not results:
        return pd.DataFrame(columns=losses.columns)

    return pd.DataFrame(results).reset_index(drop=True)


def ball_losses_leading_to_shots_against(
    events_df: pd.DataFrame,
    team_name: str,
    window_seconds: int = 20,
) -> pd.DataFrame:
    """Return ball losses that lead to opponent shots within a time window."""
    if events_df.empty:
        raise ValueError("No events available for shot-against analysis.")
    if "match_id" not in events_df.columns:
        raise ValueError("events_df must include match_id for shot-against analysis.")

    events = events_df.copy()
    events["time_sec"] = _event_time_seconds(events)

    losses = ball_loss_events(events, team_name)
    if losses.empty:
        raise ValueError("No ball loss events found for the selected team.")

    results: list[pd.Series] = []
    for _, loss in losses.iterrows():
        window = events.loc[
            (events["match_id"] == loss["match_id"])
            & (events["period"] == loss["period"])
            & (events["time_sec"] > loss["time_sec"])
            & (events["time_sec"] <= loss["time_sec"] + window_seconds)
        ]

        if window.empty:
            continue

        shots = window.loc[
            (window["team.name"] != team_name)
            & (window["type.name"] == "Shot")
        ]

        if shots.empty:
            continue

        results.append(loss)

    if not results:
        return pd.DataFrame(columns=losses.columns)

    return pd.DataFrame(results).reset_index(drop=True)


def loss_to_shot_flags(
    events_df: pd.DataFrame,
    team_name: str,
    window_seconds: int = 30,
    exclude_corners: bool = False,
) -> pd.DataFrame:
    """Return each ball loss with a flag for opponent shot within window_seconds."""
    if events_df.empty:
        raise ValueError("No events available for loss-to-shot analysis.")
    if "match_id" not in events_df.columns:
        raise ValueError("events_df must include match_id for loss-to-shot analysis.")

    events = events_df.copy()
    events["time_sec"] = _event_time_seconds(events)

    losses = ball_loss_events(events, team_name)
    if exclude_corners:
        losses = _without_corner_events(losses)

    if losses.empty:
        return pd.DataFrame(columns=list(events.columns) + ["leads_to_shot"])

    flags: list[bool] = []
    for _, loss in losses.iterrows():
        window = events.loc[
            (events["match_id"] == loss["match_id"])
            & (events["period"] == loss["period"])
            & (events["time_sec"] > loss["time_sec"])
            & (events["time_sec"] <= loss["time_sec"] + window_seconds)
            & (events["team.name"] != team_name)
            & (events["type.name"] == "Shot")
        ]
        flags.append(not window.empty)

    flagged = losses.copy()
    flagged["leads_to_shot"] = flags
    return flagged


def _loss_to_shot_rate_grid(
    events_df: pd.DataFrame,
    team_name: str,
    window_seconds: int = 30,
    bins: tuple[int, int] = (30, 20),
    sigma: float = 1.0,
    exclude_corners: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    flagged = loss_to_shot_flags(
        events_df,
        team_name,
        window_seconds=window_seconds,
        exclude_corners=exclude_corners,
    )
    if flagged.empty:
        raise ValueError(f"No ball loss events found for {team_name}.")

    coords = _split_locations(flagged["location"])
    all_losses, x_edges, y_edges = _smoothed_histogram(
        coords["x"],
        coords["y"],
        bins=bins,
        sigma=sigma,
    )

    dangerous_coords = coords.loc[flagged["leads_to_shot"].to_numpy()]
    dangerous_losses, _, _ = _smoothed_histogram(
        dangerous_coords["x"],
        dangerous_coords["y"],
        bins=bins,
        sigma=sigma,
    )

    return _safe_rate(dangerous_losses, all_losses), all_losses, x_edges, y_edges


def plot_loss_to_shot_rate(
    events_df: pd.DataFrame,
    team_name: str,
    window_seconds: int = 30,
    title: Optional[str] = None,
    bins: tuple[int, int] = (30, 20),
    sigma: float = 1.0,
    exclude_corners: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot where ball losses become opponent shots as a local rate."""
    rate, _, x_edges, y_edges = _loss_to_shot_rate_grid(
        events_df,
        team_name,
        window_seconds=window_seconds,
        bins=bins,
        sigma=sigma,
        exclude_corners=exclude_corners,
    )

    return _plot_pitch_heatmap(
        rate,
        x_edges,
        y_edges,
        team_name=team_name,
        title=title or f"{team_name} ball-loss danger: shot rate within {window_seconds}s",
        colorbar_label=f"Share of losses followed by shot within {window_seconds}s",
        cmap="Oranges",
        vmin=0,
        vmax=np.nanmax(rate) if np.isfinite(rate).any() else 1,
    )


def _teams_in_events(events_df: pd.DataFrame) -> list[str]:
    return sorted(events_df["team.name"].dropna().unique().tolist())


def _league_loss_to_shot_rate_grid(
    events_df: pd.DataFrame,
    exclude_team: str,
    window_seconds: int = 30,
    bins: tuple[int, int] = (30, 20),
    sigma: float = 1.0,
    exclude_corners: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    numerator = np.zeros(bins, dtype=float)
    denominator = np.zeros(bins, dtype=float)
    x_edges = np.linspace(0, PITCH_LENGTH, bins[0] + 1)
    y_edges = np.linspace(0, PITCH_WIDTH, bins[1] + 1)

    for team in _teams_in_events(events_df):
        if team == exclude_team:
            continue

        try:
            _, team_losses, team_x_edges, team_y_edges = _loss_to_shot_rate_grid(
                events_df,
                team,
                window_seconds=window_seconds,
                bins=bins,
                sigma=sigma,
                exclude_corners=exclude_corners,
            )
            flagged = loss_to_shot_flags(
                events_df,
                team,
                window_seconds=window_seconds,
                exclude_corners=exclude_corners,
            )
        except ValueError:
            continue

        coords = _split_locations(flagged.loc[flagged["leads_to_shot"], "location"])
        team_dangerous, _, _ = _smoothed_histogram(
            coords["x"],
            coords["y"],
            bins=bins,
            sigma=sigma,
        )

        numerator += team_dangerous
        denominator += team_losses
        x_edges = team_x_edges
        y_edges = team_y_edges

    return _safe_rate(numerator, denominator), x_edges, y_edges


def plot_loss_to_shot_rate_vs_average(
    events_df: pd.DataFrame,
    team_name: str,
    window_seconds: int = 30,
    title: Optional[str] = None,
    bins: tuple[int, int] = (30, 20),
    sigma: float = 1.0,
    exclude_corners: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot team ball-loss danger rate minus the rest-of-league rate."""
    team_rate, _, x_edges, y_edges = _loss_to_shot_rate_grid(
        events_df,
        team_name,
        window_seconds=window_seconds,
        bins=bins,
        sigma=sigma,
        exclude_corners=exclude_corners,
    )
    league_rate, _, _ = _league_loss_to_shot_rate_grid(
        events_df,
        exclude_team=team_name,
        window_seconds=window_seconds,
        bins=bins,
        sigma=sigma,
        exclude_corners=exclude_corners,
    )

    diff = team_rate - league_rate
    abs_max = np.nanmax(np.abs(diff)) if np.isfinite(diff).any() else 1

    return _plot_pitch_heatmap(
        diff,
        x_edges,
        y_edges,
        team_name=team_name,
        title=title or f"{team_name} ball-loss danger vs league average",
        colorbar_label=f"Shot-rate difference within {window_seconds}s",
        cmap="coolwarm",
        vmin=-abs_max,
        vmax=abs_max,
    )


def plot_ball_loss_goal_against_heatmap(
    events_df: pd.DataFrame,
    team_name: str,
    window_seconds: int = 20,
    title: Optional[str] = None,
    gridsize: int = 30,
) -> tuple[plt.Figure, plt.Axes]:
    losses = ball_losses_leading_to_goals_against(
        events_df, team_name, window_seconds=window_seconds
    )
    if losses.empty:
        raise ValueError("No ball losses leading to goals found.")

    coords = _split_locations(losses["location"])
    fig, ax = plt.subplots(figsize=(10, 6))
    _draw_pitch(ax, team_name=team_name, annotate=True)
    heat = ax.hexbin(
        coords["x"],
        coords["y"],
        gridsize=gridsize,
        extent=(0, PITCH_LENGTH, 0, PITCH_WIDTH),
        cmap="Reds",
        mincnt=1,
        alpha=0.85,
    )
    fig.colorbar(heat, ax=ax, fraction=0.03, pad=0.02, label="Losses leading to goals")
    ax.set_title(title or f"{team_name} ball losses leading to goals against")
    return fig, ax


def plot_ball_loss_shot_against_heatmap(
    events_df: pd.DataFrame,
    team_name: str,
    window_seconds: int = 20,
    title: Optional[str] = None,
    gridsize: int = 30,
) -> tuple[plt.Figure, plt.Axes]:
    losses = ball_losses_leading_to_shots_against(
        events_df, team_name, window_seconds=window_seconds
    )
    if losses.empty:
        raise ValueError("No ball losses leading to shots found.")

    coords = _split_locations(losses["location"])
    fig, ax = plt.subplots(figsize=(10, 6))
    _draw_pitch(ax, team_name=team_name, annotate=True)
    heat = ax.hexbin(
        coords["x"],
        coords["y"],
        gridsize=gridsize,
        extent=(0, PITCH_LENGTH, 0, PITCH_WIDTH),
        cmap="Oranges",
        mincnt=1,
        alpha=0.85,
    )
    fig.colorbar(heat, ax=ax, fraction=0.03, pad=0.02, label="Losses leading to shots")
    ax.set_title(title or f"{team_name} ball losses leading to shots against")
    return fig, ax


def plot_player_comparison(
    stats_df: pd.DataFrame,
    players: List[str],
    metrics: List[str],
    title: Optional[str] = None,
) -> tuple[plt.Figure, plt.Axes]:
    comparison = stats_df.set_index("player").loc[players, metrics]
    fig, ax = plt.subplots(figsize=(10, 5))
    comparison.plot(kind="bar", ax=ax)
    ax.set_ylabel("Count / value")
    ax.set_title(title or "Player comparison")
    ax.legend(title="Metric", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    return fig, ax


def team_headline_metrics(
    events_df: pd.DataFrame,
    team_name: str,
    loss_window_seconds: int = 30,
    recovery_max_seconds: int = 60,
) -> dict[str, float]:
    """Compute headline defensive transition metrics for one team."""
    team_df = events_df.loc[events_df["team.name"] == team_name].copy()
    matches = max(team_df["match_id"].nunique(), 1)
    losses = ball_loss_events(events_df, team_name)
    loss_flags = loss_to_shot_flags(
        events_df,
        team_name,
        window_seconds=loss_window_seconds,
    )

    try:
        recovery_df = recovery_times_after_loss(
            events_df,
            team_name,
            max_seconds=recovery_max_seconds,
        )
        recovery_time = float(recovery_df["recovery_seconds"].mean())
    except ValueError:
        recovery_time = np.nan

    return {
        "matches": float(matches),
        "pressures_per_match": float((team_df["type.name"] == "Pressure").sum() / matches),
        "recoveries_per_match": float((team_df["type.name"] == "Ball Recovery").sum() / matches),
        "interceptions_per_match": float((team_df["type.name"] == "Interception").sum() / matches),
        "ball_losses_per_match": float(len(losses) / matches),
        "loss_to_shot_rate": float(loss_flags["leads_to_shot"].mean()) if not loss_flags.empty else np.nan,
        "recovery_time_seconds": recovery_time,
    }


def headline_metric_comparison(
    events_df: pd.DataFrame,
    team_name: str,
    loss_window_seconds: int = 30,
    recovery_max_seconds: int = 60,
) -> pd.DataFrame:
    """Compare one team's headline metrics with the rest-of-league average."""
    rows = []
    for team in _teams_in_events(events_df):
        metrics = team_headline_metrics(
            events_df,
            team,
            loss_window_seconds=loss_window_seconds,
            recovery_max_seconds=recovery_max_seconds,
        )
        metrics["team"] = team
        rows.append(metrics)

    metrics_df = pd.DataFrame(rows)
    target = metrics_df.loc[metrics_df["team"] == team_name].iloc[0]
    league = metrics_df.loc[metrics_df["team"] != team_name].mean(numeric_only=True)

    labels = {
        "pressures_per_match": "Pressures",
        "recoveries_per_match": "Recoveries",
        "interceptions_per_match": "Interceptions",
        "ball_losses_per_match": "Ball losses",
        "loss_to_shot_rate": "Loss-to-shot rate",
        "recovery_time_seconds": "Recovery time",
    }

    comparison_rows = []
    for metric, label in labels.items():
        comparison_rows.append(
            {
                "metric": label,
                f"{team_name}": target[metric],
                "league_average": league[metric],
                "comparison": (
                    f"{team_name} {label.lower()} = {target[metric]:.2f} "
                    f"vs league {league[metric]:.2f}."
                ),
            }
        )

    return pd.DataFrame(comparison_rows)


def plot_pass_network(
    events_df: pd.DataFrame,
    team_name: str,
    min_passes: int = 3,
    title: Optional[str] = None,
) -> tuple[plt.Figure, plt.Axes]:
    team_df = events_df.loc[events_df["team.name"] == team_name].copy()
    passes = team_df.loc[team_df["type.name"] == "Pass"].copy()
    passes = passes.loc[passes["pass.outcome.name"].isna()]
    passes = passes.loc[passes["pass.recipient.name"].notna()]

    if passes.empty:
        raise ValueError("No completed passes found for the selected team.")

    coords = _split_locations(passes["location"])
    passes = passes.join(coords)

    player_pos = passes.groupby("player.name")[["x", "y"]].mean()
    player_pos["passes"] = passes.groupby("player.name").size()

    edges = (
        passes.groupby(["player.name", "pass.recipient.name"])
        .size()
        .reset_index(name="count")
    )
    edges = edges.loc[edges["count"] >= min_passes]

    fig, ax = plt.subplots(figsize=(10, 6))
    _draw_pitch(ax, team_name=team_name, annotate=True)

    for _, row in edges.iterrows():
        passer = row["player.name"]
        recipient = row["pass.recipient.name"]
        if passer not in player_pos.index or recipient not in player_pos.index:
            continue
        x1, y1 = player_pos.loc[passer, ["x", "y"]]
        x2, y2 = player_pos.loc[recipient, ["x", "y"]]
        ax.plot([x1, x2], [y1, y2], color="#6c757d", linewidth=0.5 + row["count"] / 6, alpha=0.7)

    ax.scatter(
        player_pos["x"],
        player_pos["y"],
        s=80 + player_pos["passes"] * 2,
        color="#ff7f0e",
        edgecolor="black",
        zorder=3,
    )

    for player, row in player_pos.iterrows():
        ax.text(row["x"], row["y"], player.split(" ")[0], fontsize=8, ha="center", va="center")

    ax.set_title(title or f"{team_name} pass network")
    return fig, ax


def summarize_team(events_df: pd.DataFrame, team_name: str) -> pd.DataFrame:
    """Quick team summary for attack/defense context."""
    team_df = events_df.loc[events_df["team.name"] == team_name].copy()
    summary = {
        "passes": (team_df["type.name"] == "Pass").sum(),
        "shots": (team_df["type.name"] == "Shot").sum(),
        "xg": team_df.loc[team_df["type.name"] == "Shot", "shot.statsbomb_xg"].sum(),
        "pressures": (team_df["type.name"] == "Pressure").sum(),
        "interceptions": (team_df["type.name"] == "Interception").sum(),
        "tackles": (team_df["type.name"] == "Tackle").sum(),
    }
    return pd.DataFrame([summary])
