#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility script to visualize the Gorge Walk 64x64 grid map.

It reads the pre-generated transition graph json (F_level_*.json) and marks:
    - obstacles / walls
    - walkable road tiles
    - configured start & end points
    - treasure positions (actual or potential)

The script can be executed from the repository root, e.g.:

    python visualize_map.py \
        --map-file conf/map_data/F_level_1.json \
        --env-conf agent_dynamic_programming/conf/train_env_conf.toml \
        --output gorge_map.png

If --output is omitted, the figure window will pop up.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover - fallback for <3.11 environments
    import tomli as tomllib  # type: ignore


GRID_SIZE = 64

# Tile categories
OBSTACLE = 0
ROAD = 1
START = 2
END = 3
TREASURE = 4
TREASURE_POSSIBLE = 5

# Treasure id to coordinates (x, y) as documented
TREASURE_COORDS: Dict[int, Tuple[int, int]] = {
    0: (19, 14),
    1: (9, 28),
    2: (9, 44),
    3: (42, 45),
    4: (32, 23),
    5: (49, 56),
    6: (35, 58),
    7: (23, 55),
    8: (41, 33),
    9: (54, 41),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize Gorge Walk map")
    parser.add_argument(
        "--map-file",
        type=Path,
        default=Path(__file__).parent / "conf" / "map_data" / "F_level_1.json",
        help="Path to the map json file (default: conf/map_data/F_level_1.json)",
    )
    parser.add_argument(
        "--env-conf",
        type=Path,
        default=Path(__file__).parent
        / "agent_dynamic_programming"
        / "conf"
        / "train_env_conf.toml",
        help=(
            "Path to the environment configuration toml file that defines start/end/"
            "treasure settings (default: agent_dynamic_programming/conf/train_env_conf.toml)"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="If provided, save the figure to this path instead of showing interactively.",
    )
    return parser.parse_args()


def load_map_states(map_path: Path) -> Iterable[int]:
    """Return the set of walkable state ids from the map json."""
    with map_path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    return (int(state_id) for state_id in data.keys())


def load_env_conf(conf_path: Path) -> Dict[str, object]:
    with conf_path.open("rb") as fp:
        cfg = tomllib.load(fp)
    return cfg.get("env_conf", {})


def state_id_from_coord(coord: Tuple[int, int]) -> int:
    row, col = coord
    return row * GRID_SIZE + col


def visualize(
    map_states: Iterable[int],
    start_coord: Tuple[int, int],
    end_coord: Tuple[int, int],
    treasure_ids: List[int],
    treasure_random: bool,
    output_path: Path | None,
) -> None:
    grid = np.full((GRID_SIZE, GRID_SIZE), OBSTACLE, dtype=np.uint8)

    for state in map_states:
        row = state // GRID_SIZE
        col = state % GRID_SIZE
        if 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE:
            grid[row, col] = ROAD

    # Mark treasures
    if treasure_random:
        # Random treasure placement enabled: highlight all potential spots
        for tid, coord in TREASURE_COORDS.items():
            row, col = coord
            if grid[row, col] == OBSTACLE:
                continue
            grid[row, col] = TREASURE_POSSIBLE
    else:
        for tid in treasure_ids:
            coord = TREASURE_COORDS.get(tid)
            if coord is None:
                continue
            row, col = coord
            if grid[row, col] == OBSTACLE:
                continue
            grid[row, col] = TREASURE

    # Mark start/end (override treasure marker if overlaps)
    start_row, start_col = start_coord
    end_row, end_col = end_coord
    grid[start_row, start_col] = START
    grid[end_row, end_col] = END

    cmap = colors.ListedColormap(
        [
            "#1f1f1f",  # obstacles
            "#d7d7d7",  # road
            "#4caf50",  # start
            "#e53935",  # end
            "#ffb300",  # treasure (active)
            "#fff176",  # treasure (potential)
        ]
    )
    bounds = np.arange(0, 7) - 0.5
    norm = colors.BoundaryNorm(bounds, cmap.N)

    display_grid = grid.T

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(display_grid, cmap=cmap, norm=norm, origin="lower")
    ax.set_title("Gorge Walk Map Visualization (64x64)")
    ax.set_xlabel("Row (x)")
    ax.set_ylabel("Column (y)")
    ax.set_xticks(np.arange(0, GRID_SIZE, 4))
    ax.set_yticks(np.arange(0, GRID_SIZE, 4))
    ax.set_xlim(-0.5, GRID_SIZE - 0.5)
    ax.set_ylim(-0.5, GRID_SIZE - 0.5)
    ax.set_aspect("equal")

    ax.grid(which="both", color="black", linestyle="-", linewidth=0.2, alpha=0.4)

    # Annotate start/end labels
    ax.text(
        start_row,
        start_col,
        "S",
        color="white",
        ha="center",
        va="center",
        fontsize=8,
        fontweight="bold",
    )
    ax.text(
        end_row,
        end_col,
        "E",
        color="white",
        ha="center",
        va="center",
        fontsize=8,
        fontweight="bold",
    )

    # Annotate treasure ids
    if treasure_random:
        ids_to_show = TREASURE_COORDS.items()
    else:
        ids_to_show = ((tid, TREASURE_COORDS[tid]) for tid in treasure_ids if tid in TREASURE_COORDS)

    for tid, (row, col) in ids_to_show:
        if grid[row, col] in {TREASURE, TREASURE_POSSIBLE}:
            ax.text(
                row,
                col,
                str(tid),
                color="black",
                ha="center",
                va="center",
                fontsize=6,
                fontweight="bold",
            )

    # Legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#1f1f1f", edgecolor="black", label="Obstacle / wall"),
        Patch(facecolor="#d7d7d7", edgecolor="black", label="Road"),
        Patch(facecolor="#4caf50", edgecolor="black", label="Start"),
        Patch(facecolor="#e53935", edgecolor="black", label="End"),
    ]
    if treasure_random:
        legend_elements.append(
            Patch(facecolor="#fff176", edgecolor="black", label="Possible treasure spot")
        )
    else:
        legend_elements.append(
            Patch(facecolor="#ffb300", edgecolor="black", label="Treasure (configured)")
        )

    ax.legend(
        handles=legend_elements,
        loc="upper right",
        bbox_to_anchor=(1.45, 1.0),
        borderaxespad=0.5,
        framealpha=0.95,
    )

    fig.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300)
        print(f"Map visualization saved to: {output_path}")
    else:
        plt.show()

    plt.close(fig)


def main() -> None:
    args = parse_args()

    map_states = list(load_map_states(args.map_file))
    env_conf = load_env_conf(args.env_conf)

    if "start" not in env_conf or "end" not in env_conf:
        raise ValueError("Environment config must contain 'start' and 'end' fields.")

    start_raw = env_conf["start"]  # type: ignore[index]
    end_raw = env_conf["end"]  # type: ignore[index]
    start_coord = tuple(int(v) for v in start_raw[:2])  # type: ignore[arg-type]
    end_coord = tuple(int(v) for v in end_raw[:2])  # type: ignore[arg-type]

    treasure_ids = env_conf.get("treasure_id", [])
    if not isinstance(treasure_ids, list):
        treasure_ids = []
    treasure_ids = [int(tid) for tid in treasure_ids if int(tid) in TREASURE_COORDS]
    treasure_random = bool(env_conf.get("treasure_random", False))

    visualize(
        map_states=map_states,
        start_coord=(int(start_coord[0]), int(start_coord[1])),  # type: ignore[index]
        end_coord=(int(end_coord[0]), int(end_coord[1])),  # type: ignore[index]
        treasure_ids=treasure_ids,
        treasure_random=treasure_random,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()

