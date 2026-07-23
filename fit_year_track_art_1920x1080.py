"""Generate a title-free 1920x1080 artwork from FIT tracks."""

import argparse
import logging
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from fitparse import FitFile
from matplotlib.collections import LineCollection


logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

TRACK_COLOR = "#FC4C02"
BG_COLOR = "#111111"
OUTPUT_WIDTH = 1920
OUTPUT_HEIGHT = 1080
OUTPUT_DPI = 120
CANVAS_MARGIN = 36
CELL_GAP = 10
TRACK_CELL_FILL = 0.86
SEMICIRCLE_TO_DEGREE = 180.0 / 2**31


def read_fit_track_and_time(fit_path):
    fitfile = FitFile(str(fit_path))
    points = []
    append_point = points.append
    start_time = None

    for record in fitfile.get_messages("record"):
        timestamp = record.get_value("timestamp")
        if start_time is None and timestamp is not None:
            start_time = timestamp

        lat_raw = record.get_value("position_lat")
        lon_raw = record.get_value("position_long")
        if lat_raw is not None and lon_raw is not None:
            append_point(
                (
                    lon_raw * SEMICIRCLE_TO_DEGREE,
                    lat_raw * SEMICIRCLE_TO_DEGREE,
                )
            )

    return np.asarray(points, dtype=np.float32), start_time


def normalize_track(track):
    """Use the bounding-box center so all tracks have a uniform scale."""
    if track is None or len(track) < 2:
        return None

    minimum = track.min(axis=0)
    maximum = track.max(axis=0)
    scale = float((maximum - minimum).max())
    if not np.isfinite(scale) or scale <= 0:
        return None

    center = (minimum + maximum) * 0.5
    return np.ascontiguousarray((track - center) / scale, dtype=np.float32)


def downsample_track(track, max_points=1200):
    if max_points <= 0 or len(track) <= max_points:
        return track
    indices = np.linspace(0, len(track) - 1, max_points, dtype=np.intp)
    return np.ascontiguousarray(track[indices])


def collect_fit_files(input_path):
    path = Path(input_path)
    if path.is_file() and path.suffix.lower() == ".fit":
        return [path]
    if not path.is_dir():
        return []

    result = []
    for root, _, names in os.walk(path):
        root_path = Path(root)
        result.extend(
            root_path / name
            for name in names
            if name.lower().endswith(".fit")
        )
    return sorted(result)


def calculate_uniform_layout(track_count, max_cols=None):
    """Find the largest equal square cells that fit the 16:9 canvas."""
    usable_w = OUTPUT_WIDTH - 2 * CANVAS_MARGIN
    usable_h = OUTPUT_HEIGHT - 2 * CANVAS_MARGIN
    col_limit = track_count if max_cols is None else max(1, max_cols)
    canvas_ratio = usable_w / usable_h
    best = None

    for rows in range(1, track_count + 1):
        cols = math.ceil(track_count / rows)
        if cols > col_limit:
            continue

        cell_w = (usable_w - (cols - 1) * CELL_GAP) / cols
        cell_h = (usable_h - (rows - 1) * CELL_GAP) / rows
        cell_size = min(cell_w, cell_h)
        if cell_size <= 0:
            continue

        score = (
            cell_size,
            -(rows * cols - track_count),
            -abs(cols / rows - canvas_ratio),
        )
        if best is None or score > best[0]:
            best = (score, rows, cols, cell_size)

    if best is None:
        raise ValueError("Unable to calculate a valid layout")
    return best[1], best[2], best[3]


def build_track_segments(items, rows, cell_size):
    """Balance tracks between rows and map them to canvas coordinates."""
    count = len(items)
    base, remainder = divmod(count, rows)
    row_counts = [base + int(row < remainder) for row in range(rows)]
    total_h = rows * cell_size + (rows - 1) * CELL_GAP
    top = (OUTPUT_HEIGHT - total_h) * 0.5
    scale = cell_size * TRACK_CELL_FILL
    segments = []
    item_index = 0

    for row, row_count in enumerate(row_counts):
        row_w = row_count * cell_size + (row_count - 1) * CELL_GAP
        left = (OUTPUT_WIDTH - row_w) * 0.5
        center_y = (
            OUTPUT_HEIGHT
            - top
            - row * (cell_size + CELL_GAP)
            - cell_size * 0.5
        )

        for col in range(row_count):
            center_x = left + col * (cell_size + CELL_GAP) + cell_size * 0.5
            offset = np.array([center_x, center_y], dtype=np.float32)
            segments.append(items[item_index][1] * scale + offset)
            item_index += 1

    return segments, row_counts


def draw_tracks(items, rows, cell_size):
    segments, row_counts = build_track_segments(items, rows, cell_size)
    fig = plt.figure(
        figsize=(OUTPUT_WIDTH / OUTPUT_DPI, OUTPUT_HEIGHT / OUTPUT_DPI),
        dpi=OUTPUT_DPI,
        facecolor=BG_COLOR,
    )
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(0, OUTPUT_WIDTH)
    ax.set_ylim(0, OUTPUT_HEIGHT)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    # Batch drawing: all shadows once, all orange tracks once.
    shadow = LineCollection(
        segments, colors="#000000", linewidths=2.8, alpha=0.30
    )
    foreground = LineCollection(
        segments, colors=TRACK_COLOR, linewidths=1.15, alpha=0.96
    )
    for collection in (shadow, foreground):
        collection.set_capstyle("round")
        collection.set_joinstyle("round")
        ax.add_collection(collection)

    return fig, row_counts


def plot_year_tracks(
    fit_files,
    output,
    year=None,
    max_tracks=None,
    max_cols=None,
    max_points_per_track=1200,
):
    items = []
    stats = dict(total=0, no_time=0, no_gps=0, wrong_year=0,
                 invalid=0, failed=0)

    for fit_file in fit_files:
        stats["total"] += 1
        print(f"Reading: {fit_file}")
        try:
            raw, start_time = read_fit_track_and_time(fit_file)
            if start_time is None:
                stats["no_time"] += 1
                print("  Skipped: no timestamp")
                continue
            if year is not None and start_time.year != year:
                stats["wrong_year"] += 1
                print(f"  Skipped: not target year {year}")
                continue
            if len(raw) < 2:
                stats["no_gps"] += 1
                print("  Skipped: no GPS track")
                continue

            track = normalize_track(raw)
            if track is None:
                stats["invalid"] += 1
                print("  Skipped: invalid track")
                continue

            track = downsample_track(track, max_points_per_track)
            items.append((start_time, track))
            print(f"  OK: {start_time.date()} | GPS points: {len(raw)}")
        except Exception as exc:
            stats["failed"] += 1
            print(f"  Failed to parse: {exc}")

    if not items:
        print("No valid GPS tracks to draw.")
        return

    items.sort(key=lambda item: item[0])
    if max_tracks is not None and len(items) > max_tracks:
        print(f"Only drawing first {max_tracks} of {len(items)} tracks.")
        items = items[:max_tracks]

    rows, cols, cell_size = calculate_uniform_layout(len(items), max_cols)
    print(f"\nLayout: {rows} rows, up to {cols} tracks per row")
    print(f"Cell size: {cell_size:.1f} x {cell_size:.1f} px")
    print(f"Output size: {OUTPUT_WIDTH} x {OUTPUT_HEIGHT} px")

    fig, row_counts = draw_tracks(items, rows, cell_size)
    output_path = Path(output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path,
        dpi=OUTPUT_DPI,
        facecolor=BG_COLOR,
        bbox_inches=None,
        pad_inches=0,
    )
    plt.close(fig)

    print(f"Tracks per row: {row_counts}")
    print("\n========== Summary ==========")
    print(f"Total FIT files: {stats['total']}")
    print(f"Valid tracks drawn: {len(items)}")
    print(f"Skipped no timestamp: {stats['no_time']}")
    print(f"Skipped no GPS: {stats['no_gps']}")
    print(f"Skipped non-target year: {stats['wrong_year']}")
    print(f"Skipped invalid tracks: {stats['invalid']}")
    print(f"Failed parse: {stats['failed']}")
    print(f"Output dimensions: {OUTPUT_WIDTH} x {OUTPUT_HEIGHT}")
    print(f"Year track art generated:\n{output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate title-free 1920x1080 yearly FIT track art."
    )
    parser.add_argument("input", help="FIT file or folder containing FIT files")
    parser.add_argument(
        "-o", "--output", default="year_track_art_1920x1080.png",
        help="Output image filename",
    )
    parser.add_argument("-y", "--year", type=int, default=None)
    parser.add_argument("--max-tracks", type=int, default=None)
    parser.add_argument("--max-cols", type=int, default=None)
    parser.add_argument(
        "--max-points", type=int, default=1200,
        help="Maximum points per track; 0 disables downsampling",
    )
    args = parser.parse_args()

    if args.max_tracks is not None and args.max_tracks < 1:
        parser.error("--max-tracks must be at least 1")
    if args.max_cols is not None and args.max_cols < 1:
        parser.error("--max-cols must be at least 1")
    if args.max_points < 0 or args.max_points == 1:
        parser.error("--max-points must be 0 or at least 2")

    fit_files = collect_fit_files(args.input)
    print(f"Input path: {args.input}")
    print(f"FIT files found: {len(fit_files)}")
    if not fit_files:
        print("No .fit files found.")
        return

    plot_year_tracks(
        fit_files=fit_files,
        output=args.output,
        year=args.year,
        max_tracks=args.max_tracks,
        max_cols=args.max_cols,
        max_points_per_track=args.max_points,
    )


if __name__ == "__main__":
    main()
