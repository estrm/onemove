# fit_year_track_art.py
# python fit_year_track_art.py "D:\fit_files" -y 2026  -o 2025.png -t "2026" --size medium

import math
import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.font_manager as fm
import numpy as np
import matplotlib.pyplot as plt
from fitparse import FitFile


logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


TRACK_COLOR = "#FC4C02"
BG_COLOR = "#111111"
TITLE_COLOR = "#FFFFFF"

PREFERRED_FONTS = [
    "Arial",
    "Segoe UI",
    "DejaVu Sans"
]


IMAGE_SIZE_PRESETS = {
    "small": {
        "cell_size": 1.35,
        "dpi": 220,
        "max_fig_width": 22,
        "max_fig_height": 22,
        "title_scale": 0.65,
        "line_scale": 0.75,
    },
    "medium": {
        "cell_size": 1.65,
        "dpi": 260,
        "max_fig_width": 30,
        "max_fig_height": 30,
        "title_scale": 0.78,
        "line_scale": 0.85,
    },
    "large": {
        "cell_size": 2.0,
        "dpi": 300,
        "max_fig_width": 38,
        "max_fig_height": 38,
        "title_scale": 0.9,
        "line_scale": 1.0,
    },
    "poster": {
        "cell_size": 2.3,
        "dpi": 300,
        "max_fig_width": 44,
        "max_fig_height": 44,
        "title_scale": 1.0,
        "line_scale": 1.0,
    }
}


def get_available_font():
    available_fonts = {f.name for f in fm.fontManager.ttflist}

    for font in PREFERRED_FONTS:
        if font in available_fonts:
            return font

    return "DejaVu Sans"


def semicircles_to_degrees(value):
    return value * 180 / 2**31 if value is not None else None


def read_fit_track_and_time(fit_path):
    fitfile = FitFile(str(fit_path))
    points = []
    start_time = None

    for record in fitfile.get_messages("record"):
        timestamp = record.get_value("timestamp")
        lat_raw = record.get_value("position_lat")
        lon_raw = record.get_value("position_long")

        if start_time is None and timestamp is not None:
            start_time = timestamp

        if lat_raw is None or lon_raw is None:
            continue

        lat = semicircles_to_degrees(lat_raw)
        lon = semicircles_to_degrees(lon_raw)

        if lat is not None and lon is not None:
            points.append((lon, lat))

    return np.array(points), start_time


def normalize_track(track):
    if track is None or len(track) < 2:
        return None

    x = track[:, 0] - np.mean(track[:, 0])
    y = track[:, 1] - np.mean(track[:, 1])

    scale = max(np.ptp(x), np.ptp(y))

    if scale == 0:
        return None

    return np.column_stack([x / scale, y / scale])


def downsample_track(track, max_points=1200):
    if len(track) <= max_points:
        return track

    idx = np.linspace(0, len(track) - 1, max_points).astype(int)
    return track[idx]


def collect_fit_files(input_path):
    p = Path(input_path)

    if p.is_file() and p.suffix.lower() == ".fit":
        return [p]

    if p.is_dir():
        return sorted(p.rglob("*.fit"))

    return []


def auto_layout(n, max_cols=None):
    cols = math.ceil(math.sqrt(n * 1.55))

    if max_cols is not None:
        cols = min(cols, max_cols)

    cols = max(cols, 1)
    rows = math.ceil(n / cols)

    return rows, cols


def get_title_fontsize(title, scale=1.0):
    if not title:
        return 0

    n = len(title)

    if n <= 6:
        size = 80
    elif n <= 12:
        size = 68
    elif n <= 24:
        size = 54
    elif n <= 36:
        size = 46
    else:
        size = 38

    return int(size * scale)


def draw_title(fig, title, title_scale=1.0):
    if not title:
        return [0.02, 0.02, 0.98, 0.98]

    fig.text(
        0.5,
        0.965,
        title,
        ha="center",
        va="top",
        fontsize=get_title_fontsize(title, title_scale),
        color=TITLE_COLOR,
        fontweight="bold"
    )

    return [0.02, 0.02, 0.98, 0.88]


def plot_track(ax, track, line_scale=1.0):
    x = track[:, 0]
    y = track[:, 1]

    ax.plot(
        x,
        y,
        color="#000000",
        linewidth=3.8 * line_scale,
        alpha=0.30,
        solid_capstyle="round",
        solid_joinstyle="round"
    )

    ax.plot(
        x,
        y,
        color=TRACK_COLOR,
        linewidth=1.45 * line_scale,
        alpha=0.96,
        solid_capstyle="round",
        solid_joinstyle="round"
    )


def plot_year_tracks(
    fit_files,
    output,
    title,
    year=None,
    size="medium",
    dpi=None,
    max_tracks=None,
    max_cols=None,
    max_points_per_track=1200
):
    preset = IMAGE_SIZE_PRESETS[size]

    if dpi is None:
        dpi = preset["dpi"]

    items = []

    total_files = 0
    skipped_no_time = 0
    skipped_no_gps = 0
    skipped_year = 0
    skipped_invalid = 0
    failed_parse = 0

    for fit_file in fit_files:
        total_files += 1
        print(f"Reading: {fit_file}")

        try:
            raw, start_time = read_fit_track_and_time(fit_file)

            if start_time is None:
                print("  Skipped: no timestamp")
                skipped_no_time += 1
                continue

            if year is not None and start_time.year != year:
                print(f"  Skipped: not target year {year}")
                skipped_year += 1
                continue

            if len(raw) < 2:
                print("  Skipped: no GPS track")
                skipped_no_gps += 1
                continue

            track = normalize_track(raw)

            if track is None:
                print("  Skipped: invalid track")
                skipped_invalid += 1
                continue

            track = downsample_track(track, max_points=max_points_per_track)

            items.append((start_time, track))
            print(f"  OK: {start_time.date()} | GPS points: {len(raw)}")

        except Exception as e:
            failed_parse += 1
            print(f"  Failed to parse: {e}")

    if not items:
        print("No valid GPS tracks to draw.")
        return

    items.sort(key=lambda x: x[0])

    if max_tracks is not None and len(items) > max_tracks:
        print(f"\nToo many tracks: {len(items)}")
        print(f"Only drawing first {max_tracks} tracks.")
        items = items[:max_tracks]

    rows, cols = auto_layout(len(items), max_cols=max_cols)

    selected_font = get_available_font()
    plt.rcParams["font.family"] = selected_font

    cell_size = preset["cell_size"]
    max_fig_width = preset["max_fig_width"]
    max_fig_height = preset["max_fig_height"]

    fig_w = min(cols * cell_size, max_fig_width)
    fig_h = min(rows * cell_size + 1.5, max_fig_height)

    output_px_w = int(fig_w * dpi)
    output_px_h = int(fig_h * dpi)

    print(f"\nUsing font: {selected_font}")
    print(f"Size preset: {size}")
    print(f"Layout: {rows} rows x {cols} cols")
    print(f"Figure size: {fig_w:.1f} x {fig_h:.1f} inches")
    print(f"DPI: {dpi}")
    print(f"Estimated output: {output_px_w} x {output_px_h} px")

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(fig_w, fig_h),
        facecolor=BG_COLOR,
        squeeze=False
    )

    axes = np.array(axes).reshape(rows, cols)

    for ax in axes.flatten():
        ax.set_facecolor(BG_COLOR)
        ax.axis("off")
        ax.set_aspect("equal")
        ax.set_xlim(-0.66, 0.66)
        ax.set_ylim(-0.66, 0.66)

    for idx, (_, track) in enumerate(items):
        ax = axes.flatten()[idx]
        plot_track(ax, track, line_scale=preset["line_scale"])

    for idx in range(len(items), rows * cols):
        axes.flatten()[idx].axis("off")

    rect = draw_title(
        fig,
        title,
        title_scale=preset["title_scale"]
    )

    plt.tight_layout(
        rect=rect,
        pad=0.8,
        w_pad=0.4,
        h_pad=0.4
    )

    output_path = Path(output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        str(output_path),
        dpi=dpi,
        facecolor=BG_COLOR,
        bbox_inches=None
    )

    plt.close(fig)

    print("\n========== Summary ==========")
    print(f"Total FIT files: {total_files}")
    print(f"Valid tracks drawn: {len(items)}")
    print(f"Skipped no timestamp: {skipped_no_time}")
    print(f"Skipped no GPS: {skipped_no_gps}")
    print(f"Skipped non-target year: {skipped_year}")
    print(f"Skipped invalid tracks: {skipped_invalid}")
    print(f"Failed parse: {failed_parse}")
    print("Year track art generated:")
    print(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Generate optimized yearly track art from FIT files."
    )

    parser.add_argument(
        "input",
        help="FIT file or folder containing FIT files"
    )

    parser.add_argument(
        "-o",
        "--output",
        default="year_track_art.png",
        help="Output image filename"
    )

    parser.add_argument(
        "-t",
        "--title",
        default="2026",
        help="Main title, for example 2026"
    )

    parser.add_argument(
        "-y",
        "--year",
        type=int,
        default=None,
        help="Target year, for example 2026"
    )

    parser.add_argument(
        "--size",
        choices=["small", "medium", "large", "poster"],
        default="medium",
        help="Output image size preset"
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=None,
        help="Override DPI. If not set, use preset DPI."
    )

    parser.add_argument(
        "--max-tracks",
        type=int,
        default=None,
        help="Limit number of tracks to draw, for example 300"
    )

    parser.add_argument(
        "--max-cols",
        type=int,
        default=None,
        help="Limit max columns, for example 20"
    )

    parser.add_argument(
        "--max-points",
        type=int,
        default=1200,
        help="Max GPS points per track after downsampling"
    )

    args = parser.parse_args()

    fit_files = collect_fit_files(args.input)

    print(f"Input path: {args.input}")
    print(f"FIT files found: {len(fit_files)}")

    if not fit_files:
        print("No .fit files found.")
        return

    plot_year_tracks(
        fit_files=fit_files,
        output=args.output,
        title=args.title,
        year=args.year,
        size=args.size,
        dpi=args.dpi,
        max_tracks=args.max_tracks,
        max_cols=args.max_cols,
        max_points_per_track=args.max_points
    )


if __name__ == "__main__":
    main()
