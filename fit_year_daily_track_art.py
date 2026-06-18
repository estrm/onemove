# python fit_year_daily_track_art.py "D:\fit_files" -y 2025 -o 2025.png -t "2025 IN TRACKS"

import math
import argparse
import calendar
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
from fitparse import FitFile


STRAVA_ORANGE = "#FC4C02"
BG_COLOR = "#111111"
TEXT_COLOR = "#F5F5F5"
MUTED_TEXT = "#555555"


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


def collect_fit_files(input_path):
    p = Path(input_path)

    if p.is_file() and p.suffix.lower() == ".fit":
        return [p]

    if p.is_dir():
        return sorted(p.rglob("*.fit"))

    return []


def get_auto_title_fontsize(title):
    if not title:
        return 0

    n = len(title)

    if n <= 12:
        return 42
    elif n <= 24:
        return 34
    elif n <= 36:
        return 28
    else:
        return 24


def draw_track_in_cell(ax, track, x0, y0, cell_w, cell_h):
    x = track[:, 0]
    y = track[:, 1]

    scale = min(cell_w, cell_h) * 0.38

    x = x * scale + x0 + cell_w / 2
    y = y * scale + y0 + cell_h / 2

    ax.plot(
        x,
        y,
        color="#000000",
        linewidth=1.8,
        alpha=0.35,
        solid_capstyle="round",
        solid_joinstyle="round"
    )

    ax.plot(
        x,
        y,
        color=STRAVA_ORANGE,
        linewidth=0.9,
        alpha=0.95,
        solid_capstyle="round",
        solid_joinstyle="round"
    )


def draw_day_tracks(ax, tracks, x0, y0, cell_w, cell_h):
    n = len(tracks)

    if n == 1:
        draw_track_in_cell(ax, tracks[0], x0, y0, cell_w, cell_h)
        return

    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    sub_w = cell_w / cols
    sub_h = cell_h / rows

    for idx, track in enumerate(tracks):
        r = idx // cols
        c = idx % cols

        sub_x0 = x0 + c * sub_w
        sub_y0 = y0 + (rows - 1 - r) * sub_h

        draw_track_in_cell(ax, track, sub_x0, sub_y0, sub_w, sub_h)


def draw_month_daily_blocks(ax, month, day_tracks, year):
    ax.set_facecolor(BG_COLOR)
    ax.axis("off")
    ax.set_aspect("equal")
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 6.6)

    ax.text(
        0.0,
        6.45,
        f"{month:02d}",
        ha="left",
        va="top",
        fontsize=20,
        color=TEXT_COLOR,
        fontweight="bold"
    )

    _, days_in_month = calendar.monthrange(year, month)

    cols = 7
    rows = 5

    cell_w = 7 / cols
    cell_h = 5.7 / rows
    grid_top = 5.8

    for day in range(1, days_in_month + 1):
        idx = day - 1
        r = idx // cols
        c = idx % cols

        x0 = c * cell_w
        y0 = grid_top - (r + 1) * cell_h

        tracks = day_tracks.get(day, [])

        if tracks:
            draw_day_tracks(ax, tracks, x0, y0, cell_w, cell_h)

            ax.text(
                x0 + 0.06,
                y0 + cell_h - 0.06,
                str(day),
                ha="left",
                va="top",
                fontsize=5,
                color=MUTED_TEXT,
                alpha=0.9
            )


def plot_year_tracks(fit_files, output, title, year):
    month_day_tracks = defaultdict(lambda: defaultdict(list))

    total_files = 0
    valid_files = 0
    skipped_no_gps = 0
    skipped_no_time = 0
    skipped_year = 0

    for fit_file in fit_files:
        total_files += 1
        print(f"读取: {fit_file}")

        try:
            raw, start_time = read_fit_track_and_time(fit_file)

            if start_time is None:
                print("  ⚠️ 跳过：没有时间戳")
                skipped_no_time += 1
                continue

            if start_time.year != year:
                print(f"  ⚠️ 跳过：不是目标年份 {year}")
                skipped_year += 1
                continue

            if len(raw) < 2:
                print("  ⚠️ 跳过：无 GPS 轨迹点")
                skipped_no_gps += 1
                continue

            track = normalize_track(raw)

            if track is None:
                print("  ⚠️ 跳过：轨迹无效")
                skipped_no_gps += 1
                continue

            month = start_time.month
            day = start_time.day

            month_day_tracks[month][day].append(track)
            valid_files += 1

            print(
                f"  ✅ {start_time.date()} | "
                f"{month:02d}-{day:02d} | "
                f"GPS 点数: {len(raw)}"
            )

        except Exception as e:
            print(f"  ❌ 解析失败: {e}")

    if valid_files == 0:
        print("❌ 没有可绘制的 GPS 轨迹")
        return

    fig, axes = plt.subplots(
        3,
        4,
        figsize=(16, 10.5),
        facecolor=BG_COLOR
    )

    axes = axes.flatten()

    for i in range(12):
        month = i + 1
        ax = axes[i]
        draw_month_daily_blocks(
            ax=ax,
            month=month,
            day_tracks=month_day_tracks.get(month, {}),
            year=year
        )

    if title:
        fig.suptitle(
            title,
            fontsize=get_auto_title_fontsize(title),
            color=TEXT_COLOR,
            fontweight="bold",
            y=0.985
        )
        rect = [0, 0, 1, 0.94]
    else:
        rect = [0, 0, 1, 1]

    plt.tight_layout(rect=rect, pad=1.6)

    output_path = Path(output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        str(output_path),
        dpi=300,
        facecolor=BG_COLOR,
        bbox_inches="tight",
        pad_inches=0.25
    )

    plt.close(fig)

    print("\n========== 汇总 ==========")
    print(f"FIT 文件总数: {total_files}")
    print(f"有效轨迹数量: {valid_files}")
    print(f"无 GPS 跳过: {skipped_no_gps}")
    print(f"无时间戳跳过: {skipped_no_time}")
    print(f"非目标年份跳过: {skipped_year}")
    print("✅ 年度每日轨迹图已生成:")
    print(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="批量解析 FIT 文件，生成按月份和日期分区的年度轨迹图"
    )

    parser.add_argument(
        "input",
        help="FIT 文件或 FIT 文件夹"
    )

    parser.add_argument(
        "-o",
        "--output",
        default="fit_year_daily_track_art.png",
        help="输出图片文件名"
    )

    parser.add_argument(
        "-t",
        "--title",
        default="YEAR IN TRACKS",
        help="图片大标题；不需要标题请传空字符串"
    )

    parser.add_argument(
        "-y",
        "--year",
        type=int,
        required=True,
        help="指定年份，例如 2025"
    )

    args = parser.parse_args()

    fit_files = collect_fit_files(args.input)

    print(f"输入路径: {args.input}")
    print(f"找到 FIT 文件: {len(fit_files)}")

    if not fit_files:
        print("❌ 没有找到 .fit 文件")
        return

    plot_year_tracks(
        fit_files=fit_files,
        output=args.output,
        title=args.title,
        year=args.year
    )


if __name__ == "__main__":
    main()
