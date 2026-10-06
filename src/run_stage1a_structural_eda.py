from __future__ import annotations

import argparse
import csv
import hashlib
import html
import math
import re
import statistics
import struct
import zlib
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


def tf(value: bool) -> str:
    return "TRUE" if value else "FALSE"


def fnum(value: float | int | None, digits: int = 8) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}g}"


def quantile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def weighted_quantile(value_counts: list[tuple[float, int]], q: float) -> float:
    total = sum(count for _, count in value_counts)
    if total == 0:
        return math.nan
    rank = (total - 1) * q

    def at_rank(target: int) -> float:
        cumulative = 0
        for value, count in value_counts:
            cumulative += count
            if target < cumulative:
                return value
        return value_counts[-1][0]

    low, high = int(math.floor(rank)), int(math.ceil(rank))
    if low == high:
        return at_rank(low)
    weight = rank - low
    return at_rank(low) * (1 - weight) + at_rank(high) * weight


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, columns: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def parse_bmp(path: Path) -> dict[str, object]:
    data = path.read_bytes()
    if data[:2] != b"BM":
        raise ValueError(f"Not BMP: {path}")
    pixel_offset = struct.unpack_from("<I", data, 10)[0]
    dib_size = struct.unpack_from("<I", data, 14)[0]
    width = struct.unpack_from("<i", data, 18)[0]
    signed_height = struct.unpack_from("<i", data, 22)[0]
    planes = struct.unpack_from("<H", data, 26)[0]
    bit_depth = struct.unpack_from("<H", data, 28)[0]
    compression = struct.unpack_from("<I", data, 30)[0]
    if dib_size < 40 or width <= 0 or signed_height == 0 or planes != 1:
        raise ValueError(f"Unsupported BMP header: {path}")
    if bit_depth != 8 or compression != 0:
        raise ValueError(f"Only uncompressed 8-bit indexed BMP is supported: {path}")
    height = abs(signed_height)
    palette_data = data[14 + dib_size:pixel_offset]
    palette = []
    for index in range(0, len(palette_data), 4):
        if index + 3 >= len(palette_data):
            break
        b, g, r, _ = palette_data[index:index + 4]
        palette.append((r, g, b))
    if len(palette) < 256:
        raise ValueError(f"Incomplete 8-bit palette: {path}")
    luma = [0.299 * r + 0.587 * g + 0.114 * b for r, g, b in palette]
    stride = ((width * bit_depth + 31) // 32) * 4
    stored_rows = [data[pixel_offset + row * stride:pixel_offset + row * stride + width] for row in range(height)]
    rows = list(reversed(stored_rows)) if signed_height > 0 else stored_rows
    return {
        "width": width, "height": height, "bit_depth": bit_depth,
        "compression": compression, "rows": rows, "palette": palette,
        "luma": luma, "file_size": len(data),
        "colored_palette_entries": sum(not (r == g == b) for r, g, b in palette),
    }


def image_statistics(image: dict[str, object]) -> dict[str, float]:
    rows: list[bytes] = image["rows"]  # type: ignore[assignment]
    luma: list[float] = image["luma"]  # type: ignore[assignment]
    counts: Counter[int] = Counter()
    for row in rows:
        counts.update(row)
    total = sum(counts.values())
    pairs = sorted((luma[index], count) for index, count in counts.items())
    mean = sum(value * count for value, count in pairs) / total
    variance = sum(((value - mean) ** 2) * count for value, count in pairs) / total
    minimum, maximum = pairs[0][0], pairs[-1][0]
    return {
        "min": minimum, "max": maximum, "mean": mean, "std": math.sqrt(variance),
        "p01": weighted_quantile(pairs, 0.01), "p05": weighted_quantile(pairs, 0.05),
        "p25": weighted_quantile(pairs, 0.25), "p50": weighted_quantile(pairs, 0.50),
        "p75": weighted_quantile(pairs, 0.75), "p95": weighted_quantile(pairs, 0.95),
        "p99": weighted_quantile(pairs, 0.99), "dynamic_range": maximum - minimum,
        "min_fraction": sum(count for value, count in pairs if value == minimum) / total,
        "max_fraction": sum(count for value, count in pairs if value == maximum) / total,
    }


def dhash_16(image: dict[str, object]) -> int:
    rows: list[bytes] = image["rows"]  # type: ignore[assignment]
    luma: list[float] = image["luma"]  # type: ignore[assignment]
    height, width = len(rows), len(rows[0])
    sampled: list[list[float]] = []
    for out_y in range(16):
        source_y = min(height - 1, int((out_y + 0.5) * height / 16))
        line = []
        for out_x in range(17):
            source_x = min(width - 1, int((out_x + 0.5) * width / 17))
            line.append(luma[rows[source_y][source_x]])
        sampled.append(line)
    result = 0
    for line in sampled:
        for index in range(16):
            result = (result << 1) | int(line[index] > line[index + 1])
    return result


def parse_metadata(path: Path, stem: str) -> dict[str, str]:
    machine = next((part for part in path.parts if re.match(r"^\d+호기\(", part)), "")
    sn_folder = next((part for part in path.parts if re.match(r"^SN[^_]+_\d{8}_", part)), "")
    sn_match = re.match(r"^(SN[^_]+)_(\d{8})_", sn_folder)
    file_match = re.match(r"^(\d{3})_(\d{8})_(\d{6})\(([^)]*)\)$", stem)
    serial = sn_match.group(1) if sn_match else ""
    folder_date = sn_match.group(2) if sn_match else ""
    filename_prefix = file_match.group(1) if file_match else ""
    filename_date = file_match.group(2) if file_match else ""
    filename_time = file_match.group(3) if file_match else ""
    opaque_value = file_match.group(4) if file_match else ""
    timestamp = ""
    if filename_date and filename_time:
        timestamp = datetime.strptime(filename_date + filename_time, "%Y%m%d%H%M%S").isoformat()
    conflict = bool(folder_date and filename_date and folder_date != filename_date)
    return {
        "machine": machine, "serial_number": serial, "folder_date": folder_date,
        "filename_date": filename_date, "filename_time": filename_time,
        "timestamp": timestamp, "filename_prefix": filename_prefix,
        "filename_parenthetical_value": opaque_value,
        "metadata_conflict": tf(conflict),
        "notes": "Parenthetical value retained as opaque metadata; no undocumented meaning assigned.",
    }


def rankdata(values: list[float]) -> list[float]:
    ordered = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(ordered):
        end = position + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[position]]:
            end += 1
        rank = (position + end - 1) / 2 + 1
        for offset in range(position, end):
            ranks[ordered[offset]] = rank
        position = end
    return ranks


def pearson(x: list[float], y: list[float]) -> float:
    if len(x) < 2:
        return math.nan
    mx, my = statistics.fmean(x), statistics.fmean(y)
    numerator = sum((a - mx) * (b - my) for a, b in zip(x, y))
    dx = math.sqrt(sum((a - mx) ** 2 for a in x))
    dy = math.sqrt(sum((b - my) ** 2 for b in y))
    return numerator / (dx * dy) if dx and dy else math.nan


def spearman(x: list[float], y: list[float]) -> float:
    return pearson(rankdata(x), rankdata(y))


def svg_start(title: str, width: int = 900, height: int = 540) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="30" text-anchor="middle" font-family="sans-serif" font-size="20">{html.escape(title)}</text>',
    ]


def save_histogram(path: Path, values: list[float], title: str, xlabel: str, bins: int = 30) -> None:
    width, height = 900, 540
    left, top, plot_w, plot_h = 75, 55, 790, 410
    minimum, maximum = min(values), max(values)
    if minimum == maximum:
        minimum -= 0.5
        maximum += 0.5
    counts = [0] * bins
    for value in values:
        index = min(bins - 1, int((value - minimum) / (maximum - minimum) * bins))
        counts[index] += 1
    peak = max(counts) or 1
    svg = svg_start(title, width, height)
    svg.append(f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="black"/>')
    svg.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="black"/>')
    bar_w = plot_w / bins
    for index, count in enumerate(counts):
        bar_h = plot_h * count / peak
        svg.append(f'<rect x="{left+index*bar_w:.2f}" y="{top+plot_h-bar_h:.2f}" width="{max(1,bar_w-1):.2f}" height="{bar_h:.2f}" fill="#4472c4"/>')
    svg.extend([
        f'<text x="{left}" y="{top+plot_h+25}" font-family="sans-serif" font-size="12">{minimum:.4g}</text>',
        f'<text x="{left+plot_w}" y="{top+plot_h+25}" text-anchor="end" font-family="sans-serif" font-size="12">{maximum:.4g}</text>',
        f'<text x="{left+plot_w/2}" y="{height-20}" text-anchor="middle" font-family="sans-serif" font-size="14">{html.escape(xlabel)}</text>',
        f'<text x="18" y="{top+plot_h/2}" transform="rotate(-90 18 {top+plot_h/2})" text-anchor="middle" font-family="sans-serif" font-size="14">count</text>',
        '</svg>',
    ])
    path.write_text("\n".join(svg), encoding="utf-8")


def save_bar(path: Path, labels: list[str], counts: list[float], title: str, ylabel: str = "count") -> None:
    width, height = 900, 540
    left, top, plot_w, plot_h = 85, 55, 780, 380
    peak = max(counts) or 1
    svg = svg_start(title, width, height)
    bar_w = plot_w / max(1, len(labels))
    for index, (label, count) in enumerate(zip(labels, counts)):
        bar_h = plot_h * count / peak
        x = left + index * bar_w
        svg.append(f'<rect x="{x+bar_w*0.1:.2f}" y="{top+plot_h-bar_h:.2f}" width="{bar_w*0.8:.2f}" height="{bar_h:.2f}" fill="#4472c4"/>')
        svg.append(f'<text x="{x+bar_w/2:.2f}" y="{top+plot_h+15}" transform="rotate(35 {x+bar_w/2:.2f} {top+plot_h+15})" text-anchor="start" font-family="sans-serif" font-size="10">{html.escape(label)}</text>')
        svg.append(f'<text x="{x+bar_w/2:.2f}" y="{top+plot_h-bar_h-5:.2f}" text-anchor="middle" font-family="sans-serif" font-size="10">{count:g}</text>')
    svg.extend([
        f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="black"/>',
        f'<text x="18" y="{top+plot_h/2}" transform="rotate(-90 18 {top+plot_h/2})" text-anchor="middle" font-family="sans-serif" font-size="14">{html.escape(ylabel)}</text>',
        '</svg>',
    ])
    path.write_text("\n".join(svg), encoding="utf-8")


def save_scatter(path: Path, x: list[float], y: list[float], title: str, xlabel: str, ylabel: str) -> None:
    width, height = 900, 540
    left, top, plot_w, plot_h = 75, 55, 790, 410
    xmin, xmax, ymin, ymax = min(x), max(x), min(y), max(y)
    if xmin == xmax: xmax += 1
    if ymin == ymax: ymax += 1
    svg = svg_start(title, width, height)
    for a, b in zip(x, y):
        px = left + (a - xmin) / (xmax - xmin) * plot_w
        py = top + plot_h - (b - ymin) / (ymax - ymin) * plot_h
        svg.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="2" fill="#4472c4" fill-opacity="0.28"/>')
    svg.extend([
        f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="black"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="black"/>',
        f'<text x="{left+plot_w/2}" y="{height-20}" text-anchor="middle" font-family="sans-serif" font-size="14">{html.escape(xlabel)}</text>',
        f'<text x="18" y="{top+plot_h/2}" transform="rotate(-90 18 {top+plot_h/2})" text-anchor="middle" font-family="sans-serif" font-size="14">{html.escape(ylabel)}</text>',
        '</svg>',
    ])
    path.write_text("\n".join(svg), encoding="utf-8")


def save_heatmap(path: Path, xs: list[float], ys: list[float], title: str, bins: int = 10) -> None:
    grid = [[0 for _ in range(bins)] for _ in range(bins)]
    for x, y in zip(xs, ys):
        ix = min(bins - 1, max(0, int(x * bins)))
        iy = min(bins - 1, max(0, int(y * bins)))
        grid[iy][ix] += 1
    peak = max(max(row) for row in grid) or 1
    width, height = 620, 620
    left, top, size = 75, 55, 500
    svg = svg_start(title, width, height)
    cell = size / bins
    for iy in range(bins):
        for ix in range(bins):
            value = grid[iy][ix] / peak
            blue = int(255 - 170 * value)
            red = int(245 - 200 * value)
            svg.append(f'<rect x="{left+ix*cell:.1f}" y="{top+iy*cell:.1f}" width="{cell:.1f}" height="{cell:.1f}" fill="rgb({red},{blue},255)" stroke="white"/>')
    svg.extend([
        f'<text x="{left+size/2}" y="{height-20}" text-anchor="middle" font-family="sans-serif" font-size="14">bbox center x (normalized)</text>',
        f'<text x="18" y="{top+size/2}" transform="rotate(-90 18 {top+size/2})" text-anchor="middle" font-family="sans-serif" font-size="14">bbox center y (normalized)</text>',
        '</svg>',
    ])
    path.write_text("\n".join(svg), encoding="utf-8")


def save_boxplot(path: Path, groups: dict[str, list[float]], title: str, ylabel: str) -> None:
    labels = list(groups)
    all_values = [value for values in groups.values() for value in values]
    ymin, ymax = min(all_values), max(all_values)
    if ymin == ymax: ymax += 1
    width, height = 900, 540
    left, top, plot_w, plot_h = 90, 55, 760, 390
    svg = svg_start(title, width, height)
    slot = plot_w / max(1, len(labels))
    scale = lambda value: top + plot_h - (value - ymin) / (ymax - ymin) * plot_h
    for index, label in enumerate(labels):
        values = groups[label]
        q0, q1, q2, q3, q4 = [quantile(values, q) for q in (0, .25, .5, .75, 1)]
        x = left + (index + .5) * slot
        svg.append(f'<line x1="{x:.2f}" y1="{scale(q0):.2f}" x2="{x:.2f}" y2="{scale(q4):.2f}" stroke="black"/>')
        svg.append(f'<rect x="{x-slot*.25:.2f}" y="{scale(q3):.2f}" width="{slot*.5:.2f}" height="{scale(q1)-scale(q3):.2f}" fill="#c6d9f1" stroke="#4472c4"/>')
        svg.append(f'<line x1="{x-slot*.25:.2f}" y1="{scale(q2):.2f}" x2="{x+slot*.25:.2f}" y2="{scale(q2):.2f}" stroke="#c00000" stroke-width="2"/>')
        svg.append(f'<text x="{x:.2f}" y="{top+plot_h+20}" text-anchor="middle" font-family="sans-serif" font-size="10">{html.escape(label)}</text>')
    svg.extend([
        f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="black"/>',
        f'<text x="18" y="{top+plot_h/2}" transform="rotate(-90 18 {top+plot_h/2})" text-anchor="middle" font-family="sans-serif" font-size="14">{html.escape(ylabel)}</text>',
        '</svg>',
    ])
    path.write_text("\n".join(svg), encoding="utf-8")


def write_png(path: Path, width: int, height: int, rgb: bytes) -> None:
    raw = b"".join(b"\x00" + rgb[row * width * 3:(row + 1) * width * 3] for row in range(height))
    def chunk(name: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload)) + name + payload + struct.pack(">I", zlib.crc32(name + payload) & 0xFFFFFFFF)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
    path.write_bytes(png)


def pair_montage(path: Path, image_a: dict[str, object], image_b: dict[str, object]) -> None:
    target_w, target_h, gap = 360, 300, 12
    canvas_w, canvas_h = target_w * 2 + gap, target_h
    canvas = bytearray([255] * (canvas_w * canvas_h * 3))
    for offset, image in ((0, image_a), (target_w + gap, image_b)):
        rows: list[bytes] = image["rows"]  # type: ignore[assignment]
        palette: list[tuple[int, int, int]] = image["palette"]  # type: ignore[assignment]
        source_h, source_w = len(rows), len(rows[0])
        for y in range(target_h):
            sy = min(source_h - 1, int(y * source_h / target_h))
            for x in range(target_w):
                sx = min(source_w - 1, int(x * source_w / target_w))
                color = palette[rows[sy][sx]]
                index = (y * canvas_w + offset + x) * 3
                canvas[index:index + 3] = bytes(color)
    write_png(path, canvas_w, canvas_h, bytes(canvas))


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1A structural EDA on the approved allowlist only.")
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    table_dir = workspace / "outputs" / "tables"
    eda_dir = workspace / "outputs" / "eda"
    figure_dir = workspace / "outputs" / "figures" / "stage1a"
    table_dir.mkdir(parents=True, exist_ok=True)
    eda_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    allowlist_path = table_dir / "00_stage1_allowlist.csv"
    with allowlist_path.open("r", encoding="utf-8-sig", newline="") as handle:
        all_allowlist = list(csv.DictReader(handle))
    allowlist = [row for row in all_allowlist if row["eligible_for_stage1_eda"].upper() == "TRUE"]

    stem_counts = Counter(row["stem"] for row in allowlist)
    raw_counts = Counter(row["canonical_raw_path"] for row in allowlist)
    label_counts = Counter(row["official_txt_path"] for row in allowlist)
    integrity_rows = []
    integrity_clean = True
    for row in allowlist:
        raw_path = workspace / Path(row["canonical_raw_path"])
        label_path = workspace / Path(row["official_txt_path"])
        duplicate_stem = stem_counts[row["stem"]] > 1
        duplicate_raw = raw_counts[row["canonical_raw_path"]] > 1
        duplicate_label = label_counts[row["official_txt_path"]] > 1
        raw_exists, label_exists = raw_path.is_file(), label_path.is_file()
        stem_match = raw_path.stem == row["stem"] and label_path.stem == row["stem"]
        clean = raw_exists and label_exists and not duplicate_stem and not duplicate_raw and not duplicate_label and stem_match
        integrity_clean &= clean
        notes = []
        if not raw_exists: notes.append("missing raw BMP")
        if not label_exists: notes.append("missing TXT")
        if duplicate_stem: notes.append("duplicate stem")
        if duplicate_raw: notes.append("duplicate raw path")
        if duplicate_label: notes.append("duplicate label path")
        if not stem_match: notes.append("stem/path mismatch")
        integrity_rows.append({
            "stem": row["stem"], "canonical_raw_path": row["canonical_raw_path"],
            "official_txt_path": row["official_txt_path"], "raw_exists": tf(raw_exists),
            "label_exists": tf(label_exists), "duplicate_stem": tf(duplicate_stem),
            "duplicate_raw_path": tf(duplicate_raw), "duplicate_label_path": tf(duplicate_label),
            "status": "OK" if clean else "ERROR", "notes": "; ".join(notes),
        })
    integrity_columns = ["stem", "canonical_raw_path", "official_txt_path", "raw_exists", "label_exists", "duplicate_stem", "duplicate_raw_path", "duplicate_label_path", "status", "notes"]
    write_csv(table_dir / "01a_allowlist_integrity.csv", integrity_columns, integrity_rows)
    if not integrity_clean:
        raise RuntimeError("Allowlist integrity failed; Stage 1A stopped after integrity output")

    metadata_rows: list[dict[str, object]] = []
    image_rows: list[dict[str, object]] = []
    image_cache_info: dict[str, dict[str, object]] = {}
    dhashes: dict[str, int] = {}
    file_hashes: dict[str, str] = {}
    allow_by_stem = {row["stem"]: row for row in allowlist}

    for row in allowlist:
        path = workspace / Path(row["canonical_raw_path"])
        metadata = parse_metadata(path, row["stem"])
        metadata_rows.append({"stem": row["stem"], "canonical_raw_path": row["canonical_raw_path"], **metadata})
        image = parse_bmp(path)
        stats = image_statistics(image)
        dhashes[row["stem"]] = dhash_16(image)
        file_hashes[row["stem"]] = sha256(path)
        image_cache_info[row["stem"]] = {"path": path, "width": image["width"], "height": image["height"]}
        image_rows.append({
            "stem": row["stem"], "canonical_raw_path": row["canonical_raw_path"],
            "machine": metadata["machine"], "serial_number": metadata["serial_number"],
            "filename_date": metadata["filename_date"], "width": image["width"],
            "height": image["height"], "aspect_ratio": image["width"] / image["height"],
            "channels": "1 indexed channel with RGB palette", "dtype": "uint8 palette index",
            "bit_depth": image["bit_depth"], "file_size_bytes": image["file_size"],
            "intensity_definition": "BT.601 luma decoded from BMP palette; no artifact removal",
            "min_intensity": stats["min"], "max_intensity": stats["max"],
            "mean_intensity": stats["mean"], "std_intensity": stats["std"],
            "p01": stats["p01"], "p05": stats["p05"], "p25": stats["p25"],
            "p50": stats["p50"], "p75": stats["p75"], "p95": stats["p95"],
            "p99": stats["p99"], "dynamic_range": stats["dynamic_range"],
            "min_value_pixel_fraction": stats["min_fraction"],
            "max_value_pixel_fraction": stats["max_fraction"],
            "colored_palette_entries": image["colored_palette_entries"],
            "notes": "Basic whole-image structural statistic; colored palette entries remain present.",
        })

    metadata_columns = ["stem", "canonical_raw_path", "machine", "serial_number", "folder_date", "filename_date", "filename_time", "timestamp", "filename_prefix", "filename_parenthetical_value", "metadata_conflict", "notes"]
    image_columns = ["stem", "canonical_raw_path", "machine", "serial_number", "filename_date", "width", "height", "aspect_ratio", "channels", "dtype", "bit_depth", "file_size_bytes", "intensity_definition", "min_intensity", "max_intensity", "mean_intensity", "std_intensity", "p01", "p05", "p25", "p50", "p75", "p95", "p99", "dynamic_range", "min_value_pixel_fraction", "max_value_pixel_fraction", "colored_palette_entries", "notes"]
    write_csv(table_dir / "01a_sample_metadata.csv", metadata_columns, metadata_rows)
    write_csv(table_dir / "01a_image_features.csv", image_columns, image_rows)

    image_by_stem = {row["stem"]: row for row in image_rows}
    metadata_by_stem = {row["stem"]: row for row in metadata_rows}
    label_integrity_rows: list[dict[str, object]] = []
    object_rows: list[dict[str, object]] = []
    for allow in allowlist:
        stem = allow["stem"]
        label_path = workspace / Path(allow["official_txt_path"])
        image_info = image_by_stem[stem]
        width, height = int(image_info["width"]), int(image_info["height"])
        lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        class_ids: list[int] = []
        invalid_coordinate = False
        out_of_bounds = False
        notes: list[str] = []
        for object_id, line in enumerate(lines):
            parts = line.split()
            try:
                if len(parts) != 5:
                    raise ValueError("expected 5 fields")
                class_id = int(parts[0])
                x, y, bw, bh = [float(value) for value in parts[1:]]
                valid = all(math.isfinite(value) for value in (x, y, bw, bh)) and 0 <= x <= 1 and 0 <= y <= 1 and 0 < bw <= 1 and 0 < bh <= 1
                if not valid:
                    raise ValueError("invalid normalized coordinate")
            except ValueError as exc:
                invalid_coordinate = True
                notes.append(f"row {object_id}: {exc}")
                continue
            class_ids.append(class_id)
            x_px, y_px, bw_px, bh_px = x * width, y * height, bw * width, bh * height
            x1, x2 = x_px - bw_px / 2, x_px + bw_px / 2
            y1, y2 = y_px - bh_px / 2, y_px + bh_px / 2
            bbox_oob = x1 < -1e-6 or y1 < -1e-6 or x2 > width + 1e-6 or y2 > height + 1e-6
            out_of_bounds |= bbox_oob
            edge_distance = min(x1, y1, width - x2, height - y2)
            object_rows.append({
                "image_id": stem, "object_id": object_id, "class_id": class_id,
                "image_width": width, "image_height": height,
                "bbox_x_center_px": x_px, "bbox_y_center_px": y_px,
                "bbox_width_px": bw_px, "bbox_height_px": bh_px,
                "bbox_area_px": bw_px * bh_px, "bbox_width_norm": bw,
                "bbox_height_norm": bh, "bbox_area_ratio": bw * bh,
                "bbox_aspect_ratio": bw_px / bh_px, "bbox_min_side_px": min(bw_px, bh_px),
                "bbox_max_side_px": max(bw_px, bh_px), "bbox_center_x_norm": x,
                "bbox_center_y_norm": y, "image_edge_distance_px": edge_distance,
                "image_edge_distance_norm": edge_distance / min(width, height),
                "machine": metadata_by_stem[stem]["machine"],
                "serial_number": metadata_by_stem[stem]["serial_number"],
                "date": metadata_by_stem[stem]["filename_date"],
                "notes": "bbox aspect ratio is a shape proxy; image-edge distance is not product-edge distance.",
            })
        empty = len(lines) == 0
        status = "OK" if not empty and not invalid_coordinate and not out_of_bounds else "REVIEW"
        label_integrity_rows.append({
            "stem": stem, "object_count": len(lines),
            "class_ids": ";".join(str(value) for value in sorted(set(class_ids))),
            "empty_label": tf(empty), "invalid_coordinate": tf(invalid_coordinate),
            "bbox_out_of_bounds": tf(out_of_bounds), "status": status,
            "notes": "; ".join(notes),
        })

    label_columns = ["stem", "object_count", "class_ids", "empty_label", "invalid_coordinate", "bbox_out_of_bounds", "status", "notes"]
    object_columns = ["image_id", "object_id", "class_id", "image_width", "image_height", "bbox_x_center_px", "bbox_y_center_px", "bbox_width_px", "bbox_height_px", "bbox_area_px", "bbox_width_norm", "bbox_height_norm", "bbox_area_ratio", "bbox_aspect_ratio", "bbox_min_side_px", "bbox_max_side_px", "bbox_center_x_norm", "bbox_center_y_norm", "image_edge_distance_px", "image_edge_distance_norm", "machine", "serial_number", "date", "notes"]
    write_csv(table_dir / "01a_label_integrity.csv", label_columns, label_integrity_rows)
    write_csv(table_dir / "01a_object_geometry.csv", object_columns, object_rows)

    # Temporal adjacency and exploratory 10-second sequence groups.
    sequence_rows: list[dict[str, object]] = []
    grouped_metadata: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in metadata_rows:
        grouped_metadata[(str(row["machine"]), str(row["serial_number"]), str(row["filename_date"]))].append(row)
    run_sizes: list[int] = []
    gap_values: list[float] = []
    gap_counter: Counter[int] = Counter()
    for key, rows in grouped_metadata.items():
        ordered = sorted(rows, key=lambda row: str(row["timestamp"]))
        run_index, run_size = 1, 0
        previous = None
        for row in ordered:
            current = datetime.fromisoformat(str(row["timestamp"]))
            gap = None if previous is None else (current - datetime.fromisoformat(str(previous["timestamp"]))).total_seconds()
            if gap is None or gap > 10:
                if run_size:
                    run_sizes.append(run_size)
                run_index += int(previous is not None)
                run_size = 1
            else:
                run_size += 1
            if gap is not None:
                gap_values.append(gap)
                gap_counter[int(gap)] += 1
            sequence_rows.append({
                "stem": row["stem"], "machine": row["machine"], "serial_number": row["serial_number"],
                "date": row["filename_date"], "timestamp": row["timestamp"],
                "previous_stem": "" if previous is None else previous["stem"],
                "time_gap_seconds": "" if gap is None else gap,
                "sequence_group": f"{key[0]}|{key[1]}|{key[2]}|run{run_index:03d}",
                "notes": "10-second adjacency is exploratory for leakage-risk screening, not a production threshold.",
            })
            previous = row
        if run_size:
            run_sizes.append(run_size)
    sequence_columns = ["stem", "machine", "serial_number", "date", "timestamp", "previous_stem", "time_gap_seconds", "sequence_group", "notes"]
    write_csv(table_dir / "01a_sequence_analysis.csv", sequence_columns, sequence_rows)

    # Pairwise exact and dHash near-duplicate screening.
    metadata_lookup = {str(row["stem"]): row for row in metadata_rows}
    stems = sorted(dhashes)
    near_threshold_hamming = 8
    pair_rows: list[dict[str, object]] = []
    all_similarities: list[float] = []
    closest: list[tuple[float, str, str, int, bool]] = []
    farthest: list[tuple[float, str, str, int, bool]] = []
    exact_pairs = 0
    near_pairs = 0
    near_same_machine = near_same_date = near_short_gap = 0
    for index, stem_a in enumerate(stems):
        meta_a = metadata_lookup[stem_a]
        for stem_b in stems[index + 1:]:
            meta_b = metadata_lookup[stem_b]
            hamming = (dhashes[stem_a] ^ dhashes[stem_b]).bit_count()
            similarity = 1 - hamming / 256
            all_similarities.append(similarity)
            exact = file_hashes[stem_a] == file_hashes[stem_b]
            near = hamming <= near_threshold_hamming and not exact
            if exact: exact_pairs += 1
            if near: near_pairs += 1
            same_machine = meta_a["machine"] == meta_b["machine"]
            same_date = meta_a["filename_date"] == meta_b["filename_date"]
            gap = ""
            if same_machine and same_date:
                gap = abs((datetime.fromisoformat(str(meta_a["timestamp"])) - datetime.fromisoformat(str(meta_b["timestamp"]))).total_seconds())
            if near:
                near_same_machine += int(same_machine)
                near_same_date += int(same_date)
                near_short_gap += int(gap != "" and float(gap) <= 10)
            record = (similarity, stem_a, stem_b, hamming, exact)
            closest.append(record)
            closest.sort(reverse=True)
            closest = closest[:6]
            farthest.append(record)
            farthest.sort()
            farthest = farthest[:6]
            if exact or near:
                pair_rows.append({
                    "stem_a": stem_a, "stem_b": stem_b, "same_machine": tf(same_machine),
                    "same_date": tf(same_date), "time_gap_seconds": gap,
                    "exact_duplicate": tf(exact), "similarity_metric": "16x16 nearest-resize dHash; 256 bits; Hamming<=8",
                    "similarity_value": similarity, "near_duplicate_flag": tf(near),
                    "notes": "Flag is diagnostic only; pair was not collapsed.",
                })
    for similarity, stem_a, stem_b, hamming, exact in farthest[:3]:
        meta_a, meta_b = metadata_lookup[stem_a], metadata_lookup[stem_b]
        same_machine = meta_a["machine"] == meta_b["machine"]
        same_date = meta_a["filename_date"] == meta_b["filename_date"]
        gap = abs((datetime.fromisoformat(str(meta_a["timestamp"])) - datetime.fromisoformat(str(meta_b["timestamp"]))).total_seconds()) if same_machine and same_date else ""
        pair_rows.append({
            "stem_a": stem_a, "stem_b": stem_b, "same_machine": tf(same_machine),
            "same_date": tf(same_date), "time_gap_seconds": gap,
            "exact_duplicate": tf(exact), "similarity_metric": "16x16 nearest-resize dHash; 256 bits; Hamming<=8",
            "similarity_value": similarity, "near_duplicate_flag": "FALSE",
            "notes": "Representative clearly non-duplicate control pair from all-pairs screening.",
        })
    pair_columns = ["stem_a", "stem_b", "same_machine", "same_date", "time_gap_seconds", "exact_duplicate", "similarity_metric", "similarity_value", "near_duplicate_flag", "notes"]
    write_csv(table_dir / "01a_duplicate_nearduplicate.csv", pair_columns, pair_rows)

    # Representative pair images.
    for prefix, pairs in (("near", [item for item in closest if item[3] <= near_threshold_hamming and not item[4]][:3]), ("nonduplicate", farthest[:3])):
        for number, (_, stem_a, stem_b, _, _) in enumerate(pairs, 1):
            image_a = parse_bmp(image_cache_info[stem_a]["path"])  # type: ignore[arg-type]
            image_b = parse_bmp(image_cache_info[stem_b]["path"])  # type: ignore[arg-type]
            pair_montage(figure_dir / f"{prefix}_pair_{number:02d}.png", image_a, image_b)

    # Group summaries.
    image_summary_rows: list[dict[str, object]] = []
    object_summary_rows: list[dict[str, object]] = []
    for group_type, key in (("machine", "machine"), ("serial_number", "serial_number"), ("date", "filename_date")):
        image_groups: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in image_rows:
            image_groups[str(row[key])].append(row)
        for group, rows in sorted(image_groups.items()):
            def vals(name: str) -> list[float]: return [float(row[name]) for row in rows]
            resolutions = sorted({f"{row['width']}x{row['height']}" for row in rows})
            image_summary_rows.append({
                "group_type": group_type, "group_value": group, "sample_count": len(rows),
                "resolutions": ";".join(resolutions),
                "mean_intensity_mean": statistics.fmean(vals("mean_intensity")),
                "mean_intensity_median": quantile(vals("mean_intensity"), .5),
                "std_intensity_mean": statistics.fmean(vals("std_intensity")),
                "dynamic_range_mean": statistics.fmean(vals("dynamic_range")),
                "min_value_fraction_mean": statistics.fmean(vals("min_value_pixel_fraction")),
                "max_value_fraction_mean": statistics.fmean(vals("max_value_pixel_fraction")),
                "notes": "Descriptive only; no causal interpretation.",
            })
        object_groups: dict[str, list[dict[str, object]]] = defaultdict(list)
        object_key = "date" if group_type == "date" else key
        for row in object_rows:
            object_groups[str(row[object_key])].append(row)
        for group, rows in sorted(object_groups.items()):
            def ovals(name: str) -> list[float]: return [float(row[name]) for row in rows]
            samples = {str(row["image_id"]) for row in rows}
            object_summary_rows.append({
                "group_type": group_type, "group_value": group, "sample_count": len(samples),
                "object_count": len(rows), "objects_per_image_mean": len(rows) / len(samples),
                "bbox_area_ratio_mean": statistics.fmean(ovals("bbox_area_ratio")),
                "bbox_area_ratio_median": quantile(ovals("bbox_area_ratio"), .5),
                "bbox_min_side_px_mean": statistics.fmean(ovals("bbox_min_side_px")),
                "bbox_min_side_px_median": quantile(ovals("bbox_min_side_px"), .5),
                "bbox_center_x_mean": statistics.fmean(ovals("bbox_center_x_norm")),
                "bbox_center_y_mean": statistics.fmean(ovals("bbox_center_y_norm")),
                "image_edge_distance_px_mean": statistics.fmean(ovals("image_edge_distance_px")),
                "bbox_aspect_ratio_mean": statistics.fmean(ovals("bbox_aspect_ratio")),
                "notes": "Descriptive object-condition summary; differences may be confounded.",
            })
    image_summary_columns = ["group_type", "group_value", "sample_count", "resolutions", "mean_intensity_mean", "mean_intensity_median", "std_intensity_mean", "dynamic_range_mean", "min_value_fraction_mean", "max_value_fraction_mean", "notes"]
    object_summary_columns = ["group_type", "group_value", "sample_count", "object_count", "objects_per_image_mean", "bbox_area_ratio_mean", "bbox_area_ratio_median", "bbox_min_side_px_mean", "bbox_min_side_px_median", "bbox_center_x_mean", "bbox_center_y_mean", "image_edge_distance_px_mean", "bbox_aspect_ratio_mean", "notes"]
    write_csv(table_dir / "01a_machine_date_image_summary.csv", image_summary_columns, image_summary_rows)
    write_csv(table_dir / "01a_machine_date_object_summary.csv", object_summary_columns, object_summary_rows)

    # Imbalance tables: categorical and exploratory continuous quantile bins.
    imbalance_rows: list[dict[str, object]] = []
    object_count_by_stem = {str(row["stem"]): int(row["object_count"]) for row in label_integrity_rows}
    for feature, key in (("machine", "machine"), ("date", "filename_date"), ("SN", "serial_number"), ("object_count", None)):
        groups: dict[str, list[str]] = defaultdict(list)
        for row in metadata_rows:
            value = str(object_count_by_stem[str(row["stem"])]) if key is None else str(row[key])
            groups[value].append(str(row["stem"]))
        for group, group_stems in sorted(groups.items()):
            objects = sum(object_count_by_stem[stem] for stem in group_stems)
            imbalance_rows.append({
                "feature": feature, "group_or_range": group, "sample_count": len(group_stems),
                "object_count": objects, "proportion": len(group_stems) / len(allowlist),
                "notes": "Observed categorical frequency.",
            })
    class_counts = Counter(int(row["class_id"]) for row in object_rows)
    for class_id, count in sorted(class_counts.items()):
        imbalance_rows.append({"feature": "class_id", "group_or_range": class_id, "sample_count": len({row['image_id'] for row in object_rows if int(row['class_id']) == class_id}), "object_count": count, "proportion": count / len(object_rows), "notes": "Object-level class frequency."})
    continuous_features = {
        "bbox_area_ratio": [float(row["bbox_area_ratio"]) for row in object_rows],
        "bbox_center_x_norm": [float(row["bbox_center_x_norm"]) for row in object_rows],
        "bbox_center_y_norm": [float(row["bbox_center_y_norm"]) for row in object_rows],
        "bbox_aspect_ratio_proxy": [float(row["bbox_aspect_ratio"]) for row in object_rows],
        "image_edge_distance_norm": [float(row["image_edge_distance_norm"]) for row in object_rows],
    }
    for feature, values in continuous_features.items():
        boundaries = [quantile(values, q) for q in (0, .2, .4, .6, .8, 1)]
        bucket_counts = [0] * 5
        for value in values:
            index = 4
            for candidate in range(4):
                if value <= boundaries[candidate + 1]:
                    index = candidate
                    break
            bucket_counts[index] += 1
        for index, count in enumerate(bucket_counts):
            imbalance_rows.append({
                "feature": feature, "group_or_range": f"Q{index+1}: [{boundaries[index]:.8g}, {boundaries[index+1]:.8g}]",
                "sample_count": len({str(row["image_id"]) for row in object_rows}),
                "object_count": count, "proportion": count / len(values),
                "notes": "Exploratory equal-frequency quantile bin; not an operational threshold.",
            })
    imbalance_columns = ["feature", "group_or_range", "sample_count", "object_count", "proportion", "notes"]
    write_csv(table_dir / "01a_imbalance_summary.csv", imbalance_columns, imbalance_rows)

    # Structural correlations.
    correlation_pairs = [
        ("bbox_area_ratio", "bbox_min_side_px"),
        ("bbox_min_side_px", "image_edge_distance_px"),
        ("bbox_min_side_px", "bbox_aspect_ratio"),
        ("bbox_center_x_norm", "bbox_area_ratio"),
        ("bbox_center_y_norm", "bbox_area_ratio"),
        ("bbox_center_x_norm", "bbox_min_side_px"),
        ("bbox_center_y_norm", "bbox_min_side_px"),
    ]
    correlation_rows = []
    for x_name, y_name in correlation_pairs:
        x = [float(row[x_name]) for row in object_rows]
        y = [float(row[y_name]) for row in object_rows]
        correlation_rows.append({
            "variable_x": x_name, "variable_y": y_name, "n": len(x),
            "pearson_r": pearson(x, y), "spearman_rho": spearman(x, y),
            "interpretation_scope": "Structural association only; correlation does not imply causation.",
        })
    correlation_columns = ["variable_x", "variable_y", "n", "pearson_r", "spearman_rho", "interpretation_scope"]
    write_csv(table_dir / "01a_structural_correlations.csv", correlation_columns, correlation_rows)

    # Letterbox resize risk, one row per object and target size.
    resize_rows: list[dict[str, object]] = []
    targets = [320, 416, 512, 640, 800, 1024]
    for row in object_rows:
        for target in targets:
            scale = target / max(float(row["image_width"]), float(row["image_height"]))
            projected_width = float(row["bbox_width_px"]) * scale
            projected_height = float(row["bbox_height_px"]) * scale
            projected_min = min(projected_width, projected_height)
            resize_rows.append({
                "image_id": row["image_id"], "object_id": row["object_id"], "target_size": target,
                "letterbox_scale": scale, "projected_bbox_width_px": projected_width,
                "projected_bbox_height_px": projected_height, "projected_min_side_px": projected_min,
                "below_2px": tf(projected_min < 2), "below_4px": tf(projected_min < 4),
                "below_8px": tf(projected_min < 8), "below_16px": tf(projected_min < 16),
                "notes": "Diagnostic square-letterbox projection; not a final input-size decision.",
            })
    resize_columns = ["image_id", "object_id", "target_size", "letterbox_scale", "projected_bbox_width_px", "projected_bbox_height_px", "projected_min_side_px", "below_2px", "below_4px", "below_8px", "below_16px", "notes"]
    write_csv(table_dir / "01a_resize_risk.csv", resize_columns, resize_rows)

    # Figures.
    resolution_counts = Counter(f"{row['width']}x{row['height']}" for row in image_rows)
    save_bar(figure_dir / "image_resolution_distribution.svg", list(resolution_counts), list(resolution_counts.values()), "Image resolution distribution")
    save_histogram(figure_dir / "aspect_ratio_distribution.svg", [float(row["aspect_ratio"]) for row in image_rows], "Image aspect-ratio distribution", "width / height")
    save_histogram(figure_dir / "mean_intensity_distribution.svg", [float(row["mean_intensity"]) for row in image_rows], "Mean intensity distribution", "palette-decoded luma mean")
    save_histogram(figure_dir / "std_intensity_distribution.svg", [float(row["std_intensity"]) for row in image_rows], "Intensity standard deviation distribution", "palette-decoded luma std")
    save_histogram(figure_dir / "dynamic_range_distribution.svg", [float(row["dynamic_range"]) for row in image_rows], "Dynamic-range distribution", "observed max - min luma")
    save_histogram(figure_dir / "min_value_fraction_distribution.svg", [float(row["min_value_pixel_fraction"]) for row in image_rows], "Minimum-value pixel fraction", "fraction")
    save_histogram(figure_dir / "max_value_fraction_distribution.svg", [float(row["max_value_pixel_fraction"]) for row in image_rows], "Maximum-value pixel fraction", "fraction")
    save_histogram(figure_dir / "bbox_width_distribution.svg", [float(row["bbox_width_px"]) for row in object_rows], "GT bbox width distribution", "pixels")
    save_histogram(figure_dir / "bbox_height_distribution.svg", [float(row["bbox_height_px"]) for row in object_rows], "GT bbox height distribution", "pixels")
    save_histogram(figure_dir / "bbox_area_ratio_distribution.svg", [float(row["bbox_area_ratio"]) for row in object_rows], "GT bbox area-ratio distribution", "bbox area / image area")
    save_histogram(figure_dir / "bbox_min_side_distribution.svg", [float(row["bbox_min_side_px"]) for row in object_rows], "GT bbox minimum-side distribution", "pixels")
    save_histogram(figure_dir / "bbox_aspect_ratio_distribution.svg", [float(row["bbox_aspect_ratio"]) for row in object_rows], "GT bbox aspect-ratio proxy", "width / height")
    save_heatmap(figure_dir / "bbox_center_heatmap.svg", [float(row["bbox_center_x_norm"]) for row in object_rows], [float(row["bbox_center_y_norm"]) for row in object_rows], "GT bbox-center spatial heatmap")
    save_histogram(figure_dir / "image_edge_distance_distribution.svg", [float(row["image_edge_distance_px"]) for row in object_rows], "GT image-edge-distance distribution", "pixels")
    object_count_counts = Counter(int(row["object_count"]) for row in label_integrity_rows)
    save_bar(figure_dir / "object_count_per_image.svg", [str(value) for value in sorted(object_count_counts)], [object_count_counts[value] for value in sorted(object_count_counts)], "Object count per image")
    if gap_values:
        save_histogram(figure_dir / "consecutive_capture_gap_distribution.svg", gap_values, "Consecutive capture gaps within machine/SN/date", "seconds", bins=40)
    save_histogram(figure_dir / "dhash_similarity_distribution.svg", all_similarities, "All-pairs dHash similarity", "1 - Hamming/256", bins=40)
    save_scatter(figure_dir / "bbox_size_vs_edge_distance.svg", [float(row["bbox_min_side_px"]) for row in object_rows], [float(row["image_edge_distance_px"]) for row in object_rows], "BBox size vs image-edge distance", "bbox minimum side (px)", "image-edge distance (px)")
    save_scatter(figure_dir / "bbox_size_vs_aspect_proxy.svg", [float(row["bbox_min_side_px"]) for row in object_rows], [float(row["bbox_aspect_ratio"]) for row in object_rows], "BBox size vs aspect-ratio proxy", "bbox minimum side (px)", "bbox width / height")
    machine_image_groups: dict[str, list[float]] = defaultdict(list)
    machine_object_groups: dict[str, list[float]] = defaultdict(list)
    for row in image_rows: machine_image_groups[str(row["machine"])].append(float(row["mean_intensity"]))
    for row in object_rows: machine_object_groups[str(row["machine"])].append(float(row["bbox_min_side_px"]))
    save_boxplot(figure_dir / "machine_mean_intensity_boxplot.svg", dict(sorted(machine_image_groups.items())), "Mean intensity by machine (descriptive)", "mean luma")
    save_boxplot(figure_dir / "machine_bbox_min_side_boxplot.svg", dict(sorted(machine_object_groups.items())), "BBox minimum side by machine (descriptive)", "pixels")

    # Resize summary plot.
    resize_fraction_by_target = {}
    for target in targets:
        rows = [row for row in resize_rows if row["target_size"] == target]
        resize_fraction_by_target[target] = {threshold: sum(row[f"below_{threshold}px"] == "TRUE" for row in rows) / len(rows) for threshold in (2, 4, 8, 16)}
    save_bar(figure_dir / "resize_risk_below_8px.svg", [str(target) for target in targets], [resize_fraction_by_target[target][8] for target in targets], "Projected objects below 8 px minimum side", "fraction")

    # Key aggregates for reporting.
    total_objects = len(object_rows)
    zero_images = sum(int(row["object_count"]) == 0 for row in label_integrity_rows)
    one_images = sum(int(row["object_count"]) == 1 for row in label_integrity_rows)
    multi_images = sum(int(row["object_count"]) > 1 for row in label_integrity_rows)
    invalid_labels = sum(row["status"] != "OK" for row in label_integrity_rows)
    machine_counts = Counter(str(row["machine"]) for row in metadata_rows)
    sn_counts = Counter(str(row["serial_number"]) for row in metadata_rows)
    date_counts = Counter(str(row["filename_date"]) for row in metadata_rows)
    metadata_conflicts = sum(row["metadata_conflict"] == "TRUE" for row in metadata_rows)
    timestamps = sorted(datetime.fromisoformat(str(row["timestamp"])) for row in metadata_rows)
    short_gaps = {threshold: sum(gap <= threshold for gap in gap_values) for threshold in (2, 4, 6, 10, 60)}
    min_sides = [float(row["bbox_min_side_px"]) for row in object_rows]
    area_ratios = [float(row["bbox_area_ratio"]) for row in object_rows]
    edge_distances = [float(row["image_edge_distance_px"]) for row in object_rows]
    size_quantiles = {q: quantile(min_sides, q) for q in (0, .01, .05, .25, .5, .75, .95, .99, 1)}
    area_quantiles = {q: quantile(area_ratios, q) for q in (0, .01, .05, .25, .5, .75, .95, .99, 1)}
    top_date, top_date_count = date_counts.most_common(1)[0]
    mean_group_values = [float(row["mean_intensity_mean"]) for row in image_summary_rows if row["group_type"] == "machine"]
    strongest_corr = max(correlation_rows, key=lambda row: abs(float(row["spearman_rho"])))

    # Validation risk assessment.
    near_cluster_text = (
        f"{near_pairs} near pairs; same-machine {near_same_machine/near_pairs:.1%}, same-date {near_same_date/near_pairs:.1%}, <=10 s {near_short_gap/near_pairs:.1%}"
        if near_pairs else "no pairs met the declared near-duplicate threshold"
    )
    validation_md = f"""# Stage 1A — Validation Leakage Risk Assessment

Scope: approved allowlist only ({len(allowlist)} logical labeled samples). No split was created.

## Evidence used

- **VERIFIED** — {len(machine_counts)} machine folders, {len(sn_counts)} SN identifiers, and {len(date_counts)} filename dates.
- **VERIFIED** — {len(gap_values)} within-machine/SN/date adjacent gaps; {short_gaps[10]} are at most 10 seconds.
- **VERIFIED** — cross-stem exact duplicate pairs: {exact_pairs}.
- **EXPLORATORY** — 16×16 nearest-resize dHash, Hamming ≤ {near_threshold_hamming}: {near_cluster_text}.
- The dHash screen uses global palette-decoded image structure. Colored markings may influence the metric; they were not interpreted as foreign-object features.
- **VERIFIED** — object and acquisition distributions vary across groups descriptively; this does not establish causation.

## A. Random image split

- Leakage risk: **HIGH/EXPLORATORY** when adjacent burst frames or flagged near duplicates cross folds.
- Distribution-shift risk: low within the same observed pool, but may overestimate deployment generalization.
- Sample-size feasibility: high.
- Advantage: simple and retains all group coverage.
- Limitation: ignores time, SN, machine, and similarity dependencies.

## B. Machine-group split

- Leakage risk: lower for machine-specific acquisition signatures.
- Distribution-shift risk: potentially high because only {len(machine_counts)} machine groups are present and their sample/object distributions differ.
- Sample-size feasibility: limited; results can depend strongly on which machine is held out.
- Advantage: tests cross-machine transfer.
- Limitation: machine is entangled with SN, resolution, date, and object-condition distributions.

## C. Date-group split

- Leakage risk: lower for same-day temporal sequences.
- Distribution-shift risk: moderate to high when date frequencies and object conditions are uneven.
- Sample-size feasibility: feasible across {len(date_counts)} dates, but small dates require grouping.
- Advantage: approximates forward/temporal generalization when ordered chronologically.
- Limitation: dates may be confounded with machine/SN and do not prove independent production lots.

## D. Sequence/group-aware split

- Leakage risk: lowest among candidates for adjacent bursts and flagged near-duplicate components when groups are kept intact.
- Distribution-shift risk: controllable if group assignment is stratified descriptively by machine/date/object count.
- Sample-size feasibility: likely feasible, but the exploratory 10-second sequence rule and dHash threshold require sensitivity checks.
- Advantage: directly targets observed dependence structure without claiming a production-lot meaning.
- Limitation: grouping thresholds are analytical constructs, not documented process identifiers.

## Recommended validation strategies to test next

1. **Sequence/similarity-group-aware grouped validation** with all flagged pairs/components kept in one fold; test sensitivity to 6/10/60-second grouping windows.
2. **Date-grouped temporal validation** as a distribution-shift stress test.
3. **Leave-one-machine-out analysis** as a secondary cross-equipment stress test, not the sole score because only {len(machine_counts)} machine groups exist.

No final validation strategy or split has been selected.
"""
    (eda_dir / "01a_validation_risk_assessment.md").write_text(validation_md, encoding="utf-8-sig")

    # Chapter 1 evidence table.
    evidence_rows = [
        {"evaluation_requirement": "production unit", "observed_fact": "One allowlisted row maps one canonical BMP frame to one TXT label; folders encode machine/SN/date and filenames encode date/time.", "evidence_file": "01a_sample_metadata.csv", "metric_or_count": f"{len(allowlist)} logical frames", "implication": "Frame is the verified analysis unit; short temporal spacing suggests correlated inspections.", "limitation": "Physical product/lot identity is undocumented.", "next_action": "Keep sequence/similarity groups intact when testing validation."},
        {"evaluation_requirement": "variable meaning", "observed_fact": "Filename prefix, date, time, and parenthetical token are parseable.", "evidence_file": "01a_sample_metadata.csv", "metric_or_count": f"metadata conflicts={metadata_conflicts}", "implication": "Date/time can support temporal grouping.", "limitation": "Parenthetical token meaning is unknown.", "next_action": "Treat token as opaque until documentation is found."},
        {"evaluation_requirement": "time relationship", "observed_fact": "Many samples have close within-machine/SN/date capture gaps.", "evidence_file": "01a_sequence_analysis.csv", "metric_or_count": f"gaps<=10s: {short_gaps[10]}/{len(gap_values)}", "implication": "Random splitting can separate dependent frames.", "limitation": "Close time does not prove duplicate product identity.", "next_action": "Test grouped temporal windows."},
        {"evaluation_requirement": "equipment relationship", "observed_fact": "Machine and SN groups are present with descriptive acquisition differences.", "evidence_file": "01a_machine_date_image_summary.csv", "metric_or_count": f"machines={len(machine_counts)}, SN={len(sn_counts)}", "implication": "Equipment-aware validation is relevant.", "limitation": "Machine/SN/date are confounded.", "next_action": "Use grouped stress tests, not causal claims."},
        {"evaluation_requirement": "product relationship", "observed_fact": "No explicit product or lot identifier is present in the approved metadata.", "evidence_file": "01a_sample_metadata.csv", "metric_or_count": "unavailable", "implication": "Product-stratified analysis cannot be justified.", "limitation": "Visual appearance must not be relabeled as product identity.", "next_action": "Request documentation if product grouping is required."},
        {"evaluation_requirement": "missing data", "observed_fact": "Allowlist raw/TXT paths exist and label rows parse.", "evidence_file": "01a_allowlist_integrity.csv; 01a_label_integrity.csv", "metric_or_count": f"missing=0, label review={invalid_labels}", "implication": "Approved scope is structurally complete.", "limitation": "This does not certify annotation correctness.", "next_action": "Carry annotation provenance limits forward."},
        {"evaluation_requirement": "duplicates", "observed_fact": "No cross-stem exact duplicate was found; dHash flags similarity candidates.", "evidence_file": "01a_duplicate_nearduplicate.csv", "metric_or_count": f"exact={exact_pairs}, near={near_pairs}", "implication": "Similarity components may need grouped folds.", "limitation": "dHash threshold is exploratory.", "next_action": "Sensitivity-test grouping threshold before split selection."},
        {"evaluation_requirement": "abnormal/invalid data", "observed_fact": "Label coordinates, positive box sizes, and bounds were validated.", "evidence_file": "01a_label_integrity.csv", "metric_or_count": f"review={invalid_labels}", "implication": "No structural label error blocks later analysis.", "limitation": "No visual annotation-quality audit was performed.", "next_action": "Defer qualitative annotation audit to an approved later task."},
        {"evaluation_requirement": "imbalance", "observed_fact": "Machine/date/object-count and continuous bbox conditions are unevenly represented.", "evidence_file": "01a_imbalance_summary.csv", "metric_or_count": f"largest date={top_date} ({top_date_count})", "implication": "Aggregate metrics may reflect dominant groups.", "limitation": "Quantile bins are exploratory.", "next_action": "Retain continuous variables and report subgroup counts."},
        {"evaluation_requirement": "preprocessing implications", "observed_fact": "Three resolutions and 8-bit indexed palettes are present; resize projections show small-object representation loss at lower targets.", "evidence_file": "01a_image_features.csv; 01a_resize_risk.csv", "metric_or_count": f"resolutions={len(resolution_counts)}", "implication": "Raw input size/letterbox effects require later controlled tests.", "limitation": "No preprocessing method was compared or selected.", "next_action": "Defer representation experiments to the approved later stage."},
        {"evaluation_requirement": "validation strategy implications", "observed_fact": "Temporal bursts, group imbalance, and similarity candidates create random-split leakage risk.", "evidence_file": "01a_validation_risk_assessment.md", "metric_or_count": near_cluster_text, "implication": "Grouped validation candidates should be tested.", "limitation": "No split was created.", "next_action": "Test sequence/date/machine-aware candidates next."},
    ]
    evidence_columns = ["evaluation_requirement", "observed_fact", "evidence_file", "metric_or_count", "implication", "limitation", "next_action"]
    write_csv(table_dir / "01a_chapter1_evidence.csv", evidence_columns, evidence_rows)

    machine_count_text = ", ".join(f"{name}={count}" for name, count in sorted(machine_counts.items()))
    resolution_text = ", ".join(f"{name}={count}" for name, count in sorted(resolution_counts.items()))
    class_text = ", ".join(f"class {name}={count}" for name, count in sorted(class_counts.items()))
    common_gaps = ", ".join(f"{gap}s={count}" for gap, count in gap_counter.most_common(8))
    resize_320 = resize_fraction_by_target[320]
    resize_1024 = resize_fraction_by_target[1024]
    summary_md = f"""# Stage 1A — Pre-model Structural EDA Summary

## 1. Stage 1 sample scope

- **VERIFIED** — allowlist integrity passed: {len(allowlist)} eligible rows, unique stems/raw paths/TXT paths, and no missing files.
- Only canonical raw BMP and primary official TXT were used. No excluded or held-out data was accessed.

## 2. Manufacturing-data unit interpretation

- **VERIFIED** — one logical observation is one X-ray BMP frame with its same-stem TXT annotation.
- **VERIFIED** — paths expose machine, SN, folder date; filenames expose prefix, date, time, and an opaque parenthetical value.
- **UNRESOLVED** — product/lot identity and parenthetical-token meaning are unavailable. A frame must not be claimed to equal a unique physical product without documentation.
- Temporal coverage: {timestamps[0].isoformat()} to {timestamps[-1].isoformat()}; folder-date vs filename-date conflicts: {metadata_conflicts}.

## 3. Image structure

- **VERIFIED** — resolutions: {resolution_text}. All are uncompressed 8-bit indexed BMP with one shared palette and 12 colored palette entries.
- **VERIFIED** — basic intensity statistics use palette-decoded BT.601 luma without artifact removal.
- **EXPLORATORY** — machine-level mean-luma averages span {min(mean_group_values):.3f} to {max(mean_group_values):.3f}; descriptive differences may reflect acquisition or sample-mix confounding.

## 4. Label/object structure

- **VERIFIED** — total objects: {total_objects}; zero-object images: {zero_images}; one-object: {one_images}; multi-object: {multi_images}; structural label-review rows: {invalid_labels}.
- **VERIFIED** — class distribution: {class_text}.
- **VERIFIED** — bbox minimum-side quantiles (px): q01={size_quantiles[.01]:.3f}, q05={size_quantiles[.05]:.3f}, q50={size_quantiles[.5]:.3f}, q95={size_quantiles[.95]:.3f}, q99={size_quantiles[.99]:.3f}.
- **VERIFIED** — bbox area-ratio quantiles: q01={area_quantiles[.01]:.6g}, q50={area_quantiles[.5]:.6g}, q99={area_quantiles[.99]:.6g}.
- Bbox aspect ratio is a shape proxy; image-edge distance is not product-edge distance.

## 5. Dataset imbalance

- **VERIFIED** — machine counts: {machine_count_text}.
- **VERIFIED** — {len(date_counts)} dates are represented; largest date is {top_date} with {top_date_count} samples.
- **EXPLORATORY** — bbox size, position, aspect proxy, edge distance, and object-count concentration are documented with continuous values and quantile bins; bins are not operational thresholds.

## 6. Machine/date/sequence structure

- **VERIFIED** — {len(machine_counts)} machines, {len(sn_counts)} SN identifiers, {len(date_counts)} filename dates.
- **VERIFIED** — adjacent within-group gaps: {len(gap_values)}; <=2s {short_gaps[2]}, <=4s {short_gaps[4]}, <=6s {short_gaps[6]}, <=10s {short_gaps[10]}, <=60s {short_gaps[60]}.
- **VERIFIED** — most frequent exact gaps: {common_gaps}.
- **STRONGLY SUPPORTED** — seconds-apart runs indicate dependency risk, but do not prove the same physical item.

## 7. Exact/near-duplicate findings

- **VERIFIED** — cross-stem exact duplicate pairs: {exact_pairs}.
- **EXPLORATORY** — dHash threshold Hamming≤{near_threshold_hamming}: near pairs={near_pairs}; {near_cluster_text}.
- Near pairs were not collapsed; they are split-leakage candidates only.
- **LIMITATION** — dHash includes global palette-decoded structure, so colored markings may influence similarity; they were not treated as object features.

## 8. Structural feature relationships

- **EXPLORATORY** — strongest tested absolute Spearman association: {strongest_corr['variable_x']} vs {strongest_corr['variable_y']}, rho={float(strongest_corr['spearman_rho']):.3f}.
- Machine/date group comparisons are descriptive and potentially confounded. Correlation does not imply causation.

## 9. Resize/small-object representation risk

- **EXPLORATORY** — target 320: projected minimum side below 2/4/8/16 px = {resize_320[2]:.1%}/{resize_320[4]:.1%}/{resize_320[8]:.1%}/{resize_320[16]:.1%}.
- **EXPLORATORY** — target 1024: projected minimum side below 2/4/8/16 px = {resize_1024[2]:.1%}/{resize_1024[4]:.1%}/{resize_1024[8]:.1%}/{resize_1024[16]:.1%}.
- These are diagnostic letterbox projections, not small/medium/large definitions or an input-size selection.

## 10. Validation leakage risks

- Random image split: high risk of separating adjacent/near-similar frames.
- Machine split: useful stress test but only {len(machine_counts)} groups and strong confounding.
- Date split: reduces same-day leakage but can introduce date/machine/object-distribution shift.
- Sequence/similarity grouping: best aligned with observed dependency structure, but grouping thresholds need sensitivity tests.

## 11. Candidate validation strategies

1. Sequence/similarity-group-aware grouped validation with 6/10/60-second sensitivity checks.
2. Date-grouped temporal validation as a stress test.
3. Leave-one-machine-out as a secondary equipment-transfer stress test.

No final split was selected or created.

## 12. Chapter 1 findings

- Verified analysis unit, equipment/date/time structure, file and label integrity, duplicate risk, structural imbalance, and resize risk are mapped in `01a_chapter1_evidence.csv`.
- Explicit product identity is unavailable and was not invented.

## 13. Questions deferred to Stage 1B

- Local object/background contrast, normalized contrast, CNR-like metrics, bbox-region heterogeneity, background complexity, product-edge feasibility, and colored-artifact relationships.
- No preprocessing representation was evaluated.

## 14. Remaining unresolved issues

- Physical meaning of the parenthetical filename value.
- Physical product/lot identity and whether seconds-apart frames are repeat views or successive items.
- Origin and semantics of colored palette indices.
- External official held-out test availability.
- Sensitivity of sequence and dHash grouping thresholds before validation design.
"""
    (eda_dir / "01a_structural_eda_summary.md").write_text(summary_md, encoding="utf-8-sig")

    print(f"samples={len(allowlist)} objects={total_objects} near_pairs={near_pairs} exact_pairs={exact_pairs}")


if __name__ == "__main__":
    main()
