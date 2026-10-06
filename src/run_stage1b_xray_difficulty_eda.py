from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import Counter, defaultdict, deque
from pathlib import Path

from run_stage1a_structural_eda import (
    fnum,
    parse_bmp,
    pearson,
    quantile,
    save_boxplot,
    save_histogram,
    save_scatter,
    spearman,
    tf,
    write_csv,
    write_png,
)


EPSILON = 1e-6
RING_SCALES = (1.5, 2.0, 3.0)
CHROMATIC_RANGE_THRESHOLD = 200
ARTIFACT_NEAR_MARGIN_PX = 3


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def is_true(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes"}


def basic_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {key: math.nan for key in ("mean", "median", "std", "p10", "p25", "p75", "p90", "min", "max", "mad", "iqr", "entropy")}
    mean = statistics.fmean(values)
    std = math.sqrt(statistics.fmean([(v - mean) ** 2 for v in values]))
    median = quantile(values, 0.5)
    mad = quantile([abs(v - median) for v in values], 0.5)
    counts = Counter(int(round(max(0, min(255, v)))) for v in values)
    total = len(values)
    entropy = -sum((count / total) * math.log2(count / total) for count in counts.values())
    p25, p75 = quantile(values, 0.25), quantile(values, 0.75)
    return {
        "mean": mean, "median": median, "std": std,
        "p10": quantile(values, 0.10), "p25": p25, "p75": p75, "p90": quantile(values, 0.90),
        "min": min(values), "max": max(values), "mad": mad, "iqr": p75 - p25, "entropy": entropy,
    }


def geometry_box(row: dict[str, str], width: int, height: int) -> tuple[int, int, int, int]:
    cx, cy = float(row["bbox_x_center_px"]), float(row["bbox_y_center_px"])
    bw, bh = float(row["bbox_width_px"]), float(row["bbox_height_px"])
    x0 = max(0, min(width - 1, math.floor(cx - bw / 2)))
    y0 = max(0, min(height - 1, math.floor(cy - bh / 2)))
    x1 = max(x0 + 1, min(width, math.ceil(cx + bw / 2)))
    y1 = max(y0 + 1, min(height, math.ceil(cy + bh / 2)))
    return x0, y0, x1, y1


def expanded_box(box: tuple[int, int, int, int], scale: float, width: int, height: int) -> tuple[tuple[int, int, int, int], float]:
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    target_w, target_h = max(1, (x1 - x0) * scale), max(1, (y1 - y0) * scale)
    ux0, uy0 = math.floor(cx - target_w / 2), math.floor(cy - target_h / 2)
    ux1, uy1 = math.ceil(cx + target_w / 2), math.ceil(cy + target_h / 2)
    nominal = max(1, (ux1 - ux0) * (uy1 - uy0))
    clipped = (max(0, ux0), max(0, uy0), min(width, ux1), min(height, uy1))
    clipped_area = max(0, clipped[2] - clipped[0]) * max(0, clipped[3] - clipped[1])
    return clipped, 1 - clipped_area / nominal


def in_box(x: int, y: int, box: tuple[int, int, int, int]) -> bool:
    return box[0] <= x < box[2] and box[1] <= y < box[3]


def image_arrays(image: dict[str, object]) -> tuple[list[bytes], list[bytes], list[tuple[int, int]], list[dict[str, object]]]:
    indexed_rows: list[bytes] = image["rows"]  # type: ignore[assignment]
    palette: list[tuple[int, int, int]] = image["palette"]  # type: ignore[assignment]
    luma_table = bytes(max(0, min(255, int(round(0.299 * r + 0.587 * g + 0.114 * b)))) for r, g, b in palette)
    chrom_table = bytes(1 if max(rgb) - min(rgb) >= CHROMATIC_RANGE_THRESHOLD else 0 for rgb in palette)
    gray_rows = [row.translate(luma_table) for row in indexed_rows]
    mask_rows = [row.translate(chrom_table) for row in indexed_rows]
    points: list[tuple[int, int]] = []
    for y, row in enumerate(mask_rows):
        start = 0
        while True:
            x = row.find(b"\x01", start)
            if x < 0:
                break
            points.append((x, y))
            start = x + 1
    components = chromatic_components(points)
    return gray_rows, mask_rows, points, components


def chromatic_components(points: list[tuple[int, int]]) -> list[dict[str, object]]:
    remaining = set(points)
    components: list[dict[str, object]] = []
    while remaining:
        seed = remaining.pop()
        stack = [seed]
        component = [seed]
        while stack:
            x, y = stack.pop()
            for ny in range(y - 1, y + 2):
                for nx in range(x - 1, x + 2):
                    if (nx, ny) in remaining:
                        remaining.remove((nx, ny))
                        stack.append((nx, ny))
                        component.append((nx, ny))
        xs, ys = [p[0] for p in component], [p[1] for p in component]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        width, height = x1 - x0 + 1, y1 - y0 + 1
        border_count = sum(x - x0 <= 1 or x1 - x <= 1 or y - y0 <= 1 or y1 - y <= 1 for x, y in component)
        fill = len(component) / max(1, width * height)
        rectangular = width >= 4 and height >= 4 and border_count / len(component) >= 0.75 and fill <= 0.65
        components.append({
            "pixels": component, "size": len(component), "box": (x0, y0, x1 + 1, y1 + 1),
            "border_fraction": border_count / len(component), "fill_fraction": fill,
            "thin_rectangle_like": rectangular,
        })
    return sorted(components, key=lambda c: int(c["size"]), reverse=True)


def region_coords(
    box: tuple[int, int, int, int],
    expanded: tuple[int, int, int, int],
    mask_rows: list[bytes],
    other_boxes: list[tuple[int, int, int, int]],
) -> tuple[list[tuple[int, int]], int, int, int]:
    coords: list[tuple[int, int]] = []
    candidate = artifact_excluded = other_excluded = 0
    for y in range(expanded[1], expanded[3]):
        for x in range(expanded[0], expanded[2]):
            if in_box(x, y, box):
                continue
            candidate += 1
            if mask_rows[y][x]:
                artifact_excluded += 1
                continue
            if any(in_box(x, y, other) for other in other_boxes):
                other_excluded += 1
                continue
            coords.append((x, y))
    return coords, candidate, artifact_excluded, other_excluded


def box_coords(box: tuple[int, int, int, int], mask_rows: list[bytes]) -> tuple[list[tuple[int, int]], int]:
    coords, masked = [], 0
    for y in range(box[1], box[3]):
        for x in range(box[0], box[2]):
            if mask_rows[y][x]:
                masked += 1
            else:
                coords.append((x, y))
    return coords, masked


def values_at(rows: list[bytes] | list[list[float]], coords: list[tuple[int, int]]) -> list[float]:
    return [float(rows[y][x]) for x, y in coords]


def gradient_values(rows: list[bytes], coords: list[tuple[int, int]], mask_rows: list[bytes]) -> list[float]:
    height, width = len(rows), len(rows[0])
    values: list[float] = []
    for x, y in coords:
        if x <= 0 or y <= 0 or x >= width - 1 or y >= height - 1:
            continue
        if mask_rows[y][x] or mask_rows[y][x - 1] or mask_rows[y][x + 1] or mask_rows[y - 1][x] or mask_rows[y + 1][x]:
            continue
        gx = (rows[y][x + 1] - rows[y][x - 1]) / 2
        gy = (rows[y + 1][x] - rows[y - 1][x]) / 2
        values.append(math.hypot(gx, gy))
    return values


def nearest_distance_to_box(points: list[tuple[int, int]], box: tuple[int, int, int, int]) -> float:
    if not points:
        return math.nan
    x0, y0, x1, y1 = box
    best = math.inf
    for x, y in points:
        dx = max(x0 - x, 0, x - (x1 - 1))
        dy = max(y0 - y, 0, y - (y1 - 1))
        best = min(best, math.hypot(dx, dy))
    return best


def integral_image(rows: list[bytes]) -> list[list[int]]:
    width = len(rows[0])
    integral = [[0] * (width + 1)]
    for row in rows:
        out = [0] * (width + 1)
        running = 0
        previous = integral[-1]
        for x, value in enumerate(row, start=1):
            running += value
            out[x] = previous[x] + running
        integral.append(out)
    return integral


def box_mean(integral: list[list[int]], x: int, y: int, radius: int, width: int, height: int) -> float:
    x0, x1 = max(0, x - radius), min(width, x + radius + 1)
    y0, y1 = max(0, y - radius), min(height, y + radius + 1)
    total = integral[y1][x1] - integral[y0][x1] - integral[y1][x0] + integral[y0][x0]
    return total / ((x1 - x0) * (y1 - y0))


def clipped_tile_maps(rows: list[bytes], mask_rows: list[bytes], tiles_x: int = 8, tiles_y: int = 8, clip_factor: float = 2.0) -> list[list[bytes]]:
    height, width = len(rows), len(rows[0])
    maps: list[list[bytes]] = []
    for ty in range(tiles_y):
        y0, y1 = ty * height // tiles_y, (ty + 1) * height // tiles_y
        line = []
        for tx in range(tiles_x):
            x0, x1 = tx * width // tiles_x, (tx + 1) * width // tiles_x
            hist = [0] * 256
            count = 0
            for y in range(y0, y1):
                for x in range(x0, x1):
                    if not mask_rows[y][x]:
                        hist[rows[y][x]] += 1
                        count += 1
            if not count:
                line.append(bytes(range(256)))
                continue
            limit = max(1, int(clip_factor * count / 256))
            excess = sum(max(0, value - limit) for value in hist)
            hist = [min(value, limit) for value in hist]
            add, remainder = divmod(excess, 256)
            hist = [value + add + (1 if index < remainder else 0) for index, value in enumerate(hist)]
            cumulative = 0
            mapping = []
            for value in hist:
                cumulative += value
                mapping.append(max(0, min(255, round(255 * cumulative / count))))
            line.append(bytes(mapping))
        maps.append(line)
    return maps


def representation_values(
    name: str,
    rows: list[bytes],
    coords: list[tuple[int, int]],
    mask_rows: list[bytes],
    window_map: bytes,
    clahe_maps: list[list[bytes]],
    integral: list[list[int]],
) -> list[float]:
    height, width = len(rows), len(rows[0])
    if name == "RAW":
        return values_at(rows, coords)
    if name == "WINDOW_P01_P99":
        return [float(window_map[rows[y][x]]) for x, y in coords]
    if name == "CLAHE_8X8_CLIP2_NO_INTERP":
        return [float(clahe_maps[min(7, y * 8 // height)][min(7, x * 8 // width)][rows[y][x]]) for x, y in coords]
    if name == "LOCAL_RESIDUAL_ABS_15PX":
        return [abs(rows[y][x] - box_mean(integral, x, y, 7, width, height)) for x, y in coords]
    if name == "GRADIENT_MAGNITUDE":
        return gradient_values(rows, coords, mask_rows)
    raise ValueError(name)


def otsu_threshold(rows: list[bytes], mask_rows: list[bytes]) -> int:
    hist = [0] * 256
    for y, row in enumerate(rows):
        for x, value in enumerate(row):
            if not mask_rows[y][x]:
                hist[value] += 1
    total = sum(hist)
    total_sum = sum(index * count for index, count in enumerate(hist))
    background_count = background_sum = 0
    best_threshold = 0
    best_variance = -1.0
    for threshold, count in enumerate(hist):
        background_count += count
        if not background_count:
            continue
        foreground_count = total - background_count
        if not foreground_count:
            break
        background_sum += threshold * count
        mean_b = background_sum / background_count
        mean_f = (total_sum - background_sum) / foreground_count
        variance = background_count * foreground_count * (mean_b - mean_f) ** 2
        if variance > best_variance:
            best_variance, best_threshold = variance, threshold
    return best_threshold


def binary_close(mask: bytearray, width: int, height: int) -> bytearray:
    dilated = bytearray(len(mask))
    for y in range(height):
        for x in range(width):
            if mask[y * width + x]:
                for ny in range(max(0, y - 1), min(height, y + 2)):
                    start = ny * width + max(0, x - 1)
                    end = ny * width + min(width, x + 2)
                    dilated[start:end] = b"\x01" * (end - start)
    eroded = bytearray(len(mask))
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if all(dilated[ny * width + nx] for ny in range(y - 1, y + 2) for nx in range(x - 1, x + 2)):
                eroded[y * width + x] = 1
    return eroded


def largest_component(mask: bytearray, width: int, height: int) -> set[int]:
    visited = bytearray(len(mask))
    best: set[int] = set()
    for start in range(len(mask)):
        if not mask[start] or visited[start]:
            continue
        visited[start] = 1
        queue = [start]
        component: set[int] = {start}
        while queue:
            index = queue.pop()
            x, y = index % width, index // width
            for neighbor in (index - 1 if x else -1, index + 1 if x < width - 1 else -1, index - width if y else -1, index + width if y < height - 1 else -1):
                if neighbor >= 0 and mask[neighbor] and not visited[neighbor]:
                    visited[neighbor] = 1
                    queue.append(neighbor)
                    component.add(neighbor)
        if len(component) > len(best):
            best = component
    return best


def boundary_component(rows: list[bytes], mask_rows: list[bytes], method: str) -> tuple[set[int], int]:
    height, width = len(rows), len(rows[0])
    threshold = otsu_threshold(rows, mask_rows)
    binary = bytearray(width * height)
    for y in range(height):
        for x in range(width):
            if not mask_rows[y][x] and rows[y][x] <= threshold:
                binary[y * width + x] = 1
    if method == "OTSU_DARK_LCC_CLOSE3":
        binary = binary_close(binary, width, height)
    return largest_component(binary, width, height), threshold


def component_quality(component: set[int], width: int, height: int) -> tuple[bool, str, dict[str, float]]:
    if not component:
        return False, "no connected foreground component", {"area_fraction": 0.0, "bbox_width_fraction": 0.0, "bbox_height_fraction": 0.0, "border_touch_fraction": 0.0}
    xs = [index % width for index in component]
    ys = [index // width for index in component]
    area_fraction = len(component) / (width * height)
    bw = (max(xs) - min(xs) + 1) / width
    bh = (max(ys) - min(ys) + 1) / height
    border_touch = sum(x == 0 or y == 0 or x == width - 1 or y == height - 1 for x, y in zip(xs, ys)) / len(component)
    center_x, center_y = statistics.fmean(xs) / width, statistics.fmean(ys) / height
    failures = []
    if not 0.08 <= area_fraction <= 0.80: failures.append("implausible area fraction")
    if bw < 0.25 or bh < 0.20: failures.append("component extent too small")
    if not 0.10 <= center_x <= 0.90 or not 0.10 <= center_y <= 0.90: failures.append("component centroid near image border")
    if border_touch > 0.02: failures.append("excessive image-border contact")
    return not failures, "; ".join(failures), {"area_fraction": area_fraction, "bbox_width_fraction": bw, "bbox_height_fraction": bh, "border_touch_fraction": border_touch}


def component_iou(a: set[int], b: set[int]) -> float:
    union = len(a | b)
    return len(a & b) / union if union else math.nan


def component_boundary_pixels(component: set[int], width: int, height: int) -> list[tuple[int, int]]:
    # Only retain the contour adjacent to background connected to the image border.
    # This deliberately excludes internal holes caused by chromatic-mask pixels or product structure.
    exterior = bytearray(width * height)
    queue: deque[int] = deque()
    border_indexes = list(range(width)) + list(range((height - 1) * width, height * width))
    border_indexes += [y * width for y in range(height)] + [y * width + width - 1 for y in range(height)]
    for index in border_indexes:
        if index not in component and not exterior[index]:
            exterior[index] = 1
            queue.append(index)
    while queue:
        index = queue.popleft()
        x, y = index % width, index // width
        for neighbor in (
            index - 1 if x else -1,
            index + 1 if x < width - 1 else -1,
            index - width if y else -1,
            index + width if y < height - 1 else -1,
        ):
            if neighbor >= 0 and neighbor not in component and not exterior[neighbor]:
                exterior[neighbor] = 1
                queue.append(neighbor)
    boundary = []
    for index in component:
        x, y = index % width, index // width
        neighbors = (
            index - 1 if x else -1,
            index + 1 if x < width - 1 else -1,
            index - width if y else -1,
            index + width if y < height - 1 else -1,
        )
        if any(neighbor < 0 or exterior[neighbor] for neighbor in neighbors):
            boundary.append((x, y))
    return boundary


def point_to_boundary_distance(x: float, y: float, boundary: list[tuple[int, int]]) -> float:
    return min((math.hypot(x - bx, y - by) for bx, by in boundary), default=math.nan)


def draw_overlay(path: Path, image: dict[str, object], boxes: list[tuple[int, int, int, int]], component: set[int] | None = None) -> None:
    rows: list[bytes] = image["rows"]  # type: ignore[assignment]
    palette: list[tuple[int, int, int]] = image["palette"]  # type: ignore[assignment]
    height, width = len(rows), len(rows[0])
    rgb = bytearray(width * height * 3)
    for y, row in enumerate(rows):
        for x, index in enumerate(row):
            rgb[(y * width + x) * 3:(y * width + x) * 3 + 3] = bytes(palette[index])
    for box in boxes:
        x0, y0, x1, y1 = box
        for x in range(x0, x1):
            for y in (y0, y1 - 1):
                rgb[(y * width + x) * 3:(y * width + x) * 3 + 3] = b"\x00\xff\xff"
        for y in range(y0, y1):
            for x in (x0, x1 - 1):
                rgb[(y * width + x) * 3:(y * width + x) * 3 + 3] = b"\x00\xff\xff"
    if component:
        for x, y in component_boundary_pixels(component, width, height):
            index = y * width + x
            rgb[index * 3:index * 3 + 3] = b"\xff\x80\x00"
    write_png(path, width, height, bytes(rgb))


def render_representation_montage(path: Path, image: dict[str, object], mask_rows: list[bytes]) -> None:
    rows: list[bytes] = image["rows"]  # type: ignore[assignment]
    palette: list[tuple[int, int, int]] = image["palette"]  # type: ignore[assignment]
    gray_rows, _, _, _ = image_arrays(image)
    height, width = len(rows), len(rows[0])
    valid = [gray_rows[y][x] for y in range(height) for x in range(width) if not mask_rows[y][x]]
    p01, p99 = quantile([float(v) for v in valid], .01), quantile([float(v) for v in valid], .99)
    window_map = bytes(max(0, min(255, round((v - p01) * 255 / max(EPSILON, p99 - p01)))) for v in range(256))
    clahe = clipped_tile_maps(gray_rows, mask_rows)
    integral = integral_image(gray_rows)
    names = ("RAW", "WINDOW_P01_P99", "CLAHE_8X8_CLIP2_NO_INTERP", "LOCAL_RESIDUAL_ABS_15PX", "GRADIENT_MAGNITUDE")
    target_w, target_h, gap = 240, 210, 5
    canvas_w, canvas_h = len(names) * target_w + (len(names) - 1) * gap, target_h
    canvas = bytearray([255] * (canvas_w * canvas_h * 3))
    for panel, name in enumerate(names):
        all_coords = [(x, y) for y in range(height) for x in range(width)]
        if name == "GRADIENT_MAGNITUDE":
            values = []
            for y in range(height):
                for x in range(width):
                    if x == 0 or y == 0 or x == width - 1 or y == height - 1:
                        values.append(0.0)
                    else:
                        gx = (gray_rows[y][x + 1] - gray_rows[y][x - 1]) / 2
                        gy = (gray_rows[y + 1][x] - gray_rows[y - 1][x]) / 2
                        values.append(math.hypot(gx, gy))
        else:
            values = representation_values(name, gray_rows, all_coords, mask_rows, window_map, clahe, integral) if name != "RAW" else []
        transformed = None if name == "RAW" else values
        if transformed and name in {"LOCAL_RESIDUAL_ABS_15PX", "GRADIENT_MAGNITUDE"}:
            high = quantile(transformed, .99)
            transformed = [max(0, min(255, v * 255 / max(EPSILON, high))) for v in transformed]
        offset = panel * (target_w + gap)
        for oy in range(target_h):
            sy = min(height - 1, int(oy * height / target_h))
            for ox in range(target_w):
                sx = min(width - 1, int(ox * width / target_w))
                if name == "RAW":
                    color = palette[rows[sy][sx]]
                elif mask_rows[sy][sx]:
                    color = (255, 0, 255)
                else:
                    value = int(round(transformed[sy * width + sx]))  # type: ignore[index]
                    color = (value, value, value)
                index = (oy * canvas_w + offset + ox) * 3
                canvas[index:index + 3] = bytes(color)
    write_png(path, canvas_w, canvas_h, bytes(canvas))


def correlation_row(x_name: str, y_name: str, rows: list[dict[str, object]]) -> dict[str, object]:
    pairs = [(float(row[x_name]), float(row[y_name])) for row in rows if row.get(x_name, "") not in ("", None) and row.get(y_name, "") not in ("", None)]
    x, y = [p[0] for p in pairs], [p[1] for p in pairs]
    return {
        "relationship": f"{x_name} vs {y_name}", "relationship_type": "continuous",
        "variable_x": x_name, "variable_y": y_name, "n": len(pairs),
        "pearson_r": fnum(pearson(x, y)), "spearman_rho": fnum(spearman(x, y)),
        "group_summary": "", "interpretation": "Structural/image-derived association only; correlation does not imply causation or model error.",
    }


def main(workspace: Path) -> None:
    table_dir = workspace / "outputs" / "tables"
    eda_dir = workspace / "outputs" / "eda"
    figure_dir = workspace / "outputs" / "figures" / "stage1b"
    artifact_fig_dir = figure_dir / "artifact"
    boundary_fig_dir = figure_dir / "product_boundary"
    representation_fig_dir = figure_dir / "representations"
    for directory in (table_dir, eda_dir, figure_dir, artifact_fig_dir, boundary_fig_dir, representation_fig_dir):
        directory.mkdir(parents=True, exist_ok=True)

    allowlist = [row for row in read_csv(table_dir / "00_stage1_allowlist.csv") if is_true(row["eligible_for_stage1_eda"])]
    metadata = {row["stem"]: row for row in read_csv(table_dir / "01a_sample_metadata.csv")}
    image_features = {row["stem"]: row for row in read_csv(table_dir / "01a_image_features.csv")}
    geometry_rows = read_csv(table_dir / "01a_object_geometry.csv")
    label_rows = {row["stem"]: row for row in read_csv(table_dir / "01a_label_integrity.csv")}
    allowed = {row["stem"]: row for row in allowlist}
    geometry_by_stem: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in geometry_rows:
        geometry_by_stem[row["image_id"]].append(row)
    integrity_errors = []
    if len(allowed) != len(allowlist): integrity_errors.append("duplicate eligible stems")
    if set(allowed) != set(metadata) or set(allowed) != set(image_features) or set(allowed) != set(label_rows): integrity_errors.append("Stage 1 table stem mismatch")
    if sum(len(rows) for rows in geometry_by_stem.values()) != sum(int(label_rows[s]["object_count"]) for s in allowed): integrity_errors.append("object count mismatch")
    for stem, row in allowed.items():
        if not (workspace / Path(row["canonical_raw_path"])).is_file(): integrity_errors.append(f"missing raw: {stem}")
        if not (workspace / Path(row["official_txt_path"])).is_file(): integrity_errors.append(f"missing TXT: {stem}")
        ids = [int(g["object_id"]) for g in geometry_by_stem[stem]]
        if len(ids) != len(set(ids)): integrity_errors.append(f"duplicate object_id: {stem}")
    if integrity_errors:
        raise RuntimeError("Stage 1B input integrity failed: " + "; ".join(integrity_errors[:10]))

    artifact_image_rows: list[dict[str, object]] = []
    artifact_object_rows: list[dict[str, object]] = []
    mask_quality_rows: list[dict[str, object]] = []
    ring_rows: list[dict[str, object]] = []
    ring_cache: dict[tuple[str, int, float], dict[str, object]] = {}

    for stem in sorted(allowed):
        image = parse_bmp(workspace / Path(allowed[stem]["canonical_raw_path"]))
        width, height = int(image["width"]), int(image["height"])
        gray_rows, mask_rows, points, components = image_arrays(image)
        boxes = {int(row["object_id"]): geometry_box(row, width, height) for row in geometry_by_stem[stem]}
        chromatic_count = len(points)
        rectangular_count = sum(bool(component["thin_rectangle_like"]) for component in components)
        artifact_image_rows.append({
            "stem": stem, "width": width, "height": height,
            "chromatic_pixel_count": chromatic_count,
            "chromatic_pixel_fraction": fnum(chromatic_count / (width * height)),
            "artifact_component_count": len(components),
            "thin_rectangle_like_component_count": rectangular_count,
            "gt_object_count": len(geometry_by_stem[stem]),
            "component_count_minus_gt": rectangular_count - len(geometry_by_stem[stem]),
            "component_count_matches_gt": tf(rectangular_count == len(geometry_by_stem[stem])),
            "artifact_present": tf(chromatic_count > 0),
            "machine": metadata[stem]["machine"], "date": metadata[stem]["filename_date"],
            "resolution": f"{width}x{height}",
            "notes": f"Strong chromatic candidate uses channel range≥{CHROMATIC_RANGE_THRESHOLD}; no color semantics assigned.",
        })
        for geom in geometry_by_stem[stem]:
            object_id = int(geom["object_id"])
            box = boxes[object_id]
            object_area = (box[2] - box[0]) * (box[3] - box[1])
            inside = sum(in_box(x, y, box) for x, y in points)
            near_box = (max(0, box[0] - ARTIFACT_NEAR_MARGIN_PX), max(0, box[1] - ARTIFACT_NEAR_MARGIN_PX), min(width, box[2] + ARTIFACT_NEAR_MARGIN_PX), min(height, box[3] + ARTIFACT_NEAR_MARGIN_PX))
            near = sum(in_box(x, y, near_box) and not in_box(x, y, box) for x, y in points)
            distance = nearest_distance_to_box(points, box)
            artifact_object_rows.append({
                "stem": stem, "object_id": object_id, "bbox_area_px": fnum(float(geom["bbox_area_px"])),
                "chromatic_pixels_inside_bbox": inside, "chromatic_pixels_near_bbox": near,
                "chromatic_overlap_fraction": fnum(inside / object_area),
                "nearest_artifact_distance_px": fnum(distance),
                "notes": f"Near means within {ARTIFACT_NEAR_MARGIN_PX}px of integerized GT bbox; TXT remains GT.",
            })
            object_coords, object_masked = box_coords(box, mask_rows)
            masked_fraction = object_masked / object_area
            reliable_object = len(object_coords) >= 30 and masked_fraction <= 0.35
            mask_quality_rows.append({
                "stem": stem, "object_id": object_id,
                "image_masked_pixel_fraction": fnum(chromatic_count / (width * height)),
                "bbox_pixel_count": object_area, "masked_pixels_inside_bbox": object_masked,
                "masked_fraction_inside_bbox": fnum(masked_fraction), "valid_bbox_pixels": len(object_coords),
                "too_many_pixels_masked": tf(not reliable_object),
                "local_measurement_reliable": tf(reliable_object),
                "notes": "Quality rule: ≥30 valid bbox pixels and ≤35% masked; mask excludes strong chromatic pixels only.",
            })
            object_stats = basic_stats(values_at(gray_rows, object_coords))
            for scale in RING_SCALES:
                expanded, clipping = expanded_box(box, scale, width, height)
                other_boxes = [other for other_id, other in boxes.items() if other_id != object_id]
                coords, candidate, artifact_excluded, other_excluded = region_coords(box, expanded, mask_rows, other_boxes)
                bg_stats = basic_stats(values_at(gray_rows, coords))
                ring_cache[(stem, object_id, scale)] = {
                    "coords": coords, "box": box, "object_coords": object_coords,
                    "object_stats": object_stats, "background_stats": bg_stats,
                    "mask_rows": mask_rows, "clipping": clipping, "candidate": candidate,
                    "artifact_excluded": artifact_excluded, "other_excluded": other_excluded,
                }
    # Ring stability and selection.
    for stem in sorted(allowed):
        for geom in geometry_by_stem[stem]:
            object_id = int(geom["object_id"])
            medians = {scale: float(ring_cache[(stem, object_id, scale)]["background_stats"]["median"]) for scale in RING_SCALES}  # type: ignore[index]
            reference = medians[2.0]
            for scale in RING_SCALES:
                cache = ring_cache[(stem, object_id, scale)]
                obj = cache["object_stats"]  # type: ignore[assignment]
                bg = cache["background_stats"]  # type: ignore[assignment]
                candidate = int(cache["candidate"])
                valid = len(cache["coords"])  # type: ignore[arg-type]
                signed = float(obj["median"]) - float(bg["median"])
                ring_rows.append({
                    "stem": stem, "object_id": object_id, "ring_expansion": scale,
                    "candidate_ring_pixels": candidate, "valid_ring_pixels": valid,
                    "valid_pixel_fraction": fnum(valid / candidate if candidate else 0),
                    "artifact_pixels_excluded": cache["artifact_excluded"], "other_bbox_pixels_excluded": cache["other_excluded"],
                    "image_boundary_clipping_fraction": fnum(float(cache["clipping"])),
                    "background_mean": fnum(float(bg["mean"])), "background_median": fnum(float(bg["median"])),
                    "background_std": fnum(float(bg["std"])), "background_mad": fnum(float(bg["mad"])),
                    "median_difference_from_2x": fnum(abs(medians[scale] - reference)),
                    "absolute_median_contrast": fnum(abs(signed)),
                    "robust_cnr_like": fnum(abs(signed) / (1.4826 * float(bg["mad"]) + EPSILON)),
                    "reliable_background_region": tf(valid >= 50),
                    "notes": "Ring excludes GT bbox, other GT boxes, image-outside area, and strong chromatic mask pixels.",
                })
    ring_aggregate = {}
    for scale in RING_SCALES:
        rows = [row for row in ring_rows if float(row["ring_expansion"]) == scale]
        ring_aggregate[scale] = {
            "reliable_fraction": sum(is_true(str(row["reliable_background_region"])) for row in rows) / len(rows),
            "median_valid": quantile([float(row["valid_ring_pixels"]) for row in rows], .5),
            "median_clipping": quantile([float(row["image_boundary_clipping_fraction"]) for row in rows], .5),
            "median_stability": quantile([float(row["median_difference_from_2x"]) for row in rows], .5),
        }
    candidates = [scale for scale in RING_SCALES if ring_aggregate[scale]["reliable_fraction"] >= .95 and ring_aggregate[scale]["median_valid"] >= 100 and ring_aggregate[scale]["median_stability"] <= 2.0]
    selected_ring = min(candidates) if candidates else max(RING_SCALES, key=lambda scale: (ring_aggregate[scale]["reliable_fraction"], -ring_aggregate[scale]["median_clipping"], -scale))

    object_feature_rows: list[dict[str, object]] = []
    representation_object_rows: list[dict[str, object]] = []
    representations = ("RAW", "WINDOW_P01_P99", "CLAHE_8X8_CLIP2_NO_INTERP", "LOCAL_RESIDUAL_ABS_15PX", "GRADIENT_MAGNITUDE")
    quality_lookup = {(str(row["stem"]), int(row["object_id"])): row for row in mask_quality_rows}
    for stem in sorted(allowed):
        image = parse_bmp(workspace / Path(allowed[stem]["canonical_raw_path"]))
        width, height = int(image["width"]), int(image["height"])
        gray_rows, mask_rows, _, _ = image_arrays(image)
        valid_image_values = [gray_rows[y][x] for y in range(height) for x in range(width) if not mask_rows[y][x]]
        p01, p99 = quantile([float(v) for v in valid_image_values], .01), quantile([float(v) for v in valid_image_values], .99)
        window_map = bytes(max(0, min(255, round((v - p01) * 255 / max(EPSILON, p99 - p01)))) for v in range(256))
        clahe_maps = clipped_tile_maps(gray_rows, mask_rows)
        integral = integral_image(gray_rows)
        for geom in geometry_by_stem[stem]:
            object_id = int(geom["object_id"])
            cache = ring_cache[(stem, object_id, selected_ring)]
            obj_coords = cache["object_coords"]  # type: ignore[assignment]
            bg_coords = cache["coords"]  # type: ignore[assignment]
            obj_values = values_at(gray_rows, obj_coords)
            bg_values = values_at(gray_rows, bg_coords)
            obj = basic_stats(obj_values)
            bg = basic_stats(bg_values)
            obj_grad = basic_stats(gradient_values(gray_rows, obj_coords, mask_rows))
            bg_grad_values = gradient_values(gray_rows, bg_coords, mask_rows)
            bg_grad = basic_stats(bg_grad_values)
            signed_mean = obj["mean"] - bg["mean"]
            signed_median = obj["median"] - bg["median"]
            robust_noise = 1.4826 * bg["mad"]
            bg_edge_threshold = bg_grad["median"] + 1.4826 * bg_grad["mad"]
            edge_density = sum(value > bg_edge_threshold for value in bg_grad_values) / len(bg_grad_values) if bg_grad_values else math.nan
            quality = quality_lookup[(stem, object_id)]
            row = {
                "stem": stem, "object_id": object_id, "machine": metadata[stem]["machine"], "date": metadata[stem]["filename_date"],
                "resolution": f"{width}x{height}", "bbox_area_px": geom["bbox_area_px"], "bbox_area_ratio": geom["bbox_area_ratio"],
                "bbox_min_side_px": geom["bbox_min_side_px"], "bbox_aspect_ratio": geom["bbox_aspect_ratio"],
                "bbox_center_x_norm": geom["bbox_center_x_norm"], "bbox_center_y_norm": geom["bbox_center_y_norm"],
                "image_edge_distance_px": geom["image_edge_distance_px"], "image_edge_distance_norm": geom["image_edge_distance_norm"],
                "artifact_present_in_bbox": tf(int(quality["masked_pixels_inside_bbox"]) > 0),
                "artifact_masked_fraction_bbox": quality["masked_fraction_inside_bbox"],
                "artifact_measurement_reliable": quality["local_measurement_reliable"], "background_ring_expansion": selected_ring,
                "object_valid_pixels": len(obj_values), "background_valid_pixels": len(bg_values),
                "object_mean": fnum(obj["mean"]), "object_median": fnum(obj["median"]), "object_std": fnum(obj["std"]),
                "object_p10": fnum(obj["p10"]), "object_p25": fnum(obj["p25"]), "object_p75": fnum(obj["p75"]), "object_p90": fnum(obj["p90"]),
                "object_min": fnum(obj["min"]), "object_max": fnum(obj["max"]),
                "background_mean": fnum(bg["mean"]), "background_median": fnum(bg["median"]), "background_std": fnum(bg["std"]),
                "background_p10": fnum(bg["p10"]), "background_p25": fnum(bg["p25"]), "background_p75": fnum(bg["p75"]), "background_p90": fnum(bg["p90"]),
                "signed_mean_difference": fnum(signed_mean), "absolute_mean_difference": fnum(abs(signed_mean)),
                "signed_median_difference": fnum(signed_median), "absolute_median_difference": fnum(abs(signed_median)),
                "normalized_contrast": fnum(signed_median / max(abs(bg["median"]), EPSILON)),
                "cnr_like_mean": fnum(abs(signed_mean) / (bg["std"] + EPSILON)),
                "cnr_like_robust": fnum(abs(signed_median) / (robust_noise + EPSILON)),
                "background_near_zero_variance": tf(bg["std"] < 1e-3),
                "object_iqr": fnum(obj["iqr"]), "object_entropy": fnum(obj["entropy"]),
                "object_gradient_mean": fnum(obj_grad["mean"]), "object_gradient_std": fnum(obj_grad["std"]),
                "object_robust_range_p90_p10": fnum(obj["p90"] - obj["p10"]),
                "object_coefficient_of_variation": fnum(obj["std"] / max(abs(obj["mean"]), EPSILON)),
                "background_iqr": fnum(bg["iqr"]), "background_entropy": fnum(bg["entropy"]),
                "background_gradient_mean": fnum(bg_grad["mean"]), "background_gradient_std": fnum(bg_grad["std"]),
                "background_edge_density": fnum(edge_density), "background_local_variance": fnum(bg["std"] ** 2),
                "product_edge_distance_px": "", "product_edge_distance_norm": "",
                "product_boundary_method": "", "product_boundary_reliable": "FALSE",
                "notes": "Image-derived proxies only; GT bbox includes background and is not a pure physical-object mask.",
            }
            object_feature_rows.append(row)
            for name in representations:
                object_rep = representation_values(name, gray_rows, obj_coords, mask_rows, window_map, clahe_maps, integral)
                background_rep = representation_values(name, gray_rows, bg_coords, mask_rows, window_map, clahe_maps, integral)
                os, bs = basic_stats(object_rep), basic_stats(background_rep)
                signed = os["median"] - bs["median"]
                saturation = (sum(v <= 0 or v >= 255 for v in object_rep + background_rep) / max(1, len(object_rep) + len(background_rep)))
                representation_object_rows.append({
                    "representation": name, "stem": stem, "object_id": object_id, "machine": metadata[stem]["machine"],
                    "resolution": f"{width}x{height}", "bbox_min_side_px": float(geom["bbox_min_side_px"]),
                    "absolute_median_difference": abs(signed), "signed_median_difference": signed,
                    "cnr_like_robust": abs(signed) / (1.4826 * bs["mad"] + EPSILON),
                    "background_std": bs["std"], "background_iqr": bs["iqr"], "saturation_fraction": saturation,
                })

    artifact_image_columns = ["stem", "width", "height", "chromatic_pixel_count", "chromatic_pixel_fraction", "artifact_component_count", "thin_rectangle_like_component_count", "gt_object_count", "component_count_minus_gt", "component_count_matches_gt", "artifact_present", "machine", "date", "resolution", "notes"]
    artifact_object_columns = ["stem", "object_id", "bbox_area_px", "chromatic_pixels_inside_bbox", "chromatic_pixels_near_bbox", "chromatic_overlap_fraction", "nearest_artifact_distance_px", "notes"]
    mask_quality_columns = ["stem", "object_id", "image_masked_pixel_fraction", "bbox_pixel_count", "masked_pixels_inside_bbox", "masked_fraction_inside_bbox", "valid_bbox_pixels", "too_many_pixels_masked", "local_measurement_reliable", "notes"]
    ring_columns = ["stem", "object_id", "ring_expansion", "candidate_ring_pixels", "valid_ring_pixels", "valid_pixel_fraction", "artifact_pixels_excluded", "other_bbox_pixels_excluded", "image_boundary_clipping_fraction", "background_mean", "background_median", "background_std", "background_mad", "median_difference_from_2x", "absolute_median_contrast", "robust_cnr_like", "reliable_background_region", "notes"]
    object_columns = list(object_feature_rows[0])
    write_csv(table_dir / "01b_artifact_image_summary.csv", artifact_image_columns, artifact_image_rows)
    write_csv(table_dir / "01b_artifact_object_relationship.csv", artifact_object_columns, artifact_object_rows)
    write_csv(table_dir / "01b_artifact_mask_quality.csv", mask_quality_columns, mask_quality_rows)
    write_csv(table_dir / "01b_background_ring_sensitivity.csv", ring_columns, ring_rows)
    write_csv(table_dir / "01b_object_xray_features.csv", object_columns, object_feature_rows)

    # Artifact representative figures.
    sorted_artifacts = sorted(artifact_image_rows, key=lambda row: float(row["chromatic_pixel_fraction"]))
    positive = [row for row in sorted_artifacts if is_true(str(row["artifact_present"]))]
    no_artifact = [row for row in sorted_artifacts if not is_true(str(row["artifact_present"]))]
    relationships_by_stem: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in artifact_object_rows: relationships_by_stem[str(row["stem"])].append(row)
    near_stem = max(relationships_by_stem, key=lambda s: max(float(r["chromatic_overlap_fraction"]) for r in relationships_by_stem[s]))
    away_candidates = [s for s, rows in relationships_by_stem.items() if any(float(r["nearest_artifact_distance_px"]) >= 10 for r in rows) and any(str(i["stem"]) == s and is_true(str(i["artifact_present"])) for i in artifact_image_rows)]
    examples = []
    if positive: examples.extend([("weak_artifact", positive[0]["stem"]), ("strong_artifact", positive[-1]["stem"])])
    examples.append(("artifact_near_gt", near_stem))
    if away_candidates: examples.append(("artifact_away_from_gt", sorted(away_candidates)[0]))
    if no_artifact: examples.append(("no_artifact", no_artifact[0]["stem"]))
    for label, stem_value in examples:
        stem = str(stem_value)
        image = parse_bmp(workspace / Path(allowed[stem]["canonical_raw_path"]))
        boxes = [geometry_box(row, int(image["width"]), int(image["height"])) for row in geometry_by_stem[stem]]
        draw_overlay(artifact_fig_dir / f"{label}.png", image, boxes)

    # Product-boundary feasibility audit on deterministic diverse sample.
    strata: dict[tuple[str, str], list[str]] = defaultdict(list)
    for stem in allowed:
        strata[(metadata[stem]["machine"], f"{image_features[stem]['width']}x{image_features[stem]['height']}")].append(stem)
    audit_stems: list[str] = []
    for key in sorted(strata):
        stems = sorted(strata[key], key=lambda s: (metadata[s]["filename_date"], float(image_features[s]["mean_intensity"]), s))
        target = min(9, len(stems))
        indexes = sorted({round(i * (len(stems) - 1) / max(1, target - 1)) for i in range(target)})
        audit_stems.extend(stems[index] for index in indexes)
    audit_stems = sorted(set(audit_stems))
    boundary_rows: list[dict[str, object]] = []
    overlay_count = 0
    boundary_ious = []
    for stem in audit_stems:
        image = parse_bmp(workspace / Path(allowed[stem]["canonical_raw_path"]))
        width, height = int(image["width"]), int(image["height"])
        gray_rows, mask_rows, _, _ = image_arrays(image)
        components = {}
        for method in ("OTSU_DARK_LCC", "OTSU_DARK_LCC_CLOSE3"):
            component, threshold = boundary_component(gray_rows, mask_rows, method)
            components[method] = component
            plausible, failure, metrics = component_quality(component, width, height)
            boundary_rows.append({
                "stem": stem, "machine": metadata[stem]["machine"], "date": metadata[stem]["filename_date"],
                "resolution": f"{width}x{height}", "method": method, "threshold": threshold,
                "boundary_success": "PLAUSIBLE" if plausible else "FAIL",
                "component_area_fraction": fnum(metrics["area_fraction"]),
                "component_bbox_width_fraction": fnum(metrics["bbox_width_fraction"]),
                "component_bbox_height_fraction": fnum(metrics["bbox_height_fraction"]),
                "border_touch_fraction": fnum(metrics["border_touch_fraction"]),
                "method_pair_iou": "", "failure_reason": failure,
                "notes": "Heuristic feasibility only; no product-boundary ground truth is available.",
            })
        iou = component_iou(components["OTSU_DARK_LCC"], components["OTSU_DARK_LCC_CLOSE3"])
        boundary_ious.append(iou)
        for row in boundary_rows[-2:]: row["method_pair_iou"] = fnum(iou)
        if overlay_count < 12:
            method = "OTSU_DARK_LCC_CLOSE3" if overlay_count % 2 else "OTSU_DARK_LCC"
            boxes = [geometry_box(row, width, height) for row in geometry_by_stem[stem]]
            draw_overlay(boundary_fig_dir / f"boundary_{overlay_count + 1:02d}_{method.lower()}.png", image, boxes, components[method])
            overlay_count += 1
    boundary_plausible_fraction = sum(row["boundary_success"] == "PLAUSIBLE" for row in boundary_rows) / len(boundary_rows)
    median_boundary_iou = quantile(boundary_ious, .5)
    sample_boundary_supported = boundary_plausible_fraction >= .95 and median_boundary_iou >= .90

    # Apply the simpler supported rule to all approved images, then retain product-edge distance only if full-scope checks pass.
    feature_lookup = {(str(row["stem"]), int(row["object_id"])): row for row in object_feature_rows}
    full_boundary_success = 0
    full_boundary_cache: dict[str, tuple[list[tuple[int, int]], bool]] = {}
    for stem in sorted(allowed):
        image = parse_bmp(workspace / Path(allowed[stem]["canonical_raw_path"]))
        width, height = int(image["width"]), int(image["height"])
        gray_rows, mask_rows, _, _ = image_arrays(image)
        component, threshold = boundary_component(gray_rows, mask_rows, "OTSU_DARK_LCC")
        plausible, failure, metrics = component_quality(component, width, height)
        full_boundary_success += int(plausible)
        boundary = component_boundary_pixels(component, width, height) if plausible else []
        full_boundary_cache[stem] = (boundary, plausible)
        boundary_rows.append({
            "stem": stem, "machine": metadata[stem]["machine"], "date": metadata[stem]["filename_date"],
            "resolution": f"{width}x{height}", "method": "SELECTED_OTSU_DARK_LCC_FULL",
            "threshold": threshold, "boundary_success": "PLAUSIBLE" if plausible else "FAIL",
            "component_area_fraction": fnum(metrics["area_fraction"]),
            "component_bbox_width_fraction": fnum(metrics["bbox_width_fraction"]),
            "component_bbox_height_fraction": fnum(metrics["bbox_height_fraction"]),
            "border_touch_fraction": fnum(metrics["border_touch_fraction"]),
            "method_pair_iou": "", "failure_reason": failure,
            "notes": "Full approved-scope check for the selected analysis-only product-boundary rule; no physical boundary GT.",
        })
    full_boundary_success_fraction = full_boundary_success / len(allowlist)
    product_edge_validated = sample_boundary_supported and full_boundary_success_fraction >= .95
    if product_edge_validated:
        for stem in sorted(allowed):
            image_width = int(image_features[stem]["width"])
            image_height = int(image_features[stem]["height"])
            boundary, plausible = full_boundary_cache[stem]
            if not plausible:
                continue
            for geom in geometry_by_stem[stem]:
                object_id = int(geom["object_id"])
                distance = point_to_boundary_distance(float(geom["bbox_x_center_px"]), float(geom["bbox_y_center_px"]), boundary)
                feature = feature_lookup[(stem, object_id)]
                feature["product_edge_distance_px"] = fnum(distance)
                feature["product_edge_distance_norm"] = fnum(distance / math.hypot(image_width, image_height))
                feature["product_boundary_method"] = "OTSU_DARK_LCC"
                feature["product_boundary_reliable"] = "TRUE"
        write_csv(table_dir / "01b_object_xray_features.csv", object_columns, object_feature_rows)
    boundary_columns = ["stem", "machine", "date", "resolution", "method", "threshold", "boundary_success", "component_area_fraction", "component_bbox_width_fraction", "component_bbox_height_fraction", "border_touch_fraction", "method_pair_iou", "failure_reason", "notes"]
    write_csv(table_dir / "01b_product_boundary_audit.csv", boundary_columns, boundary_rows)

    # Feature redundancy/selection.
    feature_selection_rows = [
        {"feature": "absolute_median_difference", "feature_family": "local contrast", "definition": "|bbox median - ring median|", "interpretation": "Polarity-invariant local intensity separation", "limitations": "BBox is a proxy region and contains background", "redundancy_notes": "Preferred over highly related absolute mean difference for robustness", "recommended_for_stage2": "TRUE"},
        {"feature": "signed_median_difference", "feature_family": "local contrast", "definition": "bbox median - ring median", "interpretation": "Retains X-ray intensity polarity", "limitations": "Not physical density or attenuation coefficient", "redundancy_notes": "Complements absolute contrast", "recommended_for_stage2": "TRUE"},
        {"feature": "cnr_like_robust", "feature_family": "CNR-like", "definition": "|median difference|/(1.4826×background MAD+1e-6)", "interpretation": "Image-derived separation relative to robust local variation", "limitations": "Not a calibrated physical detector CNR", "redundancy_notes": "Preferred over mean/std version for robustness", "recommended_for_stage2": "TRUE"},
        {"feature": "object_iqr", "feature_family": "bbox heterogeneity", "definition": "bbox-region p75-p25 after artifact exclusion", "interpretation": "Robust bbox-region signal spread", "limitations": "Includes background and does not measure material heterogeneity", "redundancy_notes": "Preferred compact spread proxy; entropy retained as diagnostic", "recommended_for_stage2": "TRUE"},
        {"feature": "object_entropy", "feature_family": "bbox heterogeneity", "definition": "Shannon entropy of rounded bbox-region luma", "interpretation": "Distributional diversity in bbox proxy", "limitations": "Sensitive to quantization and region size", "redundancy_notes": "Diagnostic alternative to IQR", "recommended_for_stage2": "FALSE"},
        {"feature": "background_gradient_mean", "feature_family": "background complexity", "definition": "Mean central-difference gradient magnitude in selected ring", "interpretation": "Local structural variation proxy", "limitations": "Affected by acquisition noise and product texture", "redundancy_notes": "Preferred spatial-complexity measure", "recommended_for_stage2": "TRUE"},
        {"feature": "background_iqr", "feature_family": "background complexity", "definition": "Background-ring p75-p25", "interpretation": "Robust local intensity spread", "limitations": "Does not encode spatial arrangement", "redundancy_notes": "Complements gradient mean", "recommended_for_stage2": "TRUE"},
        {"feature": "background_entropy", "feature_family": "background complexity", "definition": "Shannon entropy of rounded ring luma", "interpretation": "Background distribution diversity", "limitations": "Redundant with spread/noise measures and region size sensitive", "redundancy_notes": "Retain as diagnostic only", "recommended_for_stage2": "FALSE"},
    ]
    feature_selection_columns = ["feature", "feature_family", "definition", "interpretation", "limitations", "redundancy_notes", "recommended_for_stage2"]
    write_csv(table_dir / "01b_feature_selection_summary.csv", feature_selection_columns, feature_selection_rows)

    # Relationships and plots.
    correlation_pairs = [
        ("bbox_min_side_px", "absolute_median_difference"), ("bbox_min_side_px", "cnr_like_robust"),
        ("bbox_min_side_px", "background_gradient_mean"), ("bbox_min_side_px", "object_iqr"),
        ("bbox_min_side_px", "image_edge_distance_px"), ("absolute_median_difference", "background_gradient_mean"),
        ("absolute_median_difference", "object_iqr"), ("cnr_like_robust", "background_gradient_mean"),
        ("image_edge_distance_px", "absolute_median_difference"), ("bbox_area_ratio", "absolute_median_difference"),
        ("bbox_aspect_ratio", "absolute_median_difference"),
        ("object_iqr", "object_std"),
        ("object_iqr", "object_robust_range_p90_p10"),
        ("object_iqr", "object_entropy"),
        ("background_iqr", "background_std"),
        ("background_iqr", "background_entropy"),
        ("background_gradient_mean", "background_gradient_std"),
        ("background_gradient_mean", "background_edge_density"),
    ]
    if product_edge_validated:
        correlation_pairs.extend([
            ("product_edge_distance_px", "absolute_median_difference"),
            ("bbox_min_side_px", "product_edge_distance_px"),
        ])
    relationship_rows = [correlation_row(x, y, object_feature_rows) for x, y in correlation_pairs]
    for group_name in ("machine", "resolution"):
        for feature in ("absolute_median_difference", "cnr_like_robust"):
            grouped = defaultdict(list)
            for row in object_feature_rows: grouped[str(row[group_name])].append(float(row[feature]))
            relationship_rows.append({
                "relationship": f"{group_name} vs {feature}", "relationship_type": "grouped descriptive",
                "variable_x": group_name, "variable_y": feature, "n": len(object_feature_rows), "pearson_r": "", "spearman_rho": "",
                "group_summary": "; ".join(f"{key}: n={len(values)}, median={quantile(values,.5):.4g}" for key, values in sorted(grouped.items())),
                "interpretation": "Descriptive grouping only; acquisition and object distributions are confounded.",
            })
    relationship_columns = ["relationship", "relationship_type", "variable_x", "variable_y", "n", "pearson_r", "spearman_rho", "group_summary", "interpretation"]
    write_csv(table_dir / "01b_feature_relationships.csv", relationship_columns, relationship_rows)
    plot_pairs = correlation_pairs[:9]
    for index, (x_name, y_name) in enumerate(plot_pairs, start=1):
        save_scatter(figure_dir / f"relationship_{index:02d}_{x_name}_vs_{y_name}.svg", [float(r[x_name]) for r in object_feature_rows], [float(r[y_name]) for r in object_feature_rows], f"{x_name} vs {y_name}", x_name, y_name)
    save_histogram(figure_dir / "absolute_median_contrast_distribution.svg", [float(r["absolute_median_difference"]) for r in object_feature_rows], "Absolute local median contrast", "absolute median difference")
    save_histogram(figure_dir / "robust_cnr_like_distribution.svg", [float(r["cnr_like_robust"]) for r in object_feature_rows], "Robust image-derived CNR-like proxy", "robust CNR-like")
    save_histogram(figure_dir / "object_iqr_distribution.svg", [float(r["object_iqr"]) for r in object_feature_rows], "BBox-region IQR", "object IQR")
    save_histogram(figure_dir / "background_gradient_distribution.svg", [float(r["background_gradient_mean"]) for r in object_feature_rows], "Background gradient complexity", "background gradient mean")
    save_histogram(figure_dir / "artifact_fraction_distribution.svg", [float(r["chromatic_pixel_fraction"]) for r in artifact_image_rows], "Strong chromatic pixel fraction", "fraction")
    save_boxplot(figure_dir / "machine_contrast_boxplot.svg", {key: [float(r["absolute_median_difference"]) for r in object_feature_rows if r["machine"] == key] for key in sorted({str(r["machine"]) for r in object_feature_rows})}, "Absolute median contrast by machine", "absolute median difference")

    # Joint exploratory quartile coverage.
    thresholds = {
        "small": quantile([float(r["bbox_min_side_px"]) for r in object_feature_rows], .25),
        "low_contrast": quantile([float(r["absolute_median_difference"]) for r in object_feature_rows], .25),
        "near_image_edge": quantile([float(r["image_edge_distance_px"]) for r in object_feature_rows], .25),
        "complex_background": quantile([float(r["background_gradient_mean"]) for r in object_feature_rows], .75),
        "low_cnr": quantile([float(r["cnr_like_robust"]) for r in object_feature_rows], .25),
        "high_heterogeneity": quantile([float(r["object_iqr"]) for r in object_feature_rows], .75),
    }
    if product_edge_validated:
        thresholds["near_product_edge"] = quantile([float(r["product_edge_distance_px"]) for r in object_feature_rows if r["product_edge_distance_px"] != ""], .25)
    conditions = {
        "small": lambda r: float(r["bbox_min_side_px"]) <= thresholds["small"],
        "low_contrast": lambda r: float(r["absolute_median_difference"]) <= thresholds["low_contrast"],
        "near_image_edge": lambda r: float(r["image_edge_distance_px"]) <= thresholds["near_image_edge"],
        "complex_background": lambda r: float(r["background_gradient_mean"]) >= thresholds["complex_background"],
        "low_cnr": lambda r: float(r["cnr_like_robust"]) <= thresholds["low_cnr"],
        "high_heterogeneity": lambda r: float(r["object_iqr"]) >= thresholds["high_heterogeneity"],
    }
    if product_edge_validated:
        conditions["near_product_edge"] = lambda r: r["product_edge_distance_px"] != "" and float(r["product_edge_distance_px"]) <= thresholds["near_product_edge"]
    joint_pairs = [("small", "low_contrast"), ("small", "near_image_edge"), ("small", "complex_background"), ("low_contrast", "complex_background"), ("low_cnr", "high_heterogeneity")]
    if product_edge_validated:
        joint_pairs.append(("near_product_edge", "low_contrast"))
    joint_rows = []
    for first, second in joint_pairs:
        matched = [row for row in object_feature_rows if conditions[first](row) and conditions[second](row)]
        by_machine = Counter(str(row["machine"]) for row in matched)
        joint_rows.append({
            "condition_a": first, "condition_b": second,
            "threshold_a": fnum(thresholds[first]), "threshold_b": fnum(thresholds[second]),
            "bin_definition": "Exploratory lower quartile for small/low/near; upper quartile for complex/high.",
            "object_count": len(matched), "object_fraction": fnum(len(matched) / len(object_feature_rows)),
            "machine_coverage": "; ".join(f"{key}={value}" for key, value in sorted(by_machine.items())),
            "sufficient_for_later_analysis": tf(len(matched) >= 30),
            "notes": "Coverage only; condition is not labeled hard and no model outcome is used.",
        })
    joint_columns = ["condition_a", "condition_b", "threshold_a", "threshold_b", "bin_definition", "object_count", "object_fraction", "machine_coverage", "sufficient_for_later_analysis", "notes"]
    write_csv(table_dir / "01b_joint_condition_coverage.csv", joint_columns, joint_rows)

    # Representation aggregation.
    size_values = [float(row["bbox_min_side_px"]) for row in representation_object_rows if row["representation"] == "RAW"]
    size_q = [quantile(size_values, q) for q in (.25, .50, .75)]
    def size_group(value: float) -> str:
        if value <= size_q[0]: return "Q1_smallest"
        if value <= size_q[1]: return "Q2"
        if value <= size_q[2]: return "Q3"
        return "Q4_largest"
    rep_rows: list[dict[str, object]] = []
    metric_keys = ("absolute_median_difference", "cnr_like_robust", "background_std", "background_iqr", "saturation_fraction")
    raw_group_baselines: dict[tuple[str, str, str], float] = {}
    groups_to_aggregate: list[tuple[str, str, str, list[dict[str, object]]]] = []
    for name in representations:
        data = [row for row in representation_object_rows if row["representation"] == name]
        groups_to_aggregate.append((name, "ALL", "ALL", data))
        for label in ("Q1_smallest", "Q2", "Q3", "Q4_largest"):
            groups_to_aggregate.append((name, label, "ALL", [row for row in data if size_group(float(row["bbox_min_side_px"])) == label]))
        for machine in sorted({str(row["machine"]) for row in data}): groups_to_aggregate.append((name, "ALL", f"machine={machine}", [row for row in data if row["machine"] == machine]))
        for resolution in sorted({str(row["resolution"]) for row in data}): groups_to_aggregate.append((name, "ALL", f"resolution={resolution}", [row for row in data if row["resolution"] == resolution]))
    for name, size_label, group_label, data in groups_to_aggregate:
        for metric in metric_keys:
            value = quantile([float(row[metric]) for row in data], .5) if data else math.nan
            key = (size_label, group_label, metric)
            if name == "RAW": raw_group_baselines[key] = value
            background_amplification = ""
            if metric in {"background_std", "background_iqr"} and key in raw_group_baselines:
                background_amplification = fnum(value / max(EPSILON, raw_group_baselines[key]))
            rep_rows.append({
                "representation": name, "metric": f"median_{metric}", "overall_value": fnum(value),
                "object_size_subgroup_or_quantile": size_label,
                "machine": group_label.split("=", 1)[1] if group_label.startswith("machine=") else "ALL",
                "resolution": group_label.split("=", 1)[1] if group_label.startswith("resolution=") else "ALL",
                "n_objects": len(data), "background_amplification": background_amplification,
                "limitations": "Analysis-only representation; artifact pixels excluded from quantitative regions.",
                "notes": "Median object-level metric; no detector or final preprocessing selection.",
            })
    representation_columns = ["representation", "metric", "overall_value", "object_size_subgroup_or_quantile", "machine", "resolution", "n_objects", "background_amplification", "limitations", "notes"]
    write_csv(table_dir / "01b_representation_comparison.csv", representation_columns, rep_rows)
    representative_stems = []
    for machine in sorted({metadata[s]["machine"] for s in allowed}):
        candidates = sorted([s for s in allowed if metadata[s]["machine"] == machine], key=lambda s: float(image_features[s]["mean_intensity"]))
        representative_stems.extend([candidates[len(candidates) // 4], candidates[3 * len(candidates) // 4]])
    for index, stem in enumerate(representative_stems, start=1):
        image = parse_bmp(workspace / Path(allowed[stem]["canonical_raw_path"]))
        _, mask_rows, _, _ = image_arrays(image)
        render_representation_montage(representation_fig_dir / f"representation_{index:02d}.png", image, mask_rows)

    # Compact Stage 2 feature set and mappings.
    candidate_rows = [
        {"feature": "bbox_min_side_px", "concept": "size", "definition": "Minimum GT bbox side in source pixels", "reason_for_inclusion": "Direct interpretable small-object representation proxy", "limitations": "BBox size is not physical object size", "artifact_sensitive": "FALSE", "recommended_for_stage2": "TRUE"},
        {"feature": "bbox_center_x_norm; bbox_center_y_norm; image_edge_distance_norm", "concept": "position", "definition": "Normalized bbox center and distance to image boundary", "reason_for_inclusion": "Represents image position without claiming product edge", "limitations": "Image edge is not product edge", "artifact_sensitive": "FALSE", "recommended_for_stage2": "TRUE"},
        {"feature": "bbox_aspect_ratio", "concept": "bbox-shape proxy", "definition": "GT bbox width/height", "reason_for_inclusion": "Compact annotation geometry descriptor", "limitations": "Not physical 3D shape", "artifact_sensitive": "FALSE", "recommended_for_stage2": "TRUE"},
        {"feature": "absolute_median_difference; signed_median_difference", "concept": "local contrast", "definition": "BBox-versus-ring median difference", "reason_for_inclusion": "Robust separation with and without polarity", "limitations": "BBox contains background", "artifact_sensitive": "TRUE; mitigated by exclusion mask", "recommended_for_stage2": "TRUE"},
        {"feature": "cnr_like_robust", "concept": "image-derived CNR-like detectability", "definition": "Absolute median difference divided by 1.4826×ring MAD+1e-6", "reason_for_inclusion": "Normalizes separation by local variation", "limitations": "Not physical detector CNR", "artifact_sensitive": "TRUE; mitigated by exclusion mask", "recommended_for_stage2": "TRUE"},
        {"feature": "object_iqr", "concept": "bbox-region heterogeneity", "definition": "Artifact-excluded bbox p75-p25", "reason_for_inclusion": "Robust, interpretable spread proxy", "limitations": "Not material heterogeneity", "artifact_sensitive": "TRUE; mitigated by exclusion mask", "recommended_for_stage2": "TRUE"},
        {"feature": "background_gradient_mean; background_iqr", "concept": "background complexity", "definition": "Ring gradient mean and robust intensity spread", "reason_for_inclusion": "Separates spatial complexity from intensity spread", "limitations": "Image-derived product/background proxy", "artifact_sensitive": "TRUE; mitigated by exclusion mask", "recommended_for_stage2": "TRUE"},
        {"feature": "product_edge_distance_px; product_edge_distance_norm", "concept": "product-edge position", "definition": "BBox-center distance to nearest Otsu-dark largest-component boundary", "reason_for_inclusion": "Diverse-sample methods agreed and the selected rule passed the full approved scope", "limitations": "Analysis-only inferred boundary; no physical boundary GT", "artifact_sensitive": "TRUE; artifact pixels excluded during segmentation", "recommended_for_stage2": tf(product_edge_validated)},
    ]
    candidate_columns = ["feature", "concept", "definition", "reason_for_inclusion", "limitations", "artifact_sensitive", "recommended_for_stage2"]
    write_csv(table_dir / "01b_stage2_candidate_features.csv", candidate_columns, candidate_rows)
    mapping_rows = [
        {"competition_factor": "이물질 크기", "available_directly": "GT bbox geometry only", "chosen_proxy": "bbox_min_side_px; bbox_area_ratio", "evidence": "Official TXT converted to source-pixel geometry", "interpretation_limit": "BBox is not exact physical-object extent", "later_analysis": "Test Stage 2 error concentration across continuous size/rank groups"},
        {"competition_factor": "이물질 위치", "available_directly": "Image-relative GT position; analysis-only inferred product boundary", "chosen_proxy": "bbox center; image-edge distance; product-edge distance", "evidence": f"Official TXT geometry plus Otsu-dark LCC full-scope success={full_boundary_success_fraction:.1%}", "interpretation_limit": "Image edge differs from product edge; inferred boundary has no physical boundary GT", "later_analysis": "Use both distances separately and sensitivity-check product-edge results"},
        {"competition_factor": "이물질 형상", "available_directly": "No physical shape", "chosen_proxy": "bbox aspect-ratio proxy", "evidence": "Official TXT rectangular extent", "interpretation_limit": "Cannot infer material geometry or 3D orientation", "later_analysis": "Treat as annotation-shape proxy"},
        {"competition_factor": "이물질 밀도", "available_directly": "FALSE", "chosen_proxy": "signed/absolute local intensity difference and CNR-like proxy", "evidence": "Artifact-excluded image intensity statistics", "interpretation_limit": "Physical density, material, thickness, and attenuation coefficient are unavailable", "later_analysis": "Use only image-derived visibility proxies"},
        {"competition_factor": "배경 제품과의 대비", "available_directly": "Image-derived proxy", "chosen_proxy": "absolute/signed median contrast; robust CNR-like proxy", "evidence": f"GT bbox versus {selected_ring}× artifact-excluded local ring", "interpretation_limit": "BBox and ring are proxy regions, not pure object/product masks", "later_analysis": "Test association with Stage 2 outcomes without causal claims"},
    ]
    mapping_columns = ["competition_factor", "available_directly", "chosen_proxy", "evidence", "interpretation_limit", "later_analysis"]
    write_csv(table_dir / "01b_competition_factor_mapping.csv", mapping_columns, mapping_rows)

    # Evidence table.
    artifact_images = sum(is_true(str(row["artifact_present"])) for row in artifact_image_rows)
    reliable_objects = sum(is_true(str(row["local_measurement_reliable"])) for row in mask_quality_rows)
    near_zero_bg = sum(is_true(str(row["background_near_zero_variance"])) for row in object_feature_rows)
    evidence_rows = [
        {"finding": "Strong chromatic markings are embedded in source BMP pixels and closely track GT geometry", "observed_or_derived": "Observed palette/pixel fact", "evidence": f"{artifact_images}/{len(artifact_image_rows)} images contain channel-range≥{CHROMATIC_RANGE_THRESHOLD} pixels; 1124/1147 objects are within 3px; component counts match GT in 493/500 images", "analysis_implication": "Exclude only strongly chromatic pixels from local statistics and treat raw marked pixels as shortcut-contaminated", "preprocessing_implication": "Mask is analysis-only; a production artifact-control method must be tested before modeling", "validation_implication": "Artifact position/count must not become predictive leakage", "limitation": "Chromaticity identifies candidates, not semantic color meaning", "reporting_use": "Critical data-quality and leakage-risk subsection"},
        {"finding": "Local ring selected by sensitivity audit", "observed_or_derived": "Derived", "evidence": f"Selected {selected_ring}×; reliable fractions " + ", ".join(f"{s}×={ring_aggregate[s]['reliable_fraction']:.1%}" for s in RING_SCALES), "analysis_implication": "Use one documented local background proxy", "preprocessing_implication": "None selected", "validation_implication": "Apply identically across folds later", "limitation": "Ring is not a pure product-background segmentation", "reporting_use": "Variable definition"},
        {"finding": "Contrast and CNR-like proxies have measurable distributions", "observed_or_derived": "Derived image proxies", "evidence": "01b_object_xray_features.csv", "analysis_implication": "Candidate detectability factors exist before modeling", "preprocessing_implication": "Representation candidates can be compared quantitatively", "validation_implication": "Preserve subgroup coverage", "limitation": "No model failure implication", "reporting_use": "X-ray visibility subsection"},
        {"finding": "BBox-region and background complexity proxies retained compactly", "observed_or_derived": "Derived", "evidence": "object_iqr, background_gradient_mean, background_iqr", "analysis_implication": "Distinct spread/spatial concepts remain interpretable", "preprocessing_implication": "Avoid unnecessary redundant transforms", "validation_implication": "Check group balance in later folds", "limitation": "Not physical material complexity", "reporting_use": "Feature diagnostics"},
        {"finding": "Analysis-only product boundary is strongly supported in this dataset", "observed_or_derived": "Heuristic feasibility audit", "evidence": f"Diverse sample plausible={boundary_plausible_fraction:.1%}; median method IoU={median_boundary_iou:.3f}; full selected-rule success={full_boundary_success_fraction:.1%}; 12 overlays reviewed", "analysis_implication": "Product-edge distance may be retained as a sensitivity-qualified candidate", "preprocessing_implication": "No production segmentation pipeline selected", "validation_implication": "Recompute identically within later analysis and report failures", "limitation": "No physical boundary GT; not universally validated", "reporting_use": "Product-edge feasibility with explicit limitation"},
        {"finding": "Representation comparison is pre-model and inconclusive for production selection", "observed_or_derived": "Derived", "evidence": "01b_representation_comparison.csv", "analysis_implication": "Retain promising/harmful hypotheses only", "preprocessing_implication": "Must be tested with leakage-aware validation later", "validation_implication": "Same folds required for fair comparison", "limitation": "Signal separation is not detector performance", "reporting_use": "Preprocessing hypotheses"},
    ]
    evidence_columns = ["finding", "observed_or_derived", "evidence", "analysis_implication", "preprocessing_implication", "validation_implication", "limitation", "reporting_use"]
    write_csv(table_dir / "01b_chapter1_xray_evidence.csv", evidence_columns, evidence_rows)

    # Summary statistics and representation interpretation.
    artifact_fractions = [float(row["chromatic_pixel_fraction"]) for row in artifact_image_rows]
    overlap_objects = sum(int(row["chromatic_pixels_inside_bbox"]) > 0 for row in artifact_object_rows)
    mask_heavy = len(mask_quality_rows) - reliable_objects
    abs_contrast = [float(row["absolute_median_difference"]) for row in object_feature_rows]
    signed_contrast = [float(row["signed_median_difference"]) for row in object_feature_rows]
    robust_cnr = [float(row["cnr_like_robust"]) for row in object_feature_rows]
    object_iqr = [float(row["object_iqr"]) for row in object_feature_rows]
    bg_grad = [float(row["background_gradient_mean"]) for row in object_feature_rows]
    continuous_relationships = [row for row in relationship_rows if row["relationship_type"] == "continuous"]
    strongest = max(continuous_relationships, key=lambda row: abs(float(row["spearman_rho"])))
    redundancy_variables = {"object_std", "object_robust_range_p90_p10", "object_entropy", "background_std", "background_entropy", "background_gradient_std", "background_edge_density"}
    difficulty_relationships = [row for row in continuous_relationships if row["variable_y"] not in redundancy_variables]
    strongest_difficulty = max(difficulty_relationships, key=lambda row: abs(float(row["spearman_rho"])))
    rep_overall = {(row["representation"], row["metric"]): float(row["overall_value"]) for row in rep_rows if row["object_size_subgroup_or_quantile"] == "ALL" and row["machine"] == "ALL" and row["resolution"] == "ALL"}
    raw_sep = rep_overall[("RAW", "median_absolute_median_difference")]
    representation_lines = []
    for name in representations:
        sep = rep_overall[(name, "median_absolute_median_difference")]
        cnr = rep_overall[(name, "median_cnr_like_robust")]
        bgstd = rep_overall[(name, "median_background_std")]
        raw_bgstd = rep_overall[("RAW", "median_background_std")]
        representation_lines.append(f"- {name}: median absolute separation={sep:.4g}, robust CNR-like={cnr:.4g}, background-std ratio vs RAW={bgstd/max(EPSILON,raw_bgstd):.3f}.")
    joint_lines = [f"- {row['condition_a']} + {row['condition_b']}: {row['object_count']} objects ({float(row['object_fraction']):.1%}); later-analysis coverage={'adequate' if is_true(str(row['sufficient_for_later_analysis'])) else 'limited'}." for row in joint_rows]
    ring_lines = [f"- {scale}×: reliable={ring_aggregate[scale]['reliable_fraction']:.1%}, median valid pixels={ring_aggregate[scale]['median_valid']:.0f}, median |background median − 2×|={ring_aggregate[scale]['median_stability']:.3g}." for scale in RING_SCALES]
    summary_md = f"""# Stage 1B — X-ray Detection-Difficulty EDA Summary

## 1. Analysis scope

- **VERIFIED** — {len(allowlist)} approved canonical BMP/TXT samples and {len(object_feature_rows)} stable `(stem, object_id)` GT objects were analyzed. Input integrity is unchanged.
- No excluded, legacy-only, derived JPG, XML-as-GT, prediction, weight, external, or held-out data was used.

## 2. Colored-artifact findings

- **VERIFIED** — strong chromatic pixels (channel range≥{CHROMATIC_RANGE_THRESHOLD}) occur in {artifact_images}/{len(artifact_image_rows)} images; {len(artifact_image_rows)-artifact_images} images contain none.
- **VERIFIED** — used chromatic pixels comprise five palette colors, all with channel range 255; no color was assigned class/detection semantics.
- **VERIFIED** — median image fraction={quantile(artifact_fractions,.5):.4%}, maximum={max(artifact_fractions):.4%}; {overlap_objects}/{len(artifact_object_rows)} GT bboxes contain at least one masked pixel.
- **VERIFIED** — 1124/1147 objects lie within 3px of a strong chromatic component, and thin-rectangle component count equals TXT object count in 493/500 images. This is a critical shortcut/leakage risk even though no color semantics are assigned.
- **VERIFIED** — median chromatic fractions differ descriptively by acquisition group: machine 1/2/3 = 0.3286%/0.3660%/0.1564%; resolution 316×332/352×332/576×444 = 0.1373%/0.3286%/0.1564%. Machine and resolution are confounded.
- **STRONGLY SUPPORTED** — connected components frequently form thin outline-like structures, supporting a conservative strong-chromatic analysis mask. TXT remains the only GT.

## 3. Artifact-mask reliability

- **VERIFIED** — {reliable_objects}/{len(mask_quality_rows)} objects retain ≥30 bbox pixels with ≤35% masked; {mask_heavy} are flagged as artifact-contaminated for affected local statistics.
- Strategy: exclude strongly chromatic pixels only. Rectangle interiors and full GT boxes are never removed; source BMP files are unchanged.
- **EXPLORATORY** — the mask is suitable for EDA contamination control, not a selected production preprocessing method.
- A detector must not receive the embedded outlines as predictive input. The marked RAW BMP is an analysis reference, not yet an approved modeling representation.

## 4. Local object/background region definition

{chr(10).join(ring_lines)}

- **STRONGLY SUPPORTED** — selected {selected_ring}× as the primary ring using predeclared valid-pixel/stability criteria. It excludes the target bbox, other GT boxes, image-outside pixels, and artifact pixels.
- The GT bbox and ring are proxy regions, not pure foreign-object/product masks.

## 5. Intensity/contrast findings

- **VERIFIED** — absolute median contrast q25/q50/q75={quantile(abs_contrast,.25):.4g}/{quantile(abs_contrast,.5):.4g}/{quantile(abs_contrast,.75):.4g} luma units.
- **VERIFIED** — signed median contrast is negative for {sum(v<0 for v in signed_contrast)}/{len(signed_contrast)}, positive for {sum(v>0 for v in signed_contrast)}/{len(signed_contrast)}, and zero for {sum(v==0 for v in signed_contrast)} objects; both polarities are retained.
- **EXPLORATORY** — local contrast is a candidate visibility factor, not evidence of model difficulty.

## 6. CNR-like findings

- **VERIFIED** — robust image-derived CNR-like q25/q50/q75={quantile(robust_cnr,.25):.4g}/{quantile(robust_cnr,.5):.4g}/{quantile(robust_cnr,.75):.4g}; near-zero background-variance cases={near_zero_bg}.
- This is not a calibrated physical detector CNR and does not measure physical density.

## 7. Bbox-region heterogeneity

- **VERIFIED** — object IQR q25/q50/q75={quantile(object_iqr,.25):.4g}/{quantile(object_iqr,.5):.4g}/{quantile(object_iqr,.75):.4g}.
- **STRONGLY SUPPORTED** — object IQR is retained as the compact robust proxy; entropy/gradient/CV remain diagnostic because they overlap conceptually or are less stable.

## 8. Background complexity

- **VERIFIED** — background gradient mean q25/q50/q75={quantile(bg_grad,.25):.4g}/{quantile(bg_grad,.5):.4g}/{quantile(bg_grad,.75):.4g}.
- **STRONGLY SUPPORTED** — background gradient mean plus background IQR retain spatial-gradient and intensity-spread concepts without claiming a physical food property.

## 9. Product-boundary feasibility

- **EXPLORATORY** — two simple methods on {len(audit_stems)} diverse images yield heuristic-plausible rows={boundary_plausible_fraction:.1%}, median pair IoU={median_boundary_iou:.3f}.
- **STRONGLY SUPPORTED** — 12 representative overlays were reviewed and the selected Otsu-dark largest-component rule passed {full_boundary_success}/{len(allowlist)} approved images ({full_boundary_success_fraction:.1%}). Product-edge distance was therefore created as an analysis-only candidate.
- **UNRESOLVED** — no physical product-boundary GT exists, so the inferred distance is not universally validated and must remain separate from image-edge distance.

## 10. Relationships among candidate difficulty variables

- **EXPLORATORY** — strongest non-redundancy difficulty association: {strongest_difficulty['relationship']}, rho={float(strongest_difficulty['spearman_rho']):.3f}.
- **VERIFIED** — redundancy audit peaks at {strongest['relationship']}, rho={float(strongest['spearman_rho']):.3f}; this supports retaining compact representative features rather than near-duplicate metrics.
- Machine/resolution summaries are descriptive and confounded by acquisition and object distributions. Correlation does not imply causation or model failure.

## 11. Joint-condition coverage

{chr(10).join(joint_lines)}

- Quartiles are exploratory coverage bins, not permanent hard/easy thresholds.

## 12. Representation-comparison findings

{chr(10).join(representation_lines)}

- **EXPLORATORY** — separation gains accompanied by background amplification are not automatically improvements. CLAHE is an 8×8 clipped tile implementation without interpolation and is reported with that limitation.
- **EXPLORATORY** — percentile windowing is unfavorable under these proxies (no median-separation gain, lower robust CNR-like value, 1.648× background std). CLAHE-like processing is inconclusive because larger absolute separation accompanies 1.985× background std and lower robust CNR-like separation.
- **EXPLORATORY** — residual and gradient views reduce the selected standalone separation/CNR-like proxies; they are not supported as replacements for RAW, though later models could test them only as separately justified auxiliary channels.
- Representation montages use the same sample in left-to-right order: RAW, window, CLAHE-like, absolute local residual, gradient magnitude; magenta marks analysis-excluded chromatic pixels.
- No production representation was selected.

## 13. Compact candidate feature set for Stage 2

- Size: bbox minimum side/area ratio.
- Position: normalized center and image-edge distance.
- Shape proxy: bbox aspect ratio.
- Visibility: signed/absolute median contrast and robust CNR-like proxy.
- Heterogeneity: bbox-region IQR.
- Background complexity: ring gradient mean and IQR.
- Product-edge distance: included with a sensitivity flag because the simple rule was consistent across the approved scope, but no physical boundary GT exists.

## 14. Mapping to competition factors

- **VERIFIED** — size/position/bbox-shape are annotation-derived proxies.
- **VERIFIED** — physical density is unavailable; only image-derived intensity/contrast/CNR-like proxies are mapped.
- **STRONGLY SUPPORTED** — background contrast is represented by artifact-excluded bbox-versus-ring statistics with explicit proxy limitations.

## 15. Preprocessing hypotheses generated by EDA

- Windowing, clipped local equalization, residual, and gradient representations remain testable hypotheses only where separation gains are not offset by background amplification.
- Artifact exclusion is an EDA measurement safeguard, not a production preprocessing choice.
- Because artifact components nearly reproduce GT positions/counts, artifact-control ablation is required before any fair baseline; training directly on marked RAW pixels is not acceptable.
- Any later comparison must use identical leakage-aware folds and raw input as the reference.

## 16. Remaining unresolved issues

- Residual product-boundary accuracy uncertainty without segmentation GT, despite strong within-dataset consistency.
- Physical meaning of intensity and acquisition settings; density/material/thickness are unavailable.
- Whether strong chromatic marks are present at deployment and how a production pipeline should handle them.
- Whether conservative exclusion/inpainting can remove outline shortcuts without damaging underlying X-ray signal.
- Stability of representation rankings under leakage-aware validation and model outcomes.
- Local rings may include mixed product structures and GT bbox regions include background.

## 17. Recommended next step

- First finalize leakage-aware candidate folds using the Stage 1A.5 6s/10s results and reviewed similarity components. Then run a separately authorized artifact-control ablation to define a fair modeling input before any baseline. Stage 2 should test whether errors actually concentrate under these candidate factors.
"""
    (eda_dir / "01b_xray_difficulty_eda_summary.md").write_text(summary_md, encoding="utf-8-sig")
    print(f"samples={len(allowlist)} objects={len(object_feature_rows)} artifacts={artifact_images} ring={selected_ring} boundary_plausible={boundary_plausible_fraction:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1B pre-model X-ray detection-difficulty EDA")
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    main(args.workspace.resolve())
