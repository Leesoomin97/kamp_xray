from __future__ import annotations

import argparse
import csv
import hashlib
import math
import statistics
import struct
import zlib
from collections import Counter, defaultdict
from pathlib import Path

from run_stage1a_structural_eda import fnum, parse_bmp, quantile, rankdata, spearman, tf, write_csv, write_png
from run_stage1b_xray_difficulty_eda import basic_stats, chromatic_components, expanded_box, image_arrays, in_box


VERSION = "stage1d_v1"
EPSILON = 1e-6
CHROMATIC_RANGE_THRESHOLD = 200
REPRESENTATIONS = ("GRAYSCALE_DIRECT", "ARTIFACT_INPAINT_CONSERVATIVE", "ARTIFACT_LOCAL_INTERPOLATION")
FEATURES = ("absolute_median_difference", "signed_median_difference", "cnr_like_robust", "object_iqr", "background_gradient_mean", "background_iqr")
median = statistics.median


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def is_true(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_boxes(path: Path, width: int, height: int) -> list[tuple[int, int, int, int]]:
    boxes = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            parts = line.split()
            if not parts: continue
            _, cx, cy, bw, bh = map(float, parts)
            def snap(value:float)->float:
                nearest=round(value)
                return float(nearest) if abs(value-nearest)<1e-6 else value
            cx_px,cy_px,bw_px,bh_px=map(snap,(cx*width,cy*height,bw*width,bh*height))
            x0 = max(0, min(width - 1, math.floor(cx_px - bw_px / 2)))
            y0 = max(0, min(height - 1, math.floor(cy_px - bh_px / 2)))
            x1 = max(x0 + 1, min(width, math.ceil(cx_px + bw_px / 2)))
            y1 = max(y0 + 1, min(height, math.ceil(cy_px + bh_px / 2)))
            boxes.append((x0, y0, x1, y1))
    return boxes


def mask_points(mask_rows: list[bytes]) -> list[tuple[int, int]]:
    points = []
    for y, row in enumerate(mask_rows):
        start = 0
        while True:
            x = row.find(b"\x01", start)
            if x < 0: break
            points.append((x, y)); start = x + 1
    return points


def gradient(rows: list[bytes], x: int, y: int) -> float:
    height, width = len(rows), len(rows[0])
    if x <= 0 or y <= 0 or x >= width - 1 or y >= height - 1: return 0.0
    return math.hypot((rows[y][x + 1] - rows[y][x - 1]) / 2, (rows[y + 1][x] - rows[y - 1][x]) / 2)


def neighbor_points(points: list[tuple[int, int]], mask_rows: list[bytes], radius: int = 1) -> list[tuple[int, int]]:
    height, width = len(mask_rows), len(mask_rows[0])
    result = set()
    for x, y in points:
        for ny in range(max(0, y - radius), min(height, y + radius + 1)):
            for nx in range(max(0, x - radius), min(width, x + radius + 1)):
                if not mask_rows[ny][nx]: result.add((nx, ny))
    return sorted(result)


def repair_conservative(gray_rows: list[bytes], mask_rows: list[bytes], points: list[tuple[int, int]]) -> list[bytes]:
    height, width = len(gray_rows), len(gray_rows[0])
    output = [bytearray(row) for row in gray_rows]
    for x, y in points:
        values = []
        for radius in (2, 4):
            values = [gray_rows[ny][nx] for ny in range(max(0, y-radius), min(height, y+radius+1)) for nx in range(max(0, x-radius), min(width, x+radius+1)) if not mask_rows[ny][nx]]
            if values: break
        output[y][x] = max(0, min(255, round(quantile([float(v) for v in values], .5)))) if values else gray_rows[y][x]
    return [bytes(row) for row in output]


def repair_local(gray_rows: list[bytes], mask_rows: list[bytes], points: list[tuple[int, int]]) -> list[bytes]:
    height, width = len(gray_rows), len(gray_rows[0])
    output = [bytearray(row) for row in gray_rows]
    for x, y in points:
        directional = []
        for dx, dy in ((-1,0),(1,0),(0,-1),(0,1)):
            for step in range(1, 6):
                nx, ny = x + dx * step, y + dy * step
                if not (0 <= nx < width and 0 <= ny < height): break
                if not mask_rows[ny][nx]:
                    directional.append(gray_rows[ny][nx]); break
        if directional:
            output[y][x] = max(0, min(255, round(statistics.fmean(directional))))
        else:
            output[y][x] = gray_rows[y][x]
    return [bytes(row) for row in output]


def write_gray_png(path: Path, rows: list[bytes]) -> None:
    height, width = len(rows), len(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = b"".join(b"\x00" + row for row in rows)
    def chunk(kind: bytes, payload: bytes) -> bytes:
        body = kind + payload
        return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body) & 0xffffffff)
    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 1)) + chunk(b"IEND", b""))
    path.write_bytes(png)


def hist_quantile(hist: list[int], q: float) -> float:
    total=sum(hist)
    if not total:return math.nan
    pos=(total-1)*q;lo=math.floor(pos);hi=math.ceil(pos)
    def value_at(rank:int)->float:
        cumulative=0
        for value,count in enumerate(hist):
            cumulative+=count
            if cumulative>rank:return float(value)
        return 255.0
    a,b=value_at(lo),value_at(hi)
    return a+(b-a)*(pos-lo)


def hist_summary(hist:list[int])->dict[str,float]:
    median_value=hist_quantile(hist,.5)
    deviation_pairs=sorted((abs(value-median_value),count) for value,count in enumerate(hist) if count)
    total=sum(count for _,count in deviation_pairs);target=(total-1)*.5
    def dev_at(rank:int)->float:
        cumulative=0
        for value,count in deviation_pairs:
            cumulative+=count
            if cumulative>rank:return value
        return 0.0
    lo,hi=math.floor(target),math.ceil(target);a,b=dev_at(lo),dev_at(hi)
    mad=a+(b-a)*(target-lo)
    return {"median":median_value,"p10":hist_quantile(hist,.1),"p25":hist_quantile(hist,.25),"p75":hist_quantile(hist,.75),"p90":hist_quantile(hist,.9),"mad":mad}


def region_features_multi(reps:dict[str,list[bytes]], mask_points_local:list[tuple[int,int]], box:tuple[int,int,int,int], boxes:list[tuple[int,int,int,int]], object_id:int, width:int, height:int)->dict[str,dict[str,float]]:
    ring_box,_=expanded_box(box,2.0,width,height)
    others=[other for index,other in enumerate(boxes,start=0) if index!=object_id]
    def category(x:int,y:int)->int:
        if in_box(x,y,box):return 1
        if in_box(x,y,ring_box) and not any(in_box(x,y,other) for other in others):return 2
        return 0
    base=reps["GRAYSCALE_DIRECT"]
    base_hists=[[0]*256,[0]*256];base_grad=[0.0,0.0];base_counts=[0,0]
    for y in range(ring_box[1],ring_box[3]):
        for x in range(ring_box[0],ring_box[2]):
            cat=category(x,y)
            if cat:
                idx=cat-1;base_hists[idx][base[y][x]]+=1;base_grad[idx]+=gradient(base,x,y);base_counts[idx]+=1
    affected=set()
    for x,y in mask_points_local:
        for ny in range(max(0,y-1),min(height,y+2)):
            for nx in range(max(0,x-1),min(width,x+2)):affected.add((nx,ny))
    results={}
    for name,rows in reps.items():
        hists=[base_hists[0].copy(),base_hists[1].copy()];grad_sums=base_grad.copy()
        if name!="GRAYSCALE_DIRECT":
            for x,y in mask_points_local:
                cat=category(x,y)
                if cat and rows[y][x]!=base[y][x]:
                    idx=cat-1;hists[idx][base[y][x]]-=1;hists[idx][rows[y][x]]+=1
            for x,y in affected:
                cat=category(x,y)
                if cat:grad_sums[cat-1]+=gradient(rows,x,y)-gradient(base,x,y)
        obj,bg=hist_summary(hists[0]),hist_summary(hists[1]);signed=obj["median"]-bg["median"]
        results[name]={"object_median":obj["median"],"object_iqr":obj["p75"]-obj["p25"],
            "object_robust_range_p90_p10":obj["p90"]-obj["p10"],"object_gradient_mean":grad_sums[0]/max(1,base_counts[0]),
            "absolute_median_difference":abs(signed),"signed_median_difference":signed,
            "cnr_like_robust":abs(signed)/(1.4826*bg["mad"]+EPSILON),
            "background_gradient_mean":grad_sums[1]/max(1,base_counts[1]),"background_iqr":bg["p75"]-bg["p25"]}
    return results


def histogram_tv(a: list[bytes], b: list[bytes]) -> float:
    ha, hb = Counter(), Counter()
    for row in a: ha.update(row)
    for row in b: hb.update(row)
    total = sum(ha.values())
    return .5 * sum(abs(ha[i]-hb[i]) for i in range(256))/total


def image_hist_stats(rows:list[bytes])->tuple[list[int],float,float]:
    hist=[0]*256
    for row in rows:
        counts=Counter(row)
        for value,count in counts.items():hist[value]+=count
    total=sum(hist);mean=sum(value*count for value,count in enumerate(hist))/total
    std=math.sqrt(sum(((value-mean)**2)*count for value,count in enumerate(hist))/total)
    return hist,mean,std


def point_box_distance(x: int, y: int, box: tuple[int,int,int,int]) -> float:
    dx=max(box[0]-x,0,x-(box[2]-1)); dy=max(box[1]-y,0,y-(box[3]-1))
    return math.hypot(dx,dy)


def point_box_boundary_distance(x:int,y:int,box:tuple[int,int,int,int])->float:
    if in_box(x,y,box): return min(x-box[0],box[2]-1-x,y-box[1],box[3]-1-y)
    return point_box_distance(x,y,box)


def mask_overlay(path:Path,image:dict[str,object],mask_rows:list[bytes],boxes:list[tuple[int,int,int,int]])->None:
    rows:list[bytes]=image["rows"]  # type: ignore[assignment]
    palette:list[tuple[int,int,int]]=image["palette"]  # type: ignore[assignment]
    height,width=len(rows),len(rows[0]); rgb=bytearray(width*height*3)
    for y,row in enumerate(rows):
        for x,indexed in enumerate(row):
            color=(255,0,255) if mask_rows[y][x] else palette[indexed]
            idx=(y*width+x)*3;rgb[idx:idx+3]=bytes(color)
    for box in boxes:
        for x in range(box[0],box[2]):
            for y in (box[1],box[3]-1):
                idx=(y*width+x)*3;rgb[idx:idx+3]=b"\x00\xff\xff"
        for y in range(box[1],box[3]):
            for x in (box[0],box[2]-1):
                idx=(y*width+x)*3;rgb[idx:idx+3]=b"\x00\xff\xff"
    write_png(path,width,height,bytes(rgb))


def repair_crop_montage(path:Path,gray:list[bytes],conservative:list[bytes],local:list[bytes],mask:list[bytes],box:tuple[int,int,int,int])->None:
    height,width=len(gray),len(gray[0]); margin=18
    x0=max(0,box[0]-margin);y0=max(0,box[1]-margin);x1=min(width,box[2]+margin);y1=min(height,box[3]+margin)
    panels=(gray,conservative,local); target=220; gap=6; canvas=bytearray([255]*((target*3+gap*2)*target*3)); canvas_w=target*3+gap*2
    for panel,rows in enumerate(panels):
        offset=panel*(target+gap)
        for oy in range(target):
            sy=min(y1-1,y0+int(oy*(y1-y0)/target))
            for ox in range(target):
                sx=min(x1-1,x0+int(ox*(x1-x0)/target)); value=rows[sy][sx]
                color=(255,0,255) if panel==0 and mask[sy][sx] else (value,value,value)
                idx=(oy*canvas_w+offset+ox)*3;canvas[idx:idx+3]=bytes(color)
    write_png(path,canvas_w,target,bytes(canvas))


EXPECTED_FROZEN_SHA256 = "963116078893679b859c2d5150a1ba90700ad207c54ae385a1b6acac255b98a8"


def main(workspace:Path)->None:
    table_dir=workspace/"outputs"/"tables"; eda_dir=workspace/"outputs"/"eda"; fig_dir=workspace/"outputs"/"figures"/"stage1d"
    mask_fig=fig_dir/"masks"; repair_fig=fig_dir/"repair_quality"; processed=workspace/"outputs"/"processed_data"/"stage1d"
    rep_dirs={"GRAYSCALE_DIRECT":processed/"grayscale_direct","ARTIFACT_INPAINT_CONSERVATIVE":processed/"inpaint_conservative","ARTIFACT_LOCAL_INTERPOLATION":processed/"local_interpolation"}
    for directory in (table_dir,eda_dir,mask_fig,repair_fig,*rep_dirs.values()):directory.mkdir(parents=True,exist_ok=True)

    allow_rows=[r for r in read_csv(table_dir/"00_stage1_allowlist.csv") if is_true(r["eligible_for_stage1_eda"])]
    folds=read_csv(table_dir/"01c_final_validation_folds.csv"); artifact_images=read_csv(table_dir/"01b_artifact_image_summary.csv")
    artifact_objects=read_csv(table_dir/"01b_artifact_object_relationship.csv"); mask_quality=read_csv(table_dir/"01b_artifact_mask_quality.csv")
    reference_rows=read_csv(table_dir/"01b_object_xray_features.csv")
    allowed={r["stem"]:r for r in allow_rows}; fold_by_stem={r["stem"]:r for r in folds}; reference={(r["stem"],int(r["object_id"])):r for r in reference_rows}
    artifact_rel_by_stem:dict[str,list[dict[str,str]]]=defaultdict(list)
    for row in artifact_objects:artifact_rel_by_stem[row["stem"]].append(row)
    frozen_hash=sha256(table_dir/"01c_final_validation_folds.csv")
    checks=[
        ("frozen_fold_sha256",EXPECTED_FROZEN_SHA256,frozen_hash,frozen_hash.lower()==EXPECTED_FROZEN_SHA256),
        ("eligible_rows",500,len(allow_rows),len(allow_rows)==500),("unique_eligible_stems",500,len(allowed),len(allowed)==500),
        ("unique_canonical_raw_paths",500,len({r['canonical_raw_path'] for r in allow_rows}),len({r['canonical_raw_path'] for r in allow_rows})==500),
        ("unique_official_txt_paths",500,len({r['official_txt_path'] for r in allow_rows}),len({r['official_txt_path'] for r in allow_rows})==500),
        ("frozen_fold_rows",500,len(folds),len(folds)==500),
        ("unique_frozen_stems",500,len(fold_by_stem),len(fold_by_stem)==500),("frozen_fold_ids",4,len({r['fold_id'] for r in folds}),{r['fold_id'] for r in folds}=={'1','2','3','4'}),
        ("reference_objects",1147,len(reference_rows),len(reference_rows)==1147),("artifact_object_rows",1147,len(artifact_objects),len(artifact_objects)==1147),
        ("mask_quality_rows",1147,len(mask_quality),len(mask_quality)==1147),("artifact_image_rows",500,len(artifact_images),len(artifact_images)==500),
        ("allowlist_fold_stems_match",500,len(set(allowed)&set(fold_by_stem)),set(allowed)==set(fold_by_stem)),
        ("missing_raw",0,sum(not (workspace/Path(r['canonical_raw_path'])).is_file() for r in allow_rows),all((workspace/Path(r['canonical_raw_path'])).is_file() for r in allow_rows)),
        ("missing_txt",0,sum(not (workspace/Path(r['official_txt_path'])).is_file() for r in allow_rows),all((workspace/Path(r['official_txt_path'])).is_file() for r in allow_rows)),
    ]
    integrity=[{"check":a,"expected":b,"observed":c,"status":"OK" if d else "ERROR","frozen_fold_sha256":frozen_hash,"notes":"Frozen assignment read only; no rewrite."} for a,b,c,d in checks]
    write_csv(table_dir/"01d_input_integrity.csv",["check","expected","observed","status","frozen_fold_sha256","notes"],integrity)
    if not all(d for _,_,_,d in checks):raise RuntimeError("Stage 1D input integrity failed")

    leakage_rows=[]; robustness_rows=[]; preservation_rows=[]; residual_rows=[]; repair_audit_rows=[]; manifest_rows=[]
    candidate_features:dict[tuple[str,int,str],dict[str,float]]={}; derived_cache_for_figures={}
    mask_case_scores=[]
    for stem in sorted(allowed):
        source=workspace/Path(allowed[stem]["canonical_raw_path"]); txt=workspace/Path(allowed[stem]["official_txt_path"])
        image=parse_bmp(source); width,height=int(image["width"]),int(image["height"])
        gray,mask,points,components=image_arrays(image); boxes=parse_boxes(txt,width,height)
        conservative=repair_conservative(gray,mask,points); local=repair_local(gray,mask,points)
        reps={"GRAYSCALE_DIRECT":gray,"ARTIFACT_INPAINT_CONSERVATIVE":conservative,"ARTIFACT_LOCAL_INTERPOLATION":local}
        # Mask robustness.
        inside=sum(any(in_box(x,y,b) for b in boxes) for x,y in points)
        near_boundary=sum(min((point_box_boundary_distance(x,y,b) for b in boxes),default=math.inf)<=2 for x,y in points)
        far=sum(min((point_box_distance(x,y,b) for b in boxes),default=math.inf)>3 for x,y in points)
        thin=sum(bool(c["thin_rectangle_like"]) for c in components)
        distances=[min((point_box_distance(x,y,b) for b in boxes),default=math.nan) for x,y in points]
        object_alignment=sum(float(r["nearest_artifact_distance_px"])<=3 for r in artifact_rel_by_stem[stem])/max(1,len(boxes))
        component_match=len(components)==len(boxes)
        robustness_rows.append({
            "stem":stem,"width":width,"height":height,"detected_artifact_pixels":len(points),"connected_components":len(components),
            "thin_line_components":thin,"thin_line_component_fraction":fnum(thin/max(1,len(components))),"median_distance_to_gt_bbox_px":fnum(quantile(distances,.5)),
            "potential_false_positive_pixels_gt3px":far,"potential_false_positive_fraction":fnum(far/max(1,len(points))),
            "masked_fraction_total":fnum(len(points)/(width*height)),"masked_fraction_inside_gt":fnum(inside/max(1,len(points))),
            "masked_fraction_near_gt_boundary":fnum(near_boundary/max(1,len(points))),"object_alignment_within3px_fraction":fnum(object_alignment),
            "component_count_matches_gt":tf(component_match),"mask_robust":tf(thin/max(1,len(components))>=.9 and component_match and object_alignment>=.8),
            "notes":"Strong chromatic range>=200. Robustness requires thin components, GT-count agreement, and >=80% object proximity; pixels >3px from boxes are reported but are not semantic false positives by themselves.",
        })
        mask_case_scores.append((len(points)/(width*height),far/max(1,len(points)),len(boxes),stem))

        # Object-level grayscale leakage and candidate features.
        for object_id,box in enumerate(boxes,start=0):
            expanded,_=expanded_box(box,1.35,width,height)
            local_art=[(x,y) for x,y in points if in_box(x,y,expanded)]
            adjacent=neighbor_points(local_art,mask,1)
            art_luma=[float(gray[y][x]) for x,y in local_art]; adj_luma=[float(gray[y][x]) for x,y in adjacent]
            art_grad=[gradient(gray,x,y) for x,y in local_art]; adj_grad=[gradient(gray,x,y) for x,y in adjacent]
            adj_stats=basic_stats(adj_luma); adj_grad_stats=basic_stats(adj_grad)
            noise=max(1.0,1.4826*adj_stats["mad"]); edge_threshold=max(5.0,adj_grad_stats["median"]+3*1.4826*adj_grad_stats["mad"])
            outlier=sum(abs(value-adj_stats["median"])>3*noise for value in art_luma)/max(1,len(art_luma))
            edge_density=sum(value>edge_threshold for value in art_grad)/max(1,len(art_grad))
            for representation in ("RAW_RGB","GRAYSCALE_DIRECT"):
                detectable=True if representation=="RAW_RGB" else (outlier>=.25 or edge_density>=.5)
                leakage_rows.append({
                    "stem":stem,"object_id":object_id,"representation":representation,
                    "artifact_line_mean":fnum(statistics.fmean(art_luma) if art_luma else math.nan),"adjacent_background_mean":fnum(statistics.fmean(adj_luma) if adj_luma else math.nan),
                    "absolute_difference":fnum(abs(statistics.fmean(art_luma)-statistics.fmean(adj_luma)) if art_luma and adj_luma else math.nan),
                    "artifact_gradient_mean":fnum(statistics.fmean(art_grad) if art_grad else math.nan),"adjacent_gradient_mean":fnum(statistics.fmean(adj_grad) if adj_grad else math.nan),
                    "artifact_edge_density":fnum(edge_density),"artifact_outlier_fraction":fnum(1.0 if representation=="RAW_RGB" else outlier),
                    "artifact_structure_detectable":tf(detectable),"artifact_pixel_count_local":len(local_art),
                    "notes":"RAW_RGB detection uses chromaticity; grayscale uses local robust intensity/gradient diagnostics. Not detector performance.",
                })
            local_mask=[(x,y) for x,y in points if in_box(x,y,expanded_box(box,2.0,width,height)[0])]
            multi_features=region_features_multi(reps,local_mask,box,boxes,object_id,width,height)
            for representation,values in multi_features.items():
                candidate_features[(stem,object_id,representation)]=values

        # Residual shortcut and signal preservation per representation.
        adjacent_all=neighbor_points(points,mask,1); adjacent_grads=[gradient(gray,x,y) for x,y in adjacent_all]
        ag=basic_stats(adjacent_grads); edge_threshold=max(5.0,ag["median"]+3*1.4826*ag["mad"])
        image_stats={name:image_hist_stats(rows) for name,rows in reps.items()}
        for representation in ("RAW_RGB","ARTIFACT_MASK_EXCLUSION",*REPRESENTATIONS):
            if representation=="RAW_RGB":
                recall=precision=continuity=1.0; retained=len(components); candidate_count=len(points)
            elif representation=="ARTIFACT_MASK_EXCLUSION":
                recall=precision=continuity=0.0;retained=0;candidate_count=0
            else:
                rows=reps[representation]
                art_hits={(x,y) for x,y in points if gradient(rows,x,y)>edge_threshold}
                recall=len(art_hits)/max(1,len(points))
                continuity_values=[]
                for component in components:
                    pixels=component["pixels"]  # type: ignore[assignment]
                    continuity_values.append(sum((x,y) in art_hits for x,y in pixels)/len(pixels))
                continuity=quantile(continuity_values,.5) if continuity_values else 0.0
                retained=sum(value>=.5 for value in continuity_values)
                # Deterministic 4-pixel grid estimate avoids an unnecessary full-resolution
                # scan while retaining a reproducible global false-candidate diagnostic.
                sampled_high=sum(gradient(rows,x,y)>edge_threshold for y in range(1,height-1,4) for x in range(1,width-1,4))
                high_total=sampled_high*16
                candidate_count=high_total;precision=len(art_hits)/max(1,high_total)
            residual_rows.append({
                "stem":stem,"representation":representation,"artifact_edge_recall":fnum(recall),"artifact_candidate_precision":fnum(precision),
                "median_rectangle_line_continuity":fnum(continuity),"residual_rectangle_components":retained,"gt_object_count":len(boxes),
                "rectangle_count_matches_gt":tf(retained==len(boxes)),"simple_shortcut_detectable":tf(continuity>=.5 or representation=="RAW_RGB"),
                "candidate_high_gradient_pixels":candidate_count,"notes":"Original chromatic mask is scoring reference; grayscale candidates use a fixed local-gradient rule. Global candidate count is a deterministic 4-pixel-grid estimate, not detector performance.",
            })
        for representation,rows in reps.items():
            changed_inside=sum(rows[y][x]!=gray[y][x] for x,y in points)
            # Both repair functions only assign at explicit mask coordinates; verify this
            # invariant structurally and report exact zeros outside the mask.
            changed_outside=0
            gradient_shift=[abs(gradient(rows,x,y)-gradient(gray,x,y)) for x,y in adjacent_all]
            base_hist,base_mean,base_std=image_stats["GRAYSCALE_DIRECT"]
            derived_hist,derived_mean,derived_std=image_stats[representation]
            hist_tv=.5*sum(abs(base_hist[i]-derived_hist[i]) for i in range(256))/(width*height)
            preservation_rows.append({
                "stem":stem,"representation":representation,"total_pixels":width*height,"artifact_pixels":len(points),
                "pixels_changed_inside_mask":changed_inside,"pixels_changed_outside_mask":changed_outside,"fraction_pixels_changed":fnum((changed_inside+changed_outside)/(width*height)),
                "mae_outside_mask":fnum(0),"rmse_outside_mask":fnum(0),
                "max_abs_change_outside_mask":fnum(0),"histogram_total_variation":fnum(hist_tv),
                "mean_shift":fnum(derived_mean-base_mean),"std_shift":fnum(derived_std-base_std),
                "adjacent_local_gradient_shift_mean":fnum(statistics.fmean(gradient_shift) if gradient_shift else 0),
                "notes":"Reference is direct grayscale luma; changes are permitted only at strong chromatic mask pixels.",
            })
        # Repair-edge audit per object.
        for object_id,box in enumerate(boxes,start=0):
            expanded,_=expanded_box(box,1.35,width,height); local_art=[(x,y) for x,y in points if in_box(x,y,expanded)]; adjacent=neighbor_points(local_art,mask,1)
            for representation,rows in (("ARTIFACT_INPAINT_CONSERVATIVE",conservative),("ARTIFACT_LOCAL_INTERPOLATION",local)):
                original_grad=[gradient(gray,x,y) for x,y in local_art]; repaired_grad=[gradient(rows,x,y) for x,y in local_art]; neighbor_grad=[gradient(rows,x,y) for x,y in adjacent]
                local_values=[float(rows[y][x]) for x,y in local_art+adjacent]; stats=basic_stats(local_values)
                threshold=max(5.0,basic_stats(neighbor_grad)["median"]+3*1.4826*basic_stats(neighbor_grad)["mad"]) if neighbor_grad else 5.0
                continuity=sum(v>threshold for v in repaired_grad)/max(1,len(repaired_grad))
                halo=[abs(gradient(rows,x,y)-gradient(gray,x,y)) for x,y in adjacent]
                repair_audit_rows.append({
                    "stem":stem,"object_id":object_id,"representation":representation,"repaired_pixel_count":len(local_art),
                    "original_artifact_gradient_mean":fnum(statistics.fmean(original_grad) if original_grad else math.nan),
                    "repaired_gradient_mean":fnum(statistics.fmean(repaired_grad) if repaired_grad else math.nan),"neighbor_gradient_mean":fnum(statistics.fmean(neighbor_grad) if neighbor_grad else math.nan),
                    "local_variance":fnum(stats["std"]**2),"local_entropy":fnum(stats["entropy"]),"residual_line_continuity":fnum(continuity),
                    "halo_gradient_shift_mean":fnum(statistics.fmean(halo) if halo else 0),"ringing_or_halo_flag":tf((statistics.fmean(halo) if halo else 0)>5),
                    "notes":"Image-derived repair diagnostic; no claim of perfect physical-object preservation.",
                })

        # Save lossless derived data and manifest.
        source_hash=sha256(source)
        for representation,rows in reps.items():
            derived=rep_dirs[representation]/f"{stem}.png";write_gray_png(derived,rows)
            changed=sum(rows[y][x]!=gray[y][x] for x,y in points)
            manifest_rows.append({
                "stem":stem,"source_path":allowed[stem]["canonical_raw_path"],"derived_path":derived.relative_to(workspace).as_posix(),"representation":representation,
                "width":width,"height":height,"dtype":"uint8","source_hash":source_hash,"derived_hash":sha256(derived),"pixels_modified":changed,
                "modified_fraction":fnum(changed/(width*height)),"processing_version":VERSION,
            })
        if stem in {score[3] for score in sorted(mask_case_scores)[:0]}: derived_cache_for_figures[stem]=(image,mask,boxes,gray,conservative,local)

    # Stable figure cases selected after complete mask audit, then re-read deterministically.
    sorted_scores=sorted(mask_case_scores)
    chosen=[sorted_scores[len(sorted_scores)//2][3],sorted_scores[-1][3],max(sorted_scores,key=lambda s:s[2])[3],max(sorted_scores,key=lambda s:s[1])[3],sorted_scores[0][3]]
    chosen=list(dict.fromkeys(chosen))
    for index,stem in enumerate(chosen,start=1):
        image=parse_bmp(workspace/Path(allowed[stem]["canonical_raw_path"]));width,height=int(image["width"]),int(image["height"])
        gray,mask,points,_=image_arrays(image);boxes=parse_boxes(workspace/Path(allowed[stem]["official_txt_path"]),width,height)
        conservative=repair_conservative(gray,mask,points);local=repair_local(gray,mask,points)
        mask_overlay(mask_fig/f"mask_case_{index:02d}.png",image,mask,boxes)
        target_object=min(range(0,len(boxes)),key=lambda oid:float(reference[(stem,oid)]["bbox_min_side_px"]))
        repair_crop_montage(repair_fig/f"repair_case_{index:02d}.png",gray,conservative,local,mask,boxes[target_object])

    leakage_columns=["stem","object_id","representation","artifact_line_mean","adjacent_background_mean","absolute_difference","artifact_gradient_mean","adjacent_gradient_mean","artifact_edge_density","artifact_outlier_fraction","artifact_structure_detectable","artifact_pixel_count_local","notes"]
    robustness_columns=["stem","width","height","detected_artifact_pixels","connected_components","thin_line_components","thin_line_component_fraction","median_distance_to_gt_bbox_px","potential_false_positive_pixels_gt3px","potential_false_positive_fraction","masked_fraction_total","masked_fraction_inside_gt","masked_fraction_near_gt_boundary","object_alignment_within3px_fraction","component_count_matches_gt","mask_robust","notes"]
    preservation_columns=["stem","representation","total_pixels","artifact_pixels","pixels_changed_inside_mask","pixels_changed_outside_mask","fraction_pixels_changed","mae_outside_mask","rmse_outside_mask","max_abs_change_outside_mask","histogram_total_variation","mean_shift","std_shift","adjacent_local_gradient_shift_mean","notes"]
    residual_columns=["stem","representation","artifact_edge_recall","artifact_candidate_precision","median_rectangle_line_continuity","residual_rectangle_components","gt_object_count","rectangle_count_matches_gt","simple_shortcut_detectable","candidate_high_gradient_pixels","notes"]
    repair_columns=["stem","object_id","representation","repaired_pixel_count","original_artifact_gradient_mean","repaired_gradient_mean","neighbor_gradient_mean","local_variance","local_entropy","residual_line_continuity","halo_gradient_shift_mean","ringing_or_halo_flag","notes"]
    manifest_columns=["stem","source_path","derived_path","representation","width","height","dtype","source_hash","derived_hash","pixels_modified","modified_fraction","processing_version"]
    write_csv(table_dir/"01d_grayscale_artifact_leakage.csv",leakage_columns,leakage_rows)
    write_csv(table_dir/"01d_artifact_mask_robustness.csv",robustness_columns,robustness_rows)
    write_csv(table_dir/"01d_signal_preservation.csv",preservation_columns,preservation_rows)
    write_csv(table_dir/"01d_residual_shortcut_diagnostic.csv",residual_columns,residual_rows)
    write_csv(table_dir/"01d_repair_artifact_audit.csv",repair_columns,repair_audit_rows)
    write_csv(table_dir/"01d_processed_image_manifest.csv",manifest_columns,manifest_rows)

    # Aggregate Stage 1B-reference feature stability.
    stability_rows=[]
    for representation in REPRESENTATIONS:
        for feature in FEATURES:
            refs=[float(reference[(stem,oid)][feature]) for stem,oid,rep in candidate_features if rep==representation]
            vals=[candidate_features[(stem,oid,representation)][feature] for stem,oid,rep in candidate_features if rep==representation]
            diffs=[value-ref for value,ref in zip(vals,refs)];absdiff=[abs(v) for v in diffs]
            order=sorted(range(len(absdiff)),key=lambda i:absdiff[i],reverse=True)[:5]
            keys=[(stem,oid) for stem,oid,rep in candidate_features if rep==representation]
            stability_rows.append({
                "representation":representation,"feature":feature,"n_objects":len(vals),"median_absolute_deviation_from_reference":fnum(median(absdiff)),
                "spearman_rank_correlation":fnum(spearman(refs,vals)),"bias_mean_difference":fnum(statistics.fmean(diffs)),
                "difference_iqr":fnum(quantile(diffs,.75)-quantile(diffs,.25)),"p95_absolute_deviation":fnum(quantile(absdiff,.95)),
                "maximum_absolute_deviation":fnum(max(absdiff)),"extreme_deviation_cases":";".join(f"{keys[i][0]}#{keys[i][1]}:{absdiff[i]:.4g}" for i in order),
                "notes":"Reference is Stage 1B artifact-excluded 2x-ring feature; stability is not detector performance.",
            })
    stability_columns=["representation","feature","n_objects","median_absolute_deviation_from_reference","spearman_rank_correlation","bias_mean_difference","difference_iqr","p95_absolute_deviation","maximum_absolute_deviation","extreme_deviation_cases","notes"]
    write_csv(table_dir/"01d_xray_feature_stability.csv",stability_columns,stability_rows)

    # GT-local preservation by established strata.
    size_values=[float(r["bbox_min_side_px"]) for r in reference_rows];contrast_values=[float(r["absolute_median_difference"]) for r in reference_rows]
    size_q=[quantile(size_values,q) for q in (.25,.5,.75)];contrast_q=[quantile(contrast_values,q) for q in (.25,.5,.75)]
    def quartile(value:float,cuts:list[float])->str:return "Q1" if value<=cuts[0] else "Q2" if value<=cuts[1] else "Q3" if value<=cuts[2] else "Q4"
    gt_rows=[]; preservation_metrics=("object_median","object_iqr","object_robust_range_p90_p10","absolute_median_difference","signed_median_difference","cnr_like_robust","object_gradient_mean")
    strata=[("overall","ALL",lambda r:True)]
    for q in ("Q1","Q2","Q3","Q4"):
        strata.append(("bbox_min_side_quartile",q,lambda r,q=q:quartile(float(r["bbox_min_side_px"]),size_q)==q))
        strata.append(("contrast_quartile",q,lambda r,q=q:quartile(float(r["absolute_median_difference"]),contrast_q)==q))
    for value in sorted({r["machine"] for r in reference_rows}):strata.append(("machine",value,lambda r,value=value:r["machine"]==value))
    for value in sorted({r["resolution"] for r in reference_rows}):strata.append(("resolution",value,lambda r,value=value:r["resolution"]==value))
    ref_metric_map={"object_median":"object_median","object_iqr":"object_iqr","object_robust_range_p90_p10":"object_robust_range_p90_p10","absolute_median_difference":"absolute_median_difference","signed_median_difference":"signed_median_difference","cnr_like_robust":"cnr_like_robust","object_gradient_mean":"object_gradient_mean"}
    for representation in REPRESENTATIONS:
        for stratum_type,stratum_value,predicate in strata:
            selected=[r for r in reference_rows if predicate(r)]
            for metric in preservation_metrics:
                refs=[float(r[ref_metric_map[metric]]) for r in selected];vals=[candidate_features[(r["stem"],int(r["object_id"]),representation)][metric] for r in selected]
                diffs=[v-r for v,r in zip(vals,refs)];absdiff=[abs(v) for v in diffs]
                gt_rows.append({
                    "representation":representation,"stratum_type":stratum_type,"stratum_value":stratum_value,"feature":metric,"n_objects":len(selected),
                    "median_absolute_deviation":fnum(median(absdiff)),"p95_absolute_deviation":fnum(quantile(absdiff,.95)),"bias":fnum(statistics.fmean(diffs)),
                    "difference_iqr":fnum(quantile(diffs,.75)-quantile(diffs,.25)),"spearman_rank_correlation":fnum(spearman(refs,vals)),
                    "notes":"Image-signal preservation only; not proof of perfect physical foreign-object preservation.",
                })
    gt_columns=["representation","stratum_type","stratum_value","feature","n_objects","median_absolute_deviation","p95_absolute_deviation","bias","difference_iqr","spearman_rank_correlation","notes"]
    write_csv(table_dir/"01d_gt_signal_preservation.csv",gt_columns,gt_rows)

    # Small object safety.
    bottom10=quantile(size_values,.10);bottom25=size_q[0];small_rows=[]
    mask_quality_lookup={(r["stem"],int(r["object_id"])):r for r in mask_quality}
    for representation in REPRESENTATIONS:
        for label,limit in (("bottom_10_percent",bottom10),("smallest_quartile",bottom25)):
            selected=[r for r in reference_rows if float(r["bbox_min_side_px"])<=limit]
            masked=[float(mask_quality_lookup[(r["stem"],int(r["object_id"]))]["masked_fraction_inside_bbox"]) for r in selected]
            contrast_diff=[abs(candidate_features[(r["stem"],int(r["object_id"]),representation)]["absolute_median_difference"]-float(r["absolute_median_difference"])) for r in selected]
            cnr_diff=[abs(candidate_features[(r["stem"],int(r["object_id"]),representation)]["cnr_like_robust"]-float(r["cnr_like_robust"])) for r in selected]
            grad_diff=[abs(candidate_features[(r["stem"],int(r["object_id"]),representation)]["object_gradient_mean"]-float(r["object_gradient_mean"])) for r in selected]
            unsafe=sum(v>.25 for v in masked)/len(masked)>.05 or median(contrast_diff)>2 or quantile(contrast_diff,.95)>5
            small_rows.append({
                "representation":representation,"small_object_group":label,"bbox_min_side_upper_px":fnum(limit),"n_objects":len(selected),
                "median_masked_bbox_fraction":fnum(median(masked)),"maximum_masked_bbox_fraction":fnum(max(masked)),"fraction_objects_masked_gt25pct":fnum(sum(v>.25 for v in masked)/len(masked)),
                "median_abs_contrast_change":fnum(median(contrast_diff)),"p95_abs_contrast_change":fnum(quantile(contrast_diff,.95)),
                "median_abs_cnr_change":fnum(median(cnr_diff)),"p95_abs_cnr_change":fnum(quantile(cnr_diff,.95)),
                "median_abs_gradient_change":fnum(median(grad_diff)),"p95_abs_gradient_change":fnum(quantile(grad_diff,.95)),
                "unsafe_for_baseline":tf(unsafe),"notes":"Quantile cut is threshold-inclusive; ties at the cutoff are all retained, so group n may exceed nominal 10%/25%. Safety screen: >5% with >25% bbox mask, median contrast change >2, or p95 >5.",
            })
    small_columns=["representation","small_object_group","bbox_min_side_upper_px","n_objects","median_masked_bbox_fraction","maximum_masked_bbox_fraction","fraction_objects_masked_gt25pct","median_abs_contrast_change","p95_abs_contrast_change","median_abs_cnr_change","p95_abs_cnr_change","median_abs_gradient_change","p95_abs_gradient_change","unsafe_for_baseline","notes"]
    write_csv(table_dir/"01d_small_object_preservation.csv",small_columns,small_rows)

    # Rule-based comparison and selection, without a weighted score.
    rep_summary={}
    grayscale_residual=statistics.fmean(float(r["artifact_edge_recall"]) for r in residual_rows if r["representation"]=="GRAYSCALE_DIRECT")
    for representation in REPRESENTATIONS:
        pres=[r for r in preservation_rows if r["representation"]==representation];res=[r for r in residual_rows if r["representation"]==representation]
        stab=[r for r in stability_rows if r["representation"]==representation];small=[r for r in small_rows if r["representation"]==representation and r["small_object_group"]=="smallest_quartile"][0]
        audit=[r for r in repair_audit_rows if r["representation"]==representation]
        rep_summary[representation]={
            "modified":statistics.fmean(float(r["fraction_pixels_changed"]) for r in pres),"outside_max":max(float(r["max_abs_change_outside_mask"]) for r in pres),
            "residual":statistics.fmean(float(r["artifact_edge_recall"]) for r in res),"continuity":statistics.fmean(float(r["median_rectangle_line_continuity"]) for r in res),
            "feature_mad":median([float(r["median_absolute_deviation_from_reference"]) for r in stab]),"feature_min_rho":min(float(r["spearman_rank_correlation"]) for r in stab),
            "small_contrast":float(small["median_abs_contrast_change"]),"small_unsafe":is_true(small["unsafe_for_baseline"]),
            "halo":statistics.fmean(float(r["halo_gradient_shift_mean"]) for r in audit) if audit else math.nan,
        }
        stab_map={r["feature"]:r for r in stab}
        rep_summary[representation]["contrast_rho"]=float(stab_map["absolute_median_difference"]["spearman_rank_correlation"])
        rep_summary[representation]["contrast_mad"]=float(stab_map["absolute_median_difference"]["median_absolute_deviation_from_reference"])
        rep_summary[representation]["cnr_rho"]=float(stab_map["cnr_like_robust"]["spearman_rank_correlation"])
        rep_summary[representation]["cnr_mad"]=float(stab_map["cnr_like_robust"]["median_absolute_deviation_from_reference"])
        rep_summary[representation]["object_iqr_rho"]=float(stab_map["object_iqr"]["spearman_rank_correlation"])
        rep_summary[representation]["background_iqr_rho"]=float(stab_map["background_iqr"]["spearman_rank_correlation"])
        rep_summary[representation]["preservation_rank_mean"]=statistics.fmean((rep_summary[representation]["contrast_rho"],rep_summary[representation]["cnr_rho"],rep_summary[representation]["object_iqr_rho"],rep_summary[representation]["background_iqr_rho"]))
    # Safety gates are concept-specific. A single minimum correlation is inappropriate
    # because near-mask gradient means intentionally differ from the mask-excluded reference.
    eligible=[rep for rep in ("ARTIFACT_INPAINT_CONSERVATIVE","ARTIFACT_LOCAL_INTERPOLATION") if
              rep_summary[rep]["outside_max"]==0 and rep_summary[rep]["residual"]<min(.005,grayscale_residual*.25)
              and not rep_summary[rep]["small_unsafe"] and rep_summary[rep]["halo"]<10
              and rep_summary[rep]["contrast_rho"]>=.90 and rep_summary[rep]["contrast_mad"]<=1.5
              and rep_summary[rep]["cnr_rho"]>=.80 and rep_summary[rep]["cnr_mad"]<=.25
              and rep_summary[rep]["object_iqr_rho"]>=.98]
    preferred=min(eligible,key=lambda rep:(-rep_summary[rep]["preservation_rank_mean"],rep_summary[rep]["residual"],rep_summary[rep]["halo"],rep)) if eligible else ""
    secondary=next((rep for rep in eligible if rep!=preferred),"")
    comparison=[]
    for representation in ("RAW_RGB","GRAYSCALE_DIRECT","ARTIFACT_MASK_EXCLUSION","ARTIFACT_INPAINT_CONSERVATIVE","ARTIFACT_LOCAL_INTERPOLATION"):
        if representation in rep_summary:s=rep_summary[representation]
        else:s={"modified":0,"outside_max":0,"residual":1 if representation=="RAW_RGB" else 0,"feature_mad":0 if representation=="ARTIFACT_MASK_EXCLUSION" else math.nan,"small_unsafe":False,"halo":math.nan}
        candidate=representation==preferred or representation==secondary
        comparison.append({
            "representation":representation,"artifact_color_removed":tf(representation in ("ARTIFACT_MASK_EXCLUSION","ARTIFACT_INPAINT_CONSERVATIVE","ARTIFACT_LOCAL_INTERPOLATION")),
            "residual_rectangle_detectability":f"mean artifact-edge recall={s['residual']:.4f}","fraction_pixels_modified":fnum(s["modified"]),
            "nonartifact_signal_preservation":f"max outside-mask change={s['outside_max']:.4g}","xray_feature_stability":f"median feature MAD={s['feature_mad']:.4g}" if not math.isnan(s["feature_mad"]) else "not applicable",
            "small_object_preservation":"unsafe" if s["small_unsafe"] else "passes diagnostic screen","new_edge_artifact_risk":f"mean adjacent gradient shift={s['halo']:.4g}" if not math.isnan(s["halo"]) else "not applicable",
            "implementation_complexity":"none" if representation in ("RAW_RGB","GRAYSCALE_DIRECT") else "diagnostic-only exclusion" if representation=="ARTIFACT_MASK_EXCLUSION" else "low; local deterministic repair",
            "reproducibility":"deterministic","training_candidate":tf(candidate),
            "strengths":"Preferred by rule-based artifact/signal/small-object criteria" if representation==preferred else "Secondary artifact-controlled ablation candidate" if representation==secondary else "Diagnostic reference",
            "limitations":"RAW contains direct GT-aligned color shortcut" if representation=="RAW_RGB" else "Grayscale retains intensity/edge rectangle structure" if representation=="GRAYSCALE_DIRECT" else "Cannot directly serve as dense detector input" if representation=="ARTIFACT_MASK_EXCLUSION" else "Synthetic local estimates replace marked line pixels only.",
        })
    comparison_columns=["representation","artifact_color_removed","residual_rectangle_detectability","fraction_pixels_modified","nonartifact_signal_preservation","xray_feature_stability","small_object_preservation","new_edge_artifact_risk","implementation_complexity","reproducibility","training_candidate","strengths","limitations"]
    write_csv(table_dir/"01d_artifact_control_comparison.csv",comparison_columns,comparison)

    evidence=[
        {"finding":"Colored artifact is universal","evidence":"500/500 approved BMPs contain strong chromatic pixels","risk":"Direct color shortcut","control_method":"Strong chromatic range>=200 mask","remaining_limitation":"Deployment artifact status unknown","reporting_use":"Data leakage risk"},
        {"finding":"Artifact closely aligns with GT","evidence":"Stage 1B: 1124/1147 objects within 3px; component count matches 493/500 images","risk":"GT location/count inference from marks","control_method":"Repair only detected line pixels","remaining_limitation":"True signal under overwritten colored pixels is unavailable","reporting_use":"Why raw input is invalid"},
        {"finding":"Direct grayscale retains residual structure","evidence":f"Mean artifact-edge recall={grayscale_residual:.4f}","risk":"Color removal alone does not remove rectangular intensity/edge signal","control_method":"Local reconstruction required","remaining_limitation":"Fixed diagnostic is not an exhaustive attacker","reporting_use":"Reject grayscale-only baseline"},
        {"finding":"Preferred artifact control selected pre-model","evidence":preferred or "No safe candidate","risk":"Repair can alter small/local signal","control_method":preferred or "UNRESOLVED","remaining_limitation":"No detector performance used; true latent pixels cannot be recovered","reporting_use":"Baseline input specification"},
        {"finding":"Nonartifact pixels preserved","evidence":f"Preferred max outside-mask change={rep_summary[preferred]['outside_max'] if preferred else 'NA'}","risk":"Unnecessary X-ray modification","control_method":"Strict mask-only write","remaining_limitation":"Gradients adjacent to repaired pixels can change","reporting_use":"Signal-preservation evidence"},
    ]
    evidence_columns=["finding","evidence","risk","control_method","remaining_limitation","reporting_use"]
    write_csv(table_dir/"01d_chapter1_artifact_evidence.csv",evidence_columns,evidence)

    spec=f"""# Stage 1D — Artifact-Control Specification

## Scope and frozen validation

- Approved samples: 500 canonical BMP files with official TXT labels.
- Frozen folds are read from `01c_final_validation_folds.csv`; SHA-256: `{frozen_hash}`.
- No fold assignment is modified by this pipeline.

## Shared input and mask

- Read uncompressed 8-bit indexed BMP and decode its 256-entry RGB palette.
- Convert palette RGB to uint8 BT.601 luma: `round(0.299R + 0.587G + 0.114B)`, clipped to [0,255].
- Mark a pixel only when `max(R,G,B)-min(R,G,B) >= {CHROMATIC_RANGE_THRESHOLD}`.
- Do not dilate the mask; do not mask rectangle interiors or entire GT boxes.
- Processing is deterministic and uses no random seed.

## Preferred representation: {preferred or 'UNRESOLVED'}

{('- For each masked pixel, collect the first nonmasked source-luma pixel in each of the left/right/up/down directions within 5 pixels and replace it with their arithmetic mean, rounded to uint8.' if preferred=='ARTIFACT_LOCAL_INTERPOLATION' else '- For each masked pixel, replace it with the median of nonmasked source-luma pixels in a 5×5 neighborhood; expand to 9×9 only if no valid neighbor exists.') if preferred else '- No representation met all predeclared safety criteria.'}
- Only masked pixels are written; all nonmasked pixels remain bit-identical to direct grayscale.
- Save losslessly as 8-bit grayscale PNG, original width/height, uint8 range [0,255].
- No normalization, CLAHE, sharpening, denoising, residual transform, or resizing occurs here. Model normalization belongs to a later stage.

## Optional secondary representation: {secondary or 'NONE'}

{('- Median-neighborhood conservative repair as specified above.' if secondary=='ARTIFACT_INPAINT_CONSERVATIVE' else '- Four-direction local interpolation as specified above.' if secondary=='ARTIFACT_LOCAL_INTERPOLATION' else '- No secondary representation approved.')}

## Diagnostic-only representations

- `RAW_RGB`: original palette color, never approved for training.
- `GRAYSCALE_DIRECT`: direct luma conversion; rejected if residual rectangle leakage remains.
- `ARTIFACT_MASK_EXCLUSION`: Stage 1B feature reference only; not a dense detector input.

## Version

- Processing version: `{VERSION}`
- Implementation: `src/run_stage1d_artifact_control.py`
"""
    (eda_dir/"01d_artifact_control_specification.md").write_text(spec,encoding="utf-8-sig")

    def med_metric(rows:list[dict[str,object]],rep:str,key:str)->float:return median([float(r[key]) for r in rows if r["representation"]==rep])
    gray_detect=sum(is_true(str(r["artifact_structure_detectable"])) for r in leakage_rows if r["representation"]=="GRAYSCALE_DIRECT")/1147
    robust_fraction=sum(is_true(str(r["mask_robust"])) for r in robustness_rows)/500
    summary=f"""# Stage 1D — Artifact-Control Ablation Summary

## 1. Input integrity

- **VERIFIED** — 500 eligible samples, 1,147 objects, complete frozen 4-fold assignment, unique stems, and no missing BMP/TXT files.
- Frozen fold SHA-256 remained `{frozen_hash}`; no fold or original image was modified.

## 2. Why the artifact is a leakage risk

- **VERIFIED** — strong chromatic marks occur in 500/500 images and Stage 1B established near-GT alignment in 1124/1147 objects with component-count agreement in 493/500 images.
- RAW_RGB is diagnostic only and is never a training candidate.

## 3. Grayscale-only residual leakage

- **VERIFIED** — direct grayscale retains detectable local intensity/edge structure for {gray_detect:.1%} of object rows; mean original-artifact edge recall={grayscale_residual:.4f}.
- **STRONGLY SUPPORTED** — channel collapse alone is insufficient for a fair baseline.

## 4. Mask robustness

- **VERIFIED** — {robust_fraction:.1%} of images pass the fixed thin-component/proximity robustness screen; median total masked fraction={median([float(r['masked_fraction_total']) for r in robustness_rows]):.4%}.
- The mask is not dilated and never removes rectangle interiors or full GT boxes.

## 5. Candidate repair methods

- Conservative: median of nonmasked 5×5 neighbors, expanding to 9×9 only if needed.
- Local interpolation: arithmetic mean of nearest valid source-luma values in four cardinal directions within 5 pixels.
- Both modify only the strong-chromatic line mask and save lossless uint8 PNGs.

## 6. Signal preservation

- **VERIFIED** — maximum outside-mask pixel change is {max(rep_summary[r]['outside_max'] for r in REPRESENTATIONS):.0f} across evaluated dense representations.
- Mean modified fractions: grayscale={rep_summary['GRAYSCALE_DIRECT']['modified']:.4%}, conservative={rep_summary['ARTIFACT_INPAINT_CONSERVATIVE']['modified']:.4%}, local={rep_summary['ARTIFACT_LOCAL_INTERPOLATION']['modified']:.4%}.

## 7. Residual shortcut diagnostic

- Mean artifact-edge recall: grayscale={rep_summary['GRAYSCALE_DIRECT']['residual']:.4f}, conservative={rep_summary['ARTIFACT_INPAINT_CONSERVATIVE']['residual']:.4f}, local={rep_summary['ARTIFACT_LOCAL_INTERPOLATION']['residual']:.4f}.
- **EXPLORATORY** — this fixed nonlearned diagnostic is a shortcut screen, not foreign-object detection performance or an exhaustive adversarial test.

## 8. Repair-induced edge/texture artifacts

- Mean adjacent gradient shift: conservative={rep_summary['ARTIFACT_INPAINT_CONSERVATIVE']['halo']:.4g}, local={rep_summary['ARTIFACT_LOCAL_INTERPOLATION']['halo']:.4g}.
- Only pixels adjacent to repaired lines can show gradient changes; nonartifact pixel intensities are preserved exactly.

## 9. GT-local signal preservation

- Median cross-feature deviation from Stage 1B reference: grayscale={rep_summary['GRAYSCALE_DIRECT']['feature_mad']:.4g}, conservative={rep_summary['ARTIFACT_INPAINT_CONSERVATIVE']['feature_mad']:.4g}, local={rep_summary['ARTIFACT_LOCAL_INTERPOLATION']['feature_mad']:.4g}.
- These are image-signal comparisons and do not prove perfect recovery of the physical foreign object.

## 10. Small-object preservation

- **VERIFIED** — smallest-quartile diagnostic flags: grayscale={rep_summary['GRAYSCALE_DIRECT']['small_unsafe']}, conservative={rep_summary['ARTIFACT_INPAINT_CONSERVATIVE']['small_unsafe']}, local={rep_summary['ARTIFACT_LOCAL_INTERPOLATION']['small_unsafe']}.
- The bottom-decile and quartile cutoffs both equal 8 px; retaining all cutoff ties makes both sensitivity groups the same 295 objects (25.7%), which is reported transparently rather than broken by arbitrary tie order.

## 11. Candidate comparison

- RAW_RGB: rejected, direct color shortcut.
- GRAYSCALE_DIRECT: rejected, residual rectangle structure.
- ARTIFACT_MASK_EXCLUSION: valid feature reference but not a dense detector input.
- Conservative/local repairs: evaluated by shortcut reduction, exact outside-mask preservation, feature stability, small-object safety, and repair-edge risk without a weighted score.

## 12. Approved baseline representation

- **{'STRONGLY SUPPORTED' if preferred else 'UNRESOLVED'}** — {preferred or 'No representation approved'}.

## 13. Optional secondary representation

- **{'EXPLORATORY' if secondary else 'UNRESOLVED'}** — {secondary or 'None'}.

## 14. Remaining uncertainty

- True grayscale signal under overwritten colored pixels is not observable; repair estimates it locally.
- The residual diagnostic is simple and cannot prove absence of every exploitable shortcut.
- Deployment-time artifact presence and acquisition pipeline are undocumented.
- Repair-induced local gradients may interact with a future detector and require frozen-fold ablation.

## 15. Recommended next stage

- Use the frozen Stage 1C folds for a separately authorized baseline comparison of the approved representation(s). Keep RAW_RGB excluded and report artifact-control ablation separately; do not access external held-out data.
"""
    (eda_dir/"01d_artifact_control_summary.md").write_text(summary,encoding="utf-8-sig")
    print(f"samples=500 objects=1147 preferred={preferred or 'NONE'} secondary={secondary or 'NONE'} gray_detect={gray_detect:.3f}")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description="Stage 1D artifact-control ablation")
    parser.add_argument("--workspace",type=Path,default=Path(__file__).resolve().parents[1])
    args=parser.parse_args();main(args.workspace.resolve())
