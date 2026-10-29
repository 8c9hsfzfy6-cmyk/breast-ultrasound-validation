# ============================================================
# V6：统一、可追溯的特征提取
# CPU；不训练；不需要人工勾选
# 依赖：numpy pandas scipy scikit-image Pillow（Kaggle常见环境已有）
# ============================================================

from pathlib import Path
from collections import defaultdict
import os
import json
import time
import hashlib
import shutil
import zipfile

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from skimage.measure import perimeter
from IPython.display import display, FileLink

INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")

# 默认路径来自你已经成功读取的数据。
V3_DIR = (
    "/kaggle/input/datasets/kokomixueyue/bus-v3"
)

# 留空时找当前working中最新的BrEaST检查结果。
# 如果重启后找不到，可填重新上传的检查结果ZIP路径。
BREAST_AUDIT = ""

SIDE = 192
LEVELS = 16
COVERAGE = 0.80

# 固定规则，不根据BrEaST结果调整。
# 相同列名只是便于核对，不表示等价于旧V3实现。


def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_external():
    if BREAST_AUDIT.strip():
        p = Path(BREAST_AUDIT.strip())
    else:
        found = list(
            WORK.glob(
                "breast_external_audit_*/external_manifest_audit.csv"
            )
        )
        if found:
            p = max(found, key=lambda x: x.stat().st_mtime)
        else:
            found = list(INPUT.rglob("breast_external_audit_*.zip"))
            if len(found) != 1:
                raise RuntimeError(
                    "找不到唯一的BrEaST检查结果。"
                    "请填写BREAST_AUDIT，或重跑之前的外部数据检查。"
                )
            p = found[0]

    if p.is_dir():
        p = p / "external_manifest_audit.csv"

    if p.suffix.lower() == ".zip":
        with zipfile.ZipFile(p) as z:
            names = [
                n for n in z.namelist()
                if Path(n).name == "external_manifest_audit.csv"
            ]
            if len(names) != 1:
                raise ValueError("外部结果ZIP中的清单不唯一。")
            with z.open(names[0]) as f:
                df = pd.read_csv(f, dtype={"case_id": str})
    else:
        df = pd.read_csv(p, dtype={"case_id": str})

    if len(df) != 256 or not df.case_id.is_unique:
        raise ValueError("BrEaST清单不是256个唯一病例。")

    df = df[df.status == "ready_single_mask"].copy()
    if len(df) != 252:
        raise ValueError("预期252例带单个病灶掩膜的BrEaST病例。")

    rows = []
    for r in df.itertuples(index=False):
        masks = json.loads(r.mask_paths)
        hashes = json.loads(r.mask_sha256)
        if len(masks) != 1 or len(hashes) != 1:
            raise ValueError("发现非单个病灶掩膜。")

        rows.append({
            "dataset": "BrEaST",
            "image_id": str(r.case_id),
            "patient_id": str(r.case_id),
            "image_path": r.image_path,
            "mask_path": masks[0],
            "expected_image_sha": r.image_sha256,
            "expected_mask_sha": hashes[0],
            "label": int(float(r.label)),
        })

    return pd.DataFrame(rows)


# ------------------------------------------------------------
# 特征实现保存为独立源码，避免再次只留下CSV。
# ------------------------------------------------------------
EXTRACTOR_SOURCE = r'''
import numpy as np
from scipy import ndimage as ndi
from skimage.measure import perimeter

SIDE = 192
LEVELS = 16
COVERAGE = 0.80

def texture_features(x, region, valid):
    r = region & valid
    v = x[r]
    names = [
        "mean", "std", "p10", "median", "p90",
        "entropy", "edge", "contrast", "homogeneity", "energy"
    ]
    if len(v) < 9:
        return {**{k: np.nan for k in names}, "missing": 1.0}

    hist = np.histogram(v, bins=LEVELS, range=(0, 1))[0].astype(float)
    probabilities = hist[hist > 0] / hist.sum()
    entropy = -np.sum(probabilities * np.log2(probabilities))

    # 梯度只统计邻域完全有效且位于该区域内的像素。
    interior = ndi.binary_erosion(
        r, structure=np.ones((3, 3), bool), border_value=0
    )
    gx = ndi.sobel(x, axis=1, mode="reflect") / 8.0
    gy = ndi.sobel(x, axis=0, mode="reflect") / 8.0
    edge = (
        float(np.hypot(gx, gy)[interior].mean())
        if interior.any() else np.nan
    )

    # 四个方向、距离1；仅使用两个端点都位于区域内的像素对。
    q = np.clip((x * LEVELS).astype(int), 0, LEVELS - 1)
    h, w = x.shape
    counts = np.zeros((LEVELS, LEVELS), dtype=float)

    for dy, dx in [(0, 1), (1, 0), (1, 1), (1, -1)]:
        ya, yb = max(0, -dy), min(h, h - dy)
        xa, xb = max(0, -dx), min(w, w - dx)
        a = (slice(ya, yb), slice(xa, xb))
        b = (slice(ya + dy, yb + dy), slice(xa + dx, xb + dx))
        keep = r[a] & r[b]
        if keep.any():
            codes = q[a][keep] * LEVELS + q[b][keep]
            counts += np.bincount(
                codes, minlength=LEVELS * LEVELS
            ).reshape(LEVELS, LEVELS)

    counts = counts + counts.T
    if counts.sum() > 0:
        p = counts / counts.sum()
        ii, jj = np.indices(p.shape)
        contrast = float(np.sum(p * (ii - jj) ** 2))
        homogeneity = float(np.sum(p / (1 + (ii - jj) ** 2)))
        energy = float(np.sqrt(np.sum(p ** 2)))
    else:
        contrast = homogeneity = energy = np.nan

    return {
        "mean": float(v.mean()),
        "std": float(v.std(ddof=0)),
        "p10": float(np.percentile(v, 10)),
        "median": float(np.median(v)),
        "p90": float(np.percentile(v, 90)),
        "entropy": float(entropy),
        "edge": edge,
        "contrast": contrast,
        "homogeneity": homogeneity,
        "energy": energy,
        "missing": 0.0,
    }


def posterior_measurement(x, m, valid):
    yy, xx = np.where(m)
    h, w = m.shape
    x0, x1 = int(xx.min()), int(xx.max())
    y0, y1 = int(yy.min()), int(yy.max())
    bw, bh = x1 - x0 + 1, y1 - y0 + 1

    depth = max(3, round(0.5 * bh))
    width = max(3, round(0.5 * bw))
    gap = max(1, round(0.05 * bw))

    post = np.zeros_like(m)
    planned_rows = set()
    desired_post = 0

    for col in range(x0, x1 + 1):
        ys = np.flatnonzero(m[:, col])
        if len(ys):
            start = int(ys.max()) + 1
            desired_post += depth
            planned_rows.update(range(start, start + depth))
            post[start:min(h, start + depth), col] = True

    def rect(lo, hi):
        r = np.zeros_like(m)
        rows = [y for y in planned_rows if 0 <= y < h]
        cols = list(range(max(0, lo), min(w, hi)))
        if rows and cols:
            r[np.ix_(rows, cols)] = True
        return r & valid & ~m

    left = rect(x0 - gap - width, x0 - gap)
    right = rect(x1 + gap + 1, x1 + gap + 1 + width)
    post &= valid & ~m

    cp = post.sum() / max(1, desired_post)
    cl = left.sum() / max(1, len(planned_rows) * width)
    cr = right.sum() / max(1, len(planned_rows) * width)

    values = []
    for row in sorted(planned_rows):
        if not 0 <= row < h:
            continue
        p, l, r = x[row, post[row]], x[row, left[row]], x[row, right[row]]
        if min(len(p), len(l), len(r)) >= 3:
            values.append(
                float(np.median(p) - 0.5 * (np.median(l) + np.median(r)))
            )

    cm = len(values) / max(1, len(planned_rows))
    ok = bool(
        min(cp, cl, cr, cm) >= COVERAGE
        and min(post.sum(), left.sum(), right.sum()) >= 9
        and len(values) >= 3
    )
    return (
        float(np.median(values)) if ok else np.nan,
        {
            "posterior_usable": ok,
            "posterior_coverage": float(cp),
            "left_coverage": float(cl),
            "right_coverage": float(cr),
            "matched_row_coverage": float(cm),
        }
    )


def extract_features(x, m, valid):
    if x.shape != m.shape or m.shape != valid.shape:
        raise ValueError("Shape mismatch")
    if m.sum() < 9:
        raise ValueError("Lesion mask has fewer than 9 pixels")
    if (m & ~valid).any():
        raise ValueError("Lesion overlaps invalid image pixels")

    thickness = max(1, round(np.sqrt(m.sum()) * 0.06))
    core = ndi.binary_erosion(m, iterations=thickness)
    if core.sum() < 9:
        core = m.copy()
    rim = ndi.binary_dilation(m, iterations=thickness) & ~core

    result = {}
    for name, region in [("core", core), ("rim", rim)]:
        result.update({
            name + "__" + k: v
            for k, v in texture_features(x, region, valid).items()
        })

    yy, xx = np.where(m)
    bw = int(xx.max() - xx.min() + 1)
    bh = int(yy.max() - yy.min() + 1)
    area = float(m.sum())
    perim = float(perimeter(m, neighborhood=4))
    coordinates = np.column_stack([yy, xx]).astype(float)
    eig = np.linalg.eigvalsh(
        np.cov(coordinates, rowvar=False, bias=True)
    )

    result.update({
        # 这些是V6明确定义，不声称与旧V3一致。
        "shape__area": area / m.size,
        "shape__aspect": bw / bh,
        "shape__fill": area / (bw * bh),
        "shape__roughness": perim ** 2 / (4 * np.pi * area),
        "shape__elongation": float(
            np.sqrt(eig[-1] / max(eig[0], 1e-8))
        ),
        "rim_core__difference": (
            result["rim__mean"] - result["core__mean"]
        ),
    })

    delta, quality = posterior_measurement(x, m, valid)
    result["postdepth__delta"] = delta
    result["postdepth__missing"] = float(not quality["posterior_usable"])

    quality.update({
        "core_valid_fraction": float(
            (core & valid).sum() / max(1, core.sum())
        ),
        "rim_valid_fraction": float(
            (rim & valid).sum() / max(1, rim.sum())
        ),
    })
    return result, quality
'''

namespace = {}
exec(compile(EXTRACTOR_SOURCE, "v6_extractor.py", "exec"), namespace)
extract_features = namespace["extract_features"]


def main_v6():
    source = Path(V3_DIR)
    if not (source / "manifest.csv").is_file():
        raise FileNotFoundError(
            "找不到V3目录，请修改V3_DIR为包含manifest.csv的目录。"
        )

    bus = pd.read_csv(
        source / "manifest.csv",
        dtype={"image_id": str, "patient_id": str},
    )

    if (
        len(bus) != 1875
        or bus.patient_id.nunique() != 1064
        or not bus.image_id.is_unique
    ):
        raise ValueError("BUS-BRA清单数量或ID不符合预期。")

    old = pd.read_csv(source / "features_raw.csv")
    if len(old) != len(bus):
        raise ValueError("旧特征与原始清单行数不同。")

    # 在任何排序或合并之前，按原结果包行顺序绑定ID。
    old.index = bus.image_id.to_numpy()

    bus = bus[
        ["image_id", "patient_id", "image_path", "mask_path", "label"]
    ].copy()
    bus["dataset"] = "BUS-BRA"
    bus["expected_image_sha"] = ""
    bus["expected_mask_sha"] = ""

    ext = read_external()
    cohort = pd.concat([bus, ext], ignore_index=True)

    if cohort[["patient_id", "image_id", "label"]].isna().any().any():
        raise ValueError("清单存在缺失ID或标签。")

    cohort["patient_key"] = (
        cohort.dataset + ":" + cohort.patient_id.astype(str)
    )
    cohort["uid"] = cohort.dataset + ":" + cohort.image_id.astype(str)
    if not cohort.uid.is_unique:
        raise ValueError("联合清单UID重复。")

    index = defaultdict(set)
    print("建立图像索引……", flush=True)
    for directory, _, files in os.walk(INPUT, followlinks=True):
        for name in files:
            if Path(name).suffix.lower() in {".png", ".jpg", ".jpeg"}:
                index[name].add(
                    (Path(directory) / name).resolve()
                )

    hash_cache = {}

    def digest(p):
        if p not in hash_cache:
            hash_cache[p] = file_sha(p)
        return hash_cache[p]

    def locate(saved, expected, kind):
        p = Path(saved)
        candidates = set(index.get(p.name, set()))
        if p.is_file():
            candidates.add(p.resolve())

        candidates = sorted(candidates)
        if isinstance(expected, str) and expected:
            candidates = [
                q for q in candidates if digest(q) == expected
            ]
        else:
            folders = {"images", "image"} if kind == "image" else {"masks", "mask"}
            typed = [
                q for q in candidates
                if any(part.lower() in folders for part in q.parts)
            ]
            if typed:
                candidates = typed

        if len(candidates) > 1:
            if len({digest(q) for q in candidates}) == 1:
                candidates = candidates[:1]

        if len(candidates) != 1:
            raise ValueError(
                f"{kind} {p.name}: 无法唯一匹配，候选={candidates}"
            )
        return candidates[0]

    run = time.strftime("%Y%m%d_%H%M%S") + f"_{time.time_ns()%1000000}"
    out = WORK / ("bus_v6_features_" + run)
    out.mkdir(parents=True)

    (out / "v6_extractor.py").write_text(
        EXTRACTOR_SOURCE, encoding="utf-8"
    )

    cohort.to_csv(out / "cohort.csv", index=False)

    feature_rows, quality_rows, file_rows, errors = [], [], [], []
    started = time.time()

    for i, row in enumerate(cohort.itertuples(index=False)):
        try:
            ip = locate(row.image_path, row.expected_image_sha, "image")
            mp = locate(row.mask_path, row.expected_mask_sha, "mask")

            with Image.open(ip) as im:
                gray = np.asarray(im.convert("L"), dtype=np.uint8)
                valid = np.asarray(im.convert("RGBA"))[:, :, 3] > 0

            # 哈希使用原始尺寸的解码灰度像素，不受文件压缩方式影响。
            ph = hashlib.sha256()
            ph.update(str(gray.shape).encode())
            ph.update(gray.tobytes())
            pixel_hash = ph.hexdigest()

            with Image.open(mp) as im:
                arr = np.asarray(im)
                if (
                    arr.ndim == 3
                    and not np.all(arr[:, :, :3] == arr[:, :, :1])
                ):
                    raise ValueError("彩色掩膜编码尚未定义。")
                mask = (
                    (np.asarray(im.convert("L")) > 0)
                    & (np.asarray(im.convert("RGBA"))[:, :, 3] > 0)
                )

            if gray.shape != mask.shape:
                raise ValueError("原图与掩膜尺寸不同。")
            if (mask & ~valid).any():
                raise ValueError("原始掩膜覆盖透明区域。")

            scale = SIDE / max(gray.shape)
            size = (
                max(1, round(gray.shape[1] * scale)),
                max(1, round(gray.shape[0] * scale)),
            )

            x = np.asarray(
                Image.fromarray(
                    gray.astype(np.float32) / 255.0
                ).resize(size, Image.Resampling.BILINEAR)
            )
            m = np.asarray(
                Image.fromarray(mask.astype("uint8")).resize(
                    size, Image.Resampling.NEAREST
                )
            ) > 0
            v = np.asarray(
                Image.fromarray(valid.astype("uint8")).resize(
                    size, Image.Resampling.NEAREST
                )
            ) > 0

            # 仅将像素、掩膜、有效区域传入，不传入任何诊断标签。
            features, quality = extract_features(x, m, v)

            feature_rows.append({"uid": row.uid, **features})
            quality_rows.append({"uid": row.uid, **quality})
            file_rows.append({
                "uid": row.uid,
                "dataset": row.dataset,
                "image_id": row.image_id,
                "image_path": str(ip),
                "mask_path": str(mp),
                "image_sha256": digest(ip),
                "mask_sha256": digest(mp),
                "decoded_gray_sha256": pixel_hash,
            })

        except Exception as exc:
            errors.append({"uid": row.uid, "error": str(exc)})

        if (i + 1) % 100 == 0:
            pd.DataFrame(feature_rows).to_csv(
                out / "features.csv", index=False
            )
            print(
                f"{i+1}/{len(cohort)}，错误={len(errors)}，"
                f"耗时={time.time()-started:.0f}秒",
                flush=True,
            )

    features = pd.DataFrame(feature_rows)
    quality = pd.DataFrame(quality_rows)
    files = pd.DataFrame(file_rows)

    features.to_csv(out / "features.csv", index=False)
    quality.to_csv(out / "quality.csv", index=False)
    files.to_csv(out / "resolved_files.csv", index=False)
    pd.DataFrame(
        errors, columns=["uid", "error"]
    ).to_csv(out / "errors.csv", index=False)

    # 旧V3与新版同名特征核对，容差固定，不据此自动宣称等价。
    comparison = []
    if len(features):
        new_bus = features[
            features.uid.str.startswith("BUS-BRA:")
        ].copy()
        new_bus.index = new_bus.uid.str.split(":", n=1).str[1]
        ids = old.index.intersection(new_bus.index)

        for col in sorted(set(old.columns) & set(new_bus.columns)):
            a = pd.to_numeric(old.loc[ids, col], errors="coerce").to_numpy()
            b = pd.to_numeric(new_bus.loc[ids, col], errors="coerce").to_numpy()
            finite = np.isfinite(a) & np.isfinite(b)
            matching = np.isclose(
                a, b, rtol=1e-5, atol=1e-7, equal_nan=True
            )

            comparison.append({
                "feature": col,
                "compared_images": len(ids),
                "finite_pairs": int(finite.sum()),
                "missing_pattern_differences": int(
                    (np.isnan(a) != np.isnan(b)).sum()
                ),
                "fraction_matching_tolerance": float(matching.mean()),
                "median_absolute_difference": (
                    float(np.median(np.abs(a[finite] - b[finite])))
                    if finite.any() else np.nan
                ),
                "max_absolute_difference": (
                    float(np.max(np.abs(a[finite] - b[finite])))
                    if finite.any() else np.nan
                ),
            })

    pd.DataFrame(comparison).to_csv(
        out / "comparison_with_v3.csv", index=False
    )

    # 完全相同的解码灰度图；不能代替近重复/患者重叠检查。
    overlap = pd.DataFrame()
    if len(files):
        cross_hashes = files.groupby(
            "decoded_gray_sha256"
        ).dataset.nunique()
        cross_hashes = cross_hashes[cross_hashes > 1].index
        overlap = files[
            files.decoded_gray_sha256.isin(cross_hashes)
        ]

    overlap.to_csv(out / "exact_cross_dataset_overlap.csv", index=False)

    base_columns = [
        c for c in features.columns
        if c != "uid" and not c.startswith("postdepth__")
    ]

    protocol = {
        "version": "V6-unified-features-1",
        "extractor_sha256": hashlib.sha256(
            EXTRACTOR_SOURCE.encode("utf-8")
        ).hexdigest(),
        "side": SIDE,
        "grayscale_range": [0, 1],
        "resize": "Preserve aspect ratio; bilinear image; nearest masks",
        "glcm": "16 levels; distance 1; four directions pooled; symmetric; both pixels inside region",
        "feature_sets": {
            "base": base_columns,
            "base_plus_posterior": base_columns + [
                "postdepth__delta", "postdepth__missing"
            ],
        },
        "labels_used_in_extractor": False,
        "feature_selection_or_training": False,
        "missing_values": "Retained as NaN; future imputation fitted on training data only",
        "relationship_to_v3": "New explicit implementation; equivalence not assumed",
        "posterior": "Fixed V5A raw measurement; no changes using BrEaST signs",
        "limitations": [
            "No text/caliper removal",
            "Depth direction assumed from image rows",
            "No assertion of clinical contour correctness",
            "Exact grayscale overlap check does not detect all near duplicates",
            "BrEaST has already been used for posterior-sign analysis; disclose this history",
        ],
    }

    summary = {
        "expected_images": int(len(cohort)),
        "processed_images": int(len(features)),
        "errors": len(errors),
        "feature_count": max(0, len(features.columns) - 1),
        "exact_cross_dataset_overlap_rows": int(len(overlap)),
        "seconds": round(time.time() - started, 1),
        "ready_for_review": bool(
            len(features) == len(cohort)
            and not errors
            and len(overlap) == 0
        ),
    }

    if len(quality):
        q = quality.merge(
            cohort[["uid", "dataset"]], on="uid", validate="one_to_one"
        )
        summary["posterior_availability"] = (
            q.groupby("dataset").posterior_usable
            .agg(["count", "sum", "mean"])
            .to_dict(orient="index")
        )

    for name, data in [
        ("protocol.json", protocol),
        ("summary.json", summary),
    ]:
        (out / name).write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # 同时保存本次实际执行单元，包含读文件和预处理代码。
    try:
        history = get_ipython().history_manager.input_hist_raw
        this_cell = history[-1]
        (out / "executed_cell.py").write_text(
            this_cell, encoding="utf-8"
        )
    except Exception as exc:
        (out / "source_export_note.txt").write_text(
            "单元历史导出失败，请另存Notebook：" + str(exc),
            encoding="utf-8",
        )

    archive = Path(shutil.make_archive(str(out), "zip", out))

    print("\n完成：")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\n与V3同名特征的核对：")
    if comparison:
        display(pd.DataFrame(comparison))

    print("\n结果包：", archive)
    print("请下载ZIP发给我；现在不要自行删除缺失值或开始训练。")
    display(FileLink(str(archive)))


main_v6()

# ======================== 代码结束 ========================