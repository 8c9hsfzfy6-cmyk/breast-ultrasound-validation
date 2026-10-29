
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
