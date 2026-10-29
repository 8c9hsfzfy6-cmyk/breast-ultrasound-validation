# ================= BUS-UCLM 冻结模型外部验证 =================
# CPU运行；不训练、不调参、不重新校准、不改变阈值。
# 使用既有人工掩膜，属于有标注ROI条件下的分类验证。

from pathlib import Path
from collections import defaultdict
from datetime import datetime
import io
import json
import hashlib
import zipfile
import shutil
import platform

import numpy as np
import pandas as pd
import scipy
import sklearn
import joblib
from PIL import Image
from scipy.special import logit
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
from IPython.display import display, FileLink

# 一般保持空白即可自动寻找。
# 若自动寻找失败，填写Kaggle中的ZIP或已解压目录路径。
MODEL_PATH = (
    "/kaggle/working/"
    "bus_v6_external_training_20261013_121259_911015.zip"
)

AUDIT_PATH = (
    "/kaggle/working/"
    "bus_uclm_targeted_check_20261013_130605_803176.zip"
)

BOOTSTRAPS = 2000
BOOTSTRAP_SEED = 20261013

INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
OUT = WORK / ("bus_uclm_external_" + RUN_ID)
OUT.mkdir(parents=True, exist_ok=False)

EXTRACTOR_HASH = (
    "fc020bfc28022f1b476d435d0dc8728aaee"
    "48e03d181007bce545aee22a1dda2"
)
MANIFEST_HASH = (
    "7cce7f175b90e0d50cbe3c2933356a8d82"
    "ae33219e78d4a575ffc84a3456afe5"
)
MODEL_HASHES = {
    "base_LR.joblib":
        "8a9d637defd779932b20c98c19dd1da58c831532dc8673fcad5b8d708467366d",
    "base_ensemble.joblib":
        "4bf456dc03fffbcd155a83a5300349e0b51c40d4fae98994d60b0827f61bec8b",
    "base_missing_LR.joblib":
        "5c60b2a14849d3911aad681a8f77108c2ed8dd41ed061e2065c9f6bc8183d8c1",
    "base_missing_ensemble.joblib":
        "739c50d5c6a50fd349743fc541d2e4c6b9c0439b706d0848e6b993e4c25ef48a",
    "base_posterior_LR.joblib":
        "666fbafd822433e2bde58c937080e7c419d67d1856f2187ee416dfecc9f4b209",
    "base_posterior_ensemble.joblib":
        "66f46dc5319fb09f73a1142e8588f9869ff7fea3f34b716c31f43ca4b22ca999",
}


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(name, obj):
    (OUT / name).write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def read_member(source, basename):
    """只读取文件，不解压、不执行包内其他代码。"""
    source = Path(source)
    if source.is_dir():
        matches = list(source.rglob(basename))
        matches = [p for p in matches if p.is_file()]
        if len(matches) != 1:
            raise ValueError(
                f"{source} 内 {basename} 数量不是1：{len(matches)}"
            )
        return matches[0].read_bytes()

    with zipfile.ZipFile(source) as z:
        names = [
            n for n in z.namelist()
            if not n.endswith("/")
            and Path(n).name == basename
            and "__MACOSX" not in n
        ]
        if len(names) != 1:
            raise ValueError(
                f"{source.name} 内 {basename} 数量不是1"
            )
        return z.read(names[0])


def discover(explicit, zip_prefix, anchor):
    if explicit.strip():
        p = Path(explicit.strip())
        if not p.exists():
            raise FileNotFoundError(str(p))
        return p

    candidates = []
    for root in (INPUT, WORK):
        if not root.exists():
            continue
        candidates += list(root.rglob(zip_prefix + "*.zip"))
        candidates += [
            p.parent for p in root.rglob(anchor)
            if p.is_file()
        ]

    candidates = sorted(set(candidates), key=str)
    if not candidates:
        raise FileNotFoundError(
            f"没有找到 {zip_prefix}。请添加对应结果包，"
            "或填写顶部路径。"
        )

    # ZIP与解压目录同时存在时，仅合并内容一致的副本。
    groups = defaultdict(list)
    for p in candidates:
        try:
            groups[sha_bytes(read_member(p, anchor))].append(p)
        except Exception:
            continue

    if len(groups) != 1:
        print("候选位置：")
        for p in candidates:
            print(p)
        raise RuntimeError(
            "未找到唯一内容版本，请在顶部填写明确路径。"
        )
    return next(iter(groups.values()))[0]


def parse_flag(series):
    mapping = {
        "true": True, "yes": True, "y": True, "1": True,
        "false": False, "no": False, "n": False, "0": False,
    }
    values = series.astype(str).str.strip().str.lower().map(mapping)
    if values.isna().any():
        raise ValueError("发现无法识别的元数据标记。")
    return values.astype(bool)


def main():
    model_source = discover(
        MODEL_PATH,
        "bus_v6_external_training_20261013_121259_911015",
        "model_hashes.json",
    )
    audit_source = discover(
        AUDIT_PATH,
        "bus_uclm_targeted_check_20261013_130605_803176",
        "manifest_with_metadata.csv",
    )
    print("模型来源：", model_source)
    print("审计来源：", audit_source)

    protocol = json.loads(
        read_member(model_source, "protocol.json")
    )
    expected_version = protocol["versions"]["scikit-learn"]
    if sklearn.__version__ != expected_version:
        raise RuntimeError(
            f"当前scikit-learn={sklearn.__version__}，"
            f"原模型版本={expected_version}。"
            "先停止，不跨版本加载模型；请把这行输出发回。"
        )

    extractor_bytes = read_member(model_source, "v6_extractor.py")
    if sha_bytes(extractor_bytes) != EXTRACTOR_HASH:
        raise ValueError("V6特征提取器哈希不一致。")

    manifest_bytes = read_member(
        audit_source, "manifest_with_metadata.csv"
    )
    if sha_bytes(manifest_bytes) != MANIFEST_HASH:
        raise ValueError("审计清单与已核对版本不一致。")

    # 仅执行已核验哈希的原V6提取器。
    namespace = {"__name__": "frozen_v6_extractor"}
    exec(
        compile(extractor_bytes, "v6_extractor.py", "exec"),
        namespace,
    )
    extract_features = namespace["extract_features"]
    side = int(namespace["SIDE"])
    (OUT / "v6_extractor.py").write_bytes(extractor_bytes)

    manifest = pd.read_csv(
        io.BytesIO(manifest_bytes),
        dtype={"image_id": str, "patient_id": str},
    )
    if len(manifest) != 683 or not manifest.image_id.is_unique:
        raise ValueError("审计清单数量或图像ID异常。")

    for col in ("Doppler_flag", "Marks_flag", "Combined_flag"):
        manifest[col] = parse_flag(manifest[col])

    # 排除原因按固定顺序记录；每张图像只计入首个原因。
    def exclusion_reason(row):
        if str(row.image_id).startswith("HESN_"):
            return "HESN_unresolved_geometry"
        if row.category == "normal":
            return "normal_outside_binary_task"
        if row.status != "checked":
            return "audit_not_checked"
        if row.category not in ("benign", "malignant"):
            return "unknown_category"
        if row.image_id in ("CHCO_000", "CHVI_005"):
            return "duplicate_with_discordant_masks"
        if row.Doppler_flag:
            return "doppler"
        if row.Combined_flag:
            return "combined"
        if float(row.lesion_components) != 1:
            return "multiple_or_empty_components"
        return "included"

    manifest["selection"] = manifest.apply(
        exclusion_reason, axis=1
    )
    manifest.to_csv(OUT / "selection_manifest.csv", index=False)
    flow = (
        manifest.groupby("selection", dropna=False)
        .size().rename("n").reset_index()
    )
    flow.to_csv(OUT / "selection_flow.csv", index=False)
    display(flow)

    cohort = (
        manifest.loc[manifest.selection == "included"]
        .sort_values("image_id").reset_index(drop=True).copy()
    )
    cohort["label"] = pd.to_numeric(
        cohort["label"], errors="raise"
    ).astype(int)

    if (
        len(cohort) != 207
        or int((cohort.label == 0).sum()) != 141
        or int((cohort.label == 1).sum()) != 66
    ):
        raise ValueError(
            "队列与预先固定的207张（141良性/66恶性）不一致。"
        )

    # 已发现CHCO/CHVI跨编号重复，保守地将这两个编号视为
    # 一个相关性聚类；不宣称已确认其为同一患者。
    cohort["cluster_id"] = cohort.patient_id.replace({
        "CHCO": "CHCO_CHVI_linked",
        "CHVI": "CHCO_CHVI_linked",
    })
    cohort.to_csv(OUT / "cohort_frozen.csv", index=False)

    save_json("evaluation_protocol.json", {
        "version": "BUS-UCLM-frozen-external-1",
        "n": 207,
        "benign": 141,
        "malignant": 66,
        "endpoint": "image-level benign/malignant classification",
        "roi": "provided reference masks; not automatic detection",
        "selection": "fixed before UCLM model predictions",
        "training": False,
        "recalibration": False,
        "threshold_tuning": False,
        "primary_model": "LR",
        "primary_comparison": "base_posterior minus base_missing",
        "bootstrap": "patient/linked-ID cluster percentile bootstrap",
        "bootstrap_replicates": BOOTSTRAPS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "threshold_caveat":
            "Source patient-mean thresholds applied to target images; "
            "95% target sensitivity is not guaranteed.",
        "ci_caveat":
            "Conditional on fixed models and selected cohort; "
            "secondary intervals are descriptive, unadjusted.",
        "extractor_sha256": EXTRACTOR_HASH,
        "manifest_sha256": MANIFEST_HASH,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__,
        },
    })

    print(
        f"冻结队列：{len(cohort)}张；"
        f"{cohort.patient_id.nunique()}个患者编号；"
        f"{cohort.cluster_id.nunique()}个统计聚类。",
        flush=True,
    )

    # 按文件名建索引，再用原审计SHA256确定文件。
    # 不会把同名Images与Masks混淆，也不会随便选第一份。
    wanted_names = {
        Path(str(p)).name
        for col in ("image_path", "mask_path")
        for p in cohort[col]
    }
    file_index = defaultdict(set)
    print("定位原始图像和掩膜……", flush=True)
    for root in (INPUT, WORK):
        if root.exists():
            for p in root.rglob("*"):
                if p.name in wanted_names and p.is_file():
                    file_index[p.name].add(p)

    hash_cache = {}

    def locate(saved_path, expected_hash):
        direct = Path(str(saved_path))
        candidates = set(file_index.get(direct.name, set()))
        if direct.is_file():
            candidates.add(direct)
        for p in sorted(candidates, key=str):
            if p not in hash_cache:
                hash_cache[p] = sha_file(p)
            if hash_cache[p] == expected_hash:
                return p
        raise FileNotFoundError(
            f"{direct.name}：没有找到与审计SHA256一致的文件。"
            "请确认原始BUS-UCLM数据集已添加并下载完成。"
        )

    feature_rows, quality_rows, file_rows = [], [], []
    for i, row in enumerate(cohort.itertuples(index=False), 1):
        ip = locate(row.image_path, row.image_sha256)
        mp = locate(row.mask_path, row.mask_sha256)

        with Image.open(ip) as im:
            gray = np.asarray(im.convert("L"), dtype=np.uint8)
            valid = np.asarray(im.convert("RGBA"))[:, :, 3] > 0

        with Image.open(mp) as im:
            rgb = np.asarray(im.convert("RGB"))
            alpha = np.asarray(im.convert("RGBA"))[:, :, 3] > 0

        green = np.all(rgb == [0, 255, 0], axis=2)
        red = np.all(rgb == [255, 0, 0], axis=2)
        black = np.all(rgb == [0, 0, 0], axis=2)
        if not np.all((green | red | black) | ~alpha):
            raise ValueError(f"{row.image_id}：发现未知掩膜颜色。")

        # 合并红/绿为二值ROI；颜色类别绝不作为模型特征。
        mask = (green | red) & alpha
        if gray.shape != mask.shape or (mask & ~valid).any():
            raise ValueError(f"{row.image_id}：掩膜几何异常。")

        scale = side / max(gray.shape)
        size = (
            max(1, round(gray.shape[1] * scale)),
            max(1, round(gray.shape[0] * scale)),
        )
        x = np.asarray(
            Image.fromarray(gray.astype(np.float32) / 255.0)
            .resize(size, Image.Resampling.BILINEAR)
        )

        def resize_binary(a):
            return np.asarray(
                Image.fromarray(a.astype("uint8"))
                .resize(size, Image.Resampling.NEAREST)
            ) > 0

        m, v = resize_binary(mask), resize_binary(valid)
        features, quality = extract_features(x, m, v)
        feature_rows.append({"image_id": row.image_id, **features})
        quality_rows.append({"image_id": row.image_id, **quality})
        file_rows.append({
            "image_id": row.image_id,
            "image_path": str(ip),
            "mask_path": str(mp),
            "image_sha256": row.image_sha256,
            "mask_sha256": row.mask_sha256,
        })
        if i % 50 == 0 or i == len(cohort):
            print(f"已提取特征：{i}/{len(cohort)}", flush=True)

    features = pd.DataFrame(feature_rows).set_index("image_id")
    quality = pd.DataFrame(quality_rows).set_index("image_id")
    features.to_csv(OUT / "features.csv")
    quality.to_csv(OUT / "quality.csv")
    pd.DataFrame(file_rows).to_csv(
        OUT / "resolved_files.csv", index=False
    )
    features = features.loc[cohort.image_id]
    if np.isinf(features.to_numpy(dtype=float)).any():
        raise ValueError("特征出现无穷值；停止，不自动修改。")

    predictions, thresholds = {}, {}
    frozen_rows = []
    print("加载原模型并预测……", flush=True)

    with threadpool_limits(limits=2):
        for filename, expected_hash in MODEL_HASHES.items():
            data = read_member(model_source, filename)
            if sha_bytes(data) != expected_hash:
                raise ValueError(f"{filename}：模型哈希不一致。")

            # 仅加载用户此前生成且哈希一致的模型文件。
            bundle = joblib.load(io.BytesIO(data))
            if bundle["feature_version"] != "V6-unified-features-1":
                raise ValueError("模型特征版本不一致。")

            key = filename.removesuffix(".joblib")
            columns = bundle["feature_columns"]
            X = features[columns]
            raw_parts = []
            for estimator in bundle["models"].values():
                if list(estimator.classes_) != [0, 1]:
                    raise ValueError("模型类别顺序异常。")
                raw_parts.append(estimator.predict_proba(X)[:, 1])
            raw = np.mean(raw_parts, axis=0)

            # 原先保存的Platt校准器，只预测，不重新拟合。
            calibrated = bundle["calibrator"].predict_proba(
                logit(np.clip(raw, 1e-6, 1 - 1e-6)).reshape(-1, 1)
            )[:, 1]
            if not np.isfinite(calibrated).all():
                raise ValueError("预测出现非有限值。")

            threshold = float(bundle["threshold"])
            predictions[key] = calibrated
            thresholds[key] = threshold
            frozen_rows.append(pd.DataFrame({
                "image_id": cohort.image_id.to_numpy(),
                "configuration": key,
                "raw_probability": raw,
                "probability": calibrated,
                "threshold": threshold,
                "predicted_label": (calibrated >= threshold).astype(int),
            }))
            print("完成：", key, flush=True)

    # 先保存固定预测，再计算成绩。
    pred = pd.concat(frozen_rows, ignore_index=True)
    pred.to_csv(OUT / "predictions_frozen.csv", index=False)
    pred.merge(
        cohort[["image_id", "patient_id", "cluster_id", "label"]],
        on="image_id", validate="many_to_one",
    ).to_csv(OUT / "predictions_with_labels.csv", index=False)
    save_json("model_hashes.json", MODEL_HASHES)

    y = cohort.label.to_numpy()
    metric_names = ["auc", "brier", "sensitivity", "specificity"]

    def metric_values(yv, pv, threshold):
        decision = pv >= threshold
        positive, negative = yv == 1, yv == 0
        return np.array([
            roc_auc_score(yv, pv),
            np.mean((pv - yv) ** 2),
            decision[positive].mean(),
            (~decision[negative]).mean(),
        ], dtype=float)

    # 同一轮抽样用于所有模型，保证差值为配对比较。
    cluster_values = cohort.cluster_id.to_numpy()
    clusters = np.unique(cluster_values)
    blocks = [
        np.flatnonzero(cluster_values == name) for name in clusters
    ]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = []
    for _ in range(BOOTSTRAPS):
        chosen = rng.integers(0, len(blocks), size=len(blocks))
        indices = np.concatenate([blocks[j] for j in chosen])
        if np.unique(y[indices]).size == 2:
            draws.append(indices)

    if len(draws) < 0.9 * BOOTSTRAPS:
        raise RuntimeError("有效聚类bootstrap次数不足。")

    bootstrap_results = {}
    metric_rows = []
    print("计算患者聚类置信区间……", flush=True)
    for key, p in predictions.items():
        threshold = thresholds[key]
        point = metric_values(y, p, threshold)
        boot = np.array([
            metric_values(y[ix], p[ix], threshold) for ix in draws
        ])
        bootstrap_results[key] = boot
        low, high = np.quantile(boot, [0.025, 0.975], axis=0)
        record = {
            "configuration": key,
            "n_images": len(y),
            "n_patient_ids": cohort.patient_id.nunique(),
            "n_clusters": len(clusters),
            "threshold": threshold,
            "missed_malignant": int(((y == 1) & (p < threshold)).sum()),
            "false_positive": int(((y == 0) & (p >= threshold)).sum()),
            "valid_bootstraps": len(draws),
        }
        for j, name in enumerate(metric_names):
            record[name] = point[j]
            record[name + "_low"] = low[j]
            record[name + "_high"] = high[j]
        metric_rows.append(record)

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(OUT / "external_metrics.csv", index=False)

    difference_rows = []
    for model in ("LR", "ensemble"):
        for before, after, comparison in [
            ("base", "base_missing", "missing_indicator_addition"),
            ("base_missing", "base_posterior", "measurement_addition"),
            ("base", "base_posterior", "total_addition"),
        ]:
            a, b = before + "_" + model, after + "_" + model
            point_a = metric_values(y, predictions[a], thresholds[a])
            point_b = metric_values(y, predictions[b], thresholds[b])
            delta_boot = bootstrap_results[b] - bootstrap_results[a]
            low, high = np.quantile(
                delta_boot, [0.025, 0.975], axis=0
            )
            for j, name in enumerate(metric_names):
                difference_rows.append({
                    "model": model,
                    "comparison": comparison,
                    "metric": name,
                    "after_minus_before": point_b[j] - point_a[j],
                    "low": low[j],
                    "high": high[j],
                    "note": (
                        "Negative Brier difference favors after; "
                        "other positive differences favor after. "
                        "Sensitivity/specificity use each frozen threshold."
                    ),
                })
    differences = pd.DataFrame(difference_rows)
    differences.to_csv(
        OUT / "paired_metric_differences.csv", index=False
    )

    # 预先指定的描述性分层，不用于调整模型。
    usable = (
        quality.loc[cohort.image_id, "posterior_usable"]
        .to_numpy(dtype=bool)
    )
    marked = cohort.Marks_flag.to_numpy(dtype=bool)
    subgroup_rows = []
    for group_name, keep in {
        "posterior_measurable": usable,
        "posterior_unmeasurable": ~usable,
        "marks_present": marked,
        "marks_absent": ~marked,
    }.items():
        for key, p in predictions.items():
            result = {
                "subgroup": group_name,
                "configuration": key,
                "n": int(keep.sum()),
                "malignant_n": int(y[keep].sum()),
                "n_clusters": len(np.unique(cluster_values[keep])),
            }
            if np.unique(y[keep]).size == 2:
                result.update(dict(zip(
                    metric_names,
                    metric_values(y[keep], p[keep], thresholds[key]),
                )))
            subgroup_rows.append(result)
    pd.DataFrame(subgroup_rows).to_csv(
        OUT / "descriptive_subgroups.csv", index=False
    )

    save_json("summary.json", {
        "completed": True,
        "n_images": len(cohort),
        "benign": int((y == 0).sum()),
        "malignant": int((y == 1).sum()),
        "n_patient_ids": int(cohort.patient_id.nunique()),
        "n_clusters": len(clusters),
        "posterior_measurable": int(usable.sum()),
        "models_evaluated": len(predictions),
        "models_retrained": False,
        "thresholds_changed": False,
        "valid_bootstraps": len(draws),
    })

    # 保存当前单元代码，便于复现。
    try:
        source = get_ipython().history_manager.input_hist_raw[-1]
        (OUT / "executed_cell.py").write_text(
            source, encoding="utf-8"
        )
    except Exception:
        pass

    print("\n第二外部验证完成：")
    display(metrics[[
        "configuration", "n_images", "auc", "auc_low", "auc_high",
        "brier", "sensitivity", "specificity",
        "missed_malignant", "false_positive",
    ]])
    print("\n配对AUC差值：")
    display(differences[differences.metric == "auc"])

    archive = shutil.make_archive(str(OUT), "zip", root_dir=OUT)
    print("\n结果包：", archive)
    print("请下载结果ZIP发回。此次没有训练模型或调整阈值。")
    display(FileLink(archive))


main()
# ======================= 代码结束 =======================